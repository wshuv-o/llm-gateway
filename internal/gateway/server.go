package gateway

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/wshuv-o/llm-gateway/internal/api"
	"github.com/wshuv-o/llm-gateway/internal/provider"
)

func (g *Gateway) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", g.chat)
	mux.HandleFunc("GET /v1/models", g.models)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, "ok\n") })
	return mux
}

// MetricsHandler serves Prometheus metrics. Run it on a separate, internal-only listener.
func (g *Gateway) MetricsHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		for _, up := range g.upstreams {
			g.metrics.breaker.set(float64(up.br.state()), up.p.Name())
		}
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		g.metrics.reg.write(w)
	})
}

// record is the audit line for one request. It never contains prompt or completion text.
type record struct {
	ID               string
	Key, Route       string
	Stream           bool
	Status           int
	Provider, Model  string
	Cache            string
	Attempts         []attempt
	PromptTokens     int
	CompletionTokens int
	Error            string
}

func (g *Gateway) authenticate(r *http.Request) *key {
	token, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !ok || token == "" {
		return nil
	}
	sum := sha256.Sum256([]byte(token))
	return g.keys[hex.EncodeToString(sum[:])]
}

func newID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func (g *Gateway) chat(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	rec := &record{ID: newID()}
	w.Header().Set("X-Request-Id", rec.ID)
	rec.Status = g.serveChat(w, r, rec)
	g.metrics.requests.add(1, rec.Route, strconv.Itoa(rec.Status))
	attempts, _ := json.Marshal(rec.Attempts)
	g.log.Info("request", "id", rec.ID, "key", rec.Key, "route", rec.Route, "stream", rec.Stream,
		"status", rec.Status, "provider", rec.Provider, "model", rec.Model, "cache", rec.Cache,
		"prompt_tokens", rec.PromptTokens, "completion_tokens", rec.CompletionTokens,
		"attempts", string(attempts), "error", rec.Error, "ms", time.Since(start).Milliseconds())
}

func fail(w http.ResponseWriter, status int, code, msg string) int {
	api.WriteError(w, status, code, msg)
	return status
}

func (g *Gateway) serveChat(w http.ResponseWriter, r *http.Request, rec *record) int {
	k := g.authenticate(r)
	if k == nil {
		return fail(w, http.StatusUnauthorized, "invalid_api_key", "missing or invalid API key")
	}
	rec.Key = k.cfg.ID

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, g.cfg.MaxBodyBytes))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return fail(w, http.StatusRequestEntityTooLarge, "request_too_large", fmt.Sprintf("request body is over %d bytes", tooBig.Limit))
		}
		return fail(w, http.StatusBadRequest, "invalid_request", "could not read request body")
	}
	req, err := api.ParseChatRequest(body)
	if err != nil {
		return fail(w, http.StatusBadRequest, "invalid_request", err.Error())
	}
	rec.Route, rec.Stream = req.Model, req.Stream

	rt := g.routes[req.Model]
	if rt == nil {
		rec.Route = "unknown"
		return fail(w, http.StatusNotFound, "model_not_found", fmt.Sprintf("model %q is not configured; GET /v1/models lists yours", req.Model))
	}
	if !k.cfg.allows(req.Model) {
		return fail(w, http.StatusForbidden, "model_not_allowed", fmt.Sprintf("this key may not use model %q", req.Model))
	}
	if limit := k.cfg.MaxTokens; limit > 0 {
		if m := req.MaxOutput(); m > limit {
			return fail(w, http.StatusBadRequest, "max_tokens_exceeded", fmt.Sprintf("max_tokens %d is above this key's limit of %d", m, limit))
		} else if m == 0 {
			req.SetMaxTokens(limit) // the cap holds even when the client doesn't ask for one
		}
	}

	out := req.MaxOutput()
	if out == 0 {
		out = 1024
	}
	res, wait, limit := k.lim.reserve(float64(req.PromptChars()/4+out), time.Now())
	if limit != "" {
		g.metrics.ratelimited.add(1, k.cfg.ID, limit)
		w.Header().Set("Retry-After", strconv.Itoa(int(math.Ceil(wait.Seconds()))))
		return fail(w, http.StatusTooManyRequests, "rate_limit_exceeded", fmt.Sprintf("this key is over its %s-per-minute limit", limit))
	}
	if res.RemainingRequests >= 0 {
		w.Header().Set("X-RateLimit-Remaining-Requests", strconv.Itoa(res.RemainingRequests))
	}
	if res.RemainingTokens >= 0 {
		w.Header().Set("X-RateLimit-Remaining-Tokens", strconv.Itoa(res.RemainingTokens))
	}
	used := -1 // tokens actually spent; unknown (-1) keeps the whole reservation charged
	defer func() {
		if used >= 0 {
			k.lim.settle(res.Tokens, used)
		}
	}()

	if req.Stream {
		return g.stream(w, r, k, rt, req, rec, &used)
	}
	return g.complete(w, r, k, rt, req, rec, &used)
}

// cacheable: only deterministic requests, and only when the client allows it.
func cacheable(req *api.ChatRequest, r *http.Request) bool {
	return req.Temperature != nil && *req.Temperature == 0 && (req.N == nil || *req.N == 1) &&
		!strings.Contains(r.Header.Get("Cache-Control"), "no-store")
}

// cacheKey is scoped to the API key: one tenant can never be served, or time, another's response.
func cacheKey(k *key, req *api.ChatRequest) string {
	canonical, _ := json.Marshal(req.Raw) // map keys marshal sorted
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00", k.cfg.ID)
	h.Write(canonical)
	return hex.EncodeToString(h.Sum(nil))
}

func (g *Gateway) countTokens(k *key, provider string, u api.Usage, rec *record) {
	rec.PromptTokens, rec.CompletionTokens = u.PromptTokens, u.CompletionTokens
	g.metrics.tokens.add(float64(u.PromptTokens), k.cfg.ID, provider, "prompt")
	g.metrics.tokens.add(float64(u.CompletionTokens), k.cfg.ID, provider, "completion")
}

func (g *Gateway) complete(w http.ResponseWriter, r *http.Request, k *key, rt *route, req *api.ChatRequest, rec *record, used *int) int {
	call := func(ctx context.Context) (cached, error) {
		var resp *provider.Response
		t, attempts, err := g.tryTargets(ctx, rt, req, func(ctx context.Context, t target) error {
			ctx, cancel := context.WithTimeout(ctx, t.up.timeout)
			defer cancel()
			start := time.Now()
			var err error
			if resp, err = t.up.p.Complete(ctx, t.model, req); err == nil {
				g.metrics.latency.observe(time.Since(start).Seconds(), t.up.p.Name())
			}
			return err
		})
		rec.Attempts = attempts
		if err != nil {
			return cached{}, err
		}
		return cached{resp, t.up.p.Name(), t.model}, nil
	}

	var c cached
	var err error
	if g.cache != nil && cacheable(req, r) {
		ck := cacheKey(k, req)
		if hit, ok := g.cache.get(ck); ok {
			g.metrics.cache.add(1, "hit")
			rec.Cache, *used = "hit", 0
			return g.writeCompletion(w, hit, rec)
		}
		var shared bool
		c, err, shared = g.flights.do(ck, func() (cached, error) {
			// Detached from this client: other identical requests may be waiting on this call.
			// The per-attempt timeouts still bound it.
			c, err := call(context.WithoutCancel(r.Context()))
			if err == nil {
				g.cache.put(ck, c)
			}
			return c, err
		})
		rec.Cache = map[bool]string{true: "shared", false: "miss"}[shared]
		g.metrics.cache.add(1, rec.Cache)
		if shared && err == nil {
			*used = 0 // one upstream call served everyone; its caller pays the tokens
			return g.writeCompletion(w, c, rec)
		}
	} else {
		c, err = call(r.Context())
	}
	if err != nil {
		*used = 0
		rec.Error = err.Error()
		return g.writeRouteError(w, err)
	}
	*used = c.resp.Usage.TotalTokens
	g.countTokens(k, c.provider, c.resp.Usage, rec)
	return g.writeCompletion(w, c, rec)
}

func (g *Gateway) writeCompletion(w http.ResponseWriter, c cached, rec *record) int {
	rec.Provider, rec.Model = c.provider, c.model
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Gateway-Provider", c.provider)
	if rec.Cache != "" {
		w.Header().Set("X-Gateway-Cache", rec.Cache)
	}
	w.WriteHeader(http.StatusOK)
	w.Write(c.resp.Body)
	return http.StatusOK
}

func (g *Gateway) writeRouteError(w http.ResponseWriter, err error) int {
	var re *routeError
	var pe *provider.Error
	switch {
	case errors.As(err, &re):
		if re.retryAfter > 0 {
			w.Header().Set("Retry-After", strconv.Itoa(int(math.Ceil(re.retryAfter.Seconds()))))
		}
		return fail(w, re.status, re.code, re.msg)
	case errors.As(err, &pe) && !pe.Failover():
		return fail(w, pe.Status, "upstream_rejected", pe.Message)
	case errors.Is(err, context.Canceled):
		return 499 // client closed the request; nobody is listening
	}
	return fail(w, http.StatusBadGateway, "upstream_error", err.Error())
}

// stream proxies a streamed completion. Failover is possible until the first chunk reaches the
// client; after that, retrying elsewhere would send the client a second, different answer, so a
// mid-stream failure ends the stream with an error event instead.
func (g *Gateway) stream(w http.ResponseWriter, r *http.Request, k *key, rt *route, req *api.ChatRequest, rec *record, used *int) int {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return fail(w, http.StatusInternalServerError, "streaming_unsupported", "streaming is not supported by this server")
	}
	var s provider.Stream
	var first provider.Chunk
	var stop context.CancelFunc
	t, attempts, err := g.tryTargets(r.Context(), rt, req, func(ctx context.Context, t target) error {
		// The timeout covers the wait for the first chunk only; a long answer may stream for minutes.
		ctx, cancel := context.WithCancel(ctx)
		timer := time.AfterFunc(t.up.timeout, cancel)
		start := time.Now()
		st, err := t.up.p.Stream(ctx, t.model, req)
		if err == nil {
			if first, err = firstChunk(st); err != nil {
				st.Close()
			}
		}
		timer.Stop()
		if err != nil {
			cancel()
			return err
		}
		g.metrics.ttft.observe(time.Since(start).Seconds(), t.up.p.Name())
		s, stop = st, cancel
		return nil
	})
	rec.Attempts = attempts
	if err != nil {
		*used = 0
		rec.Error = err.Error()
		return g.writeRouteError(w, err)
	}
	defer stop()
	defer s.Close()
	rec.Provider, rec.Model = t.up.p.Name(), t.model

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	h.Set("X-Gateway-Provider", rec.Provider)
	w.WriteHeader(http.StatusOK)
	send := func(data []byte) bool {
		_, err := fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return err == nil
	}

	var usage *api.Usage
	chunk := first
	for {
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		if chunk.Data != nil && !send(chunk.Data) {
			rec.Error = "client disconnected"
			break
		}
		chunk, err = s.Next()
		if err == io.EOF {
			send([]byte("[DONE]"))
			break
		}
		if err != nil {
			if r.Context().Err() != nil {
				rec.Error = "client disconnected"
				break
			}
			rec.Error = err.Error()
			b, _ := json.Marshal(map[string]any{"error": map[string]string{
				"message": "the upstream stream failed: " + err.Error(), "type": "api_error", "code": "upstream_stream_error"}})
			send(b)
			break
		}
	}
	if usage != nil {
		*used = usage.TotalTokens
		g.countTokens(k, rec.Provider, *usage, rec)
	}
	return http.StatusOK
}

// firstChunk reads until the first chunk meant for the client: the commit point for failover.
func firstChunk(s provider.Stream) (provider.Chunk, error) {
	for {
		c, err := s.Next()
		if err == io.EOF {
			return c, io.ErrUnexpectedEOF
		}
		if err != nil || c.Data != nil {
			return c, err
		}
	}
}

func (g *Gateway) models(w http.ResponseWriter, r *http.Request) {
	k := g.authenticate(r)
	if k == nil {
		fail(w, http.StatusUnauthorized, "invalid_api_key", "missing or invalid API key")
		return
	}
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	var names []string
	for name := range g.routes {
		if k.cfg.allows(name) {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	data := make([]model, 0, len(names))
	for _, n := range names {
		data = append(data, model{n, "model", "llm-gateway"})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}
