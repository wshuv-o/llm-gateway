package gateway

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const (
	alphaKey = "sk-test-alpha" // may use every route
	betaKey  = "sk-test-beta"  // may use route "a" only, max_tokens 100
)

func sha(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// fake is an upstream provider whose behaviour each test scripts.
type fake struct {
	*httptest.Server
	calls atomic.Int32
	mu    sync.Mutex
	auth  string
	body  map[string]any
	h     func(w http.ResponseWriter, r *http.Request, body map[string]any)
}

func newFake(t *testing.T, h func(w http.ResponseWriter, r *http.Request, body map[string]any)) *fake {
	f := &fake{h: h}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.calls.Add(1)
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		f.auth, f.body = r.Header.Get("Authorization")+r.Header.Get("x-api-key"), body
		f.mu.Unlock()
		f.h(w, r, body)
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fake) lastBody() map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.body
}

func completion(text string) func(http.ResponseWriter, *http.Request, map[string]any) {
	return func(w http.ResponseWriter, _ *http.Request, _ map[string]any) {
		fmt.Fprintf(w, `{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`, text)
	}
}

func failing(status int) func(http.ResponseWriter, *http.Request, map[string]any) {
	return func(w http.ResponseWriter, _ *http.Request, _ map[string]any) {
		w.WriteHeader(status)
		io.WriteString(w, `{"error":{"message":"upstream says no"}}`)
	}
}

func sseStream(parts ...string) func(http.ResponseWriter, *http.Request, map[string]any) {
	return func(w http.ResponseWriter, _ *http.Request, body map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, p := range parts {
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":%q}}],\"usage\":null}\n\n", p)
			w.(http.Flusher).Flush()
		}
		if opts, _ := body["stream_options"].(map[string]any); opts["include_usage"] == true {
			io.WriteString(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n")
		}
		io.WriteString(w, "data: [DONE]\n\n")
	}
}

type env struct {
	g   *Gateway
	srv *httptest.Server
	now time.Time
}

// setup builds a gateway with providers p1, p2... pointing at the fakes, and routes
// "a" (all fakes in priority order) and "b" (the last fake only).
func setup(t *testing.T, mutate func(*Config), fakes ...*fake) *env {
	t.Setenv("FAKE_PROVIDER_KEY", "provider-secret")
	cfg := &Config{Routes: map[string]RouteConfig{}, Cache: CacheConfig{MaxEntries: 100}}
	var a RouteConfig
	for i, f := range fakes {
		name := fmt.Sprintf("p%d", i+1)
		cfg.Providers = append(cfg.Providers, ProviderConfig{Name: name, Kind: "openai", BaseURL: f.URL, APIKeyEnv: "FAKE_PROVIDER_KEY", TimeoutSeconds: 2})
		a.Targets = append(a.Targets, Target{name, "model-" + name})
	}
	cfg.Routes["a"] = a
	cfg.Routes["b"] = RouteConfig{Targets: a.Targets[len(a.Targets)-1:]}
	cfg.Keys = []KeyConfig{
		{ID: "alpha", SHA256: sha(alphaKey), Routes: []string{"*"}},
		{ID: "beta", SHA256: sha(betaKey), Routes: []string{"a"}, MaxTokens: 100},
	}
	if mutate != nil {
		mutate(cfg)
	}
	cfg.defaults()
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	e := &env{now: time.Unix(1_800_000_000, 0)}
	e.g = New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	e.g.now = func() time.Time { return e.now }
	e.srv = httptest.NewServer(e.g.Handler())
	t.Cleanup(e.srv.Close)
	return e
}

func (e *env) post(t *testing.T, key, body string, header ...string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest("POST", e.srv.URL+"/v1/chat/completions", strings.NewReader(body))
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	for i := 0; i+1 < len(header); i += 2 {
		req.Header.Set(header[i], header[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func (e *env) metrics() string {
	rec := httptest.NewRecorder()
	e.g.MetricsHandler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	return rec.Body.String()
}

const hello = `{"model":"a","messages":[{"role":"user","content":"hi"}]}`

func readAll(t *testing.T, r *http.Response) string {
	b, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func errorCode(t *testing.T, r *http.Response) string {
	var b struct {
		Error struct{ Code string } `json:"error"`
	}
	json.NewDecoder(r.Body).Decode(&b)
	return b.Error.Code
}

// --- routing and failover ---

func TestFailsOverToNextProviderOnOutage(t *testing.T) {
	p1, p2 := newFake(t, failing(503)), newFake(t, completion("from p2"))
	e := setup(t, nil, p1, p2)
	resp := e.post(t, alphaKey, hello)
	if resp.StatusCode != 200 || resp.Header.Get("X-Gateway-Provider") != "p2" {
		t.Fatalf("status %d provider %q", resp.StatusCode, resp.Header.Get("X-Gateway-Provider"))
	}
	if !strings.Contains(readAll(t, resp), "from p2") || p1.calls.Load() != 1 {
		t.Fatal("expected one failed call to p1, then p2's answer")
	}
}

func TestBadRequestIsNotRetriedElsewhere(t *testing.T) {
	p1, p2 := newFake(t, failing(400)), newFake(t, completion("x"))
	e := setup(t, nil, p1, p2)
	resp := e.post(t, alphaKey, hello)
	if resp.StatusCode != 400 || p2.calls.Load() != 0 {
		t.Fatalf("a malformed request fails everywhere; got %d and %d calls to p2", resp.StatusCode, p2.calls.Load())
	}
}

func TestSlowProviderTimesOutAndFailsOver(t *testing.T) {
	slow := newFake(t, func(w http.ResponseWriter, r *http.Request, _ map[string]any) {
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	})
	e := setup(t, func(c *Config) { c.Providers[0].TimeoutSeconds = 0.2 }, slow, newFake(t, completion("fast")))
	start := time.Now()
	resp := e.post(t, alphaKey, hello)
	if resp.StatusCode != 200 || time.Since(start) > 2*time.Second {
		t.Fatalf("status %d after %v", resp.StatusCode, time.Since(start))
	}
}

func TestAllProvidersFailing(t *testing.T) {
	e := setup(t, nil, newFake(t, failing(500)), newFake(t, failing(502)))
	resp := e.post(t, alphaKey, hello)
	if resp.StatusCode != 502 || errorCode(t, resp) != "upstream_error" {
		t.Fatalf("got %d", resp.StatusCode)
	}
}

func TestCircuitBreakerSkipsFailingProviderThenProbes(t *testing.T) {
	var healthy atomic.Bool
	p1 := newFake(t, func(w http.ResponseWriter, r *http.Request, b map[string]any) {
		if healthy.Load() {
			completion("p1 back")(w, r, b)
			return
		}
		failing(500)(w, r, b)
	})
	p2 := newFake(t, completion("p2"))
	e := setup(t, func(c *Config) { c.Breaker = BreakerConfig{Failures: 2, CooldownSeconds: 30} }, p1, p2)

	for range 4 {
		if r := e.post(t, alphaKey, hello); r.StatusCode != 200 {
			t.Fatal(r.StatusCode)
		}
	}
	if p1.calls.Load() != 2 {
		t.Fatalf("after 2 failures the breaker should stop calls to p1; it got %d", p1.calls.Load())
	}
	if !strings.Contains(e.metrics(), `gateway_breaker_state{provider="p1"} 1`) {
		t.Fatal("breaker should report open")
	}

	e.now = e.now.Add(31 * time.Second) // cooldown over: one probe, which fails, so it opens again
	e.post(t, alphaKey, hello)
	e.post(t, alphaKey, hello)
	if p1.calls.Load() != 3 {
		t.Fatalf("expected exactly one probe, p1 calls = %d", p1.calls.Load())
	}

	healthy.Store(true)
	e.now = e.now.Add(31 * time.Second)
	if r := e.post(t, alphaKey, hello); r.Header.Get("X-Gateway-Provider") != "p1" {
		t.Fatal("a successful probe should close the breaker and serve from p1")
	}
	if r := e.post(t, alphaKey, hello); r.Header.Get("X-Gateway-Provider") != "p1" {
		t.Fatal("closed breaker should keep serving from p1")
	}
}

func TestAllBreakersOpenReturns503WithRetryAfter(t *testing.T) {
	e := setup(t, func(c *Config) { c.Breaker = BreakerConfig{Failures: 1, CooldownSeconds: 20} }, newFake(t, failing(500)))
	e.post(t, alphaKey, hello)
	resp := e.post(t, alphaKey, hello)
	if resp.StatusCode != 503 || resp.Header.Get("Retry-After") != "20" {
		t.Fatalf("got %d, Retry-After %q", resp.StatusCode, resp.Header.Get("Retry-After"))
	}
}

func TestUpstreamRetryAfterExtendsCooldown(t *testing.T) {
	limited := newFake(t, func(w http.ResponseWriter, _ *http.Request, _ map[string]any) {
		w.Header().Set("Retry-After", "120")
		w.WriteHeader(429)
	})
	e := setup(t, func(c *Config) { c.Breaker = BreakerConfig{Failures: 1, CooldownSeconds: 10} }, limited, newFake(t, completion("p2")))
	e.post(t, alphaKey, hello)
	e.now = e.now.Add(60 * time.Second) // past our cooldown, inside the provider's Retry-After
	e.post(t, alphaKey, hello)
	if limited.calls.Load() != 1 {
		t.Fatal("the gateway must respect the provider's Retry-After")
	}
}

func TestCancelledProbeReleasesBreaker(t *testing.T) {
	b := newBreaker(1, time.Second, time.Now)
	b.failure(0)
	b.openUntil = time.Now().Add(-time.Millisecond)
	if !b.allow() || b.allow() {
		t.Fatal("half-open admits exactly one probe")
	}
	b.abandon()
	if !b.allow() {
		t.Fatal("an abandoned probe must not leave the breaker stuck half-open")
	}
}

func TestRoundRobinSpreadsLoad(t *testing.T) {
	p1, p2 := newFake(t, completion("1")), newFake(t, completion("2"))
	e := setup(t, func(c *Config) { r := c.Routes["a"]; r.Strategy = "round_robin"; c.Routes["a"] = r }, p1, p2)
	for range 4 {
		e.post(t, alphaKey, hello)
	}
	if p1.calls.Load() != 2 || p2.calls.Load() != 2 {
		t.Fatalf("p1 %d, p2 %d", p1.calls.Load(), p2.calls.Load())
	}
}

// --- auth and policy ---

func TestAuthAndPolicy(t *testing.T) {
	up := newFake(t, completion("ok"))
	e := setup(t, nil, up)
	cases := []struct {
		name, key, body string
		status          int
		code            string
	}{
		{"no key", "", hello, 401, "invalid_api_key"},
		{"wrong key", "sk-nope", hello, 401, "invalid_api_key"},
		{"route not granted", betaKey, `{"model":"b","messages":[{"role":"user","content":"hi"}]}`, 403, "model_not_allowed"},
		{"unknown model", alphaKey, `{"model":"gpt-9","messages":[{"role":"user","content":"hi"}]}`, 404, "model_not_found"},
		{"max_tokens above key cap", betaKey, `{"model":"a","max_tokens":500,"messages":[{"role":"user","content":"hi"}]}`, 400, "max_tokens_exceeded"},
		{"no messages", alphaKey, `{"model":"a","messages":[]}`, 400, "invalid_request"},
		{"not json", alphaKey, `nope`, 400, "invalid_request"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := e.post(t, c.key, c.body)
			if resp.StatusCode != c.status || errorCode(t, resp) != c.code {
				t.Fatalf("got %d", resp.StatusCode)
			}
		})
	}
	if up.calls.Load() != 0 {
		t.Fatal("rejected requests must never reach a provider")
	}
}

func TestClientKeyIsNeverForwardedAndCapIsInjected(t *testing.T) {
	up := newFake(t, completion("ok"))
	e := setup(t, nil, up)
	e.post(t, betaKey, hello)
	if up.auth != "Bearer provider-secret" {
		t.Fatalf("upstream saw %q", up.auth)
	}
	b := up.lastBody()
	if b["max_tokens"] != float64(100) || b["model"] != "model-p1" {
		t.Fatalf("upstream body %v", b)
	}
}

func TestBodyTooLarge(t *testing.T) {
	e := setup(t, func(c *Config) { c.MaxBodyBytes = 100 }, newFake(t, completion("x")))
	resp := e.post(t, alphaKey, `{"model":"a","messages":[{"role":"user","content":"`+strings.Repeat("x", 200)+`"}]}`)
	if resp.StatusCode != 413 {
		t.Fatal(resp.StatusCode)
	}
}

func TestModelsListsOnlyGrantedRoutes(t *testing.T) {
	e := setup(t, nil, newFake(t, completion("x")))
	req, _ := http.NewRequest("GET", e.srv.URL+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+betaKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if body := readAll(t, resp); !strings.Contains(body, `"id":"a"`) || strings.Contains(body, `"id":"b"`) {
		t.Fatal(body)
	}
}

// --- rate limits ---

func TestRequestsPerMinute(t *testing.T) {
	e := setup(t, func(c *Config) { c.Keys[0].RPM = 2 }, newFake(t, completion("x")))
	e.post(t, alphaKey, hello)
	if r := e.post(t, alphaKey, hello); r.Header.Get("X-RateLimit-Remaining-Requests") != "0" {
		t.Fatal("remaining header")
	}
	resp := e.post(t, alphaKey, hello)
	if resp.StatusCode != 429 || resp.Header.Get("Retry-After") == "" {
		t.Fatalf("got %d", resp.StatusCode)
	}
	if !strings.Contains(e.metrics(), `gateway_ratelimited_total{key="alpha",limit="requests"} 1`) {
		t.Fatal("rate limit metric")
	}
}

func TestTokenReservationIsSettledWithActualUsage(t *testing.T) {
	up := newFake(t, completion("x")) // reports 15 tokens
	e := setup(t, func(c *Config) { c.Keys[0].TPM = 1000 }, up)
	// each request reserves ~600 tokens; 15 are used, so the rest is refunded and the next one fits
	body := `{"model":"a","max_tokens":600,"messages":[{"role":"user","content":"hi"}]}`
	for i := range 3 {
		if r := e.post(t, alphaKey, body); r.StatusCode != 200 {
			t.Fatalf("request %d: %d", i, r.StatusCode)
		}
	}
}

func TestTokenUnderestimateBecomesDebt(t *testing.T) {
	up := newFake(t, func(w http.ResponseWriter, _ *http.Request, _ map[string]any) {
		io.WriteString(w, `{"choices":[],"usage":{"prompt_tokens":4000,"completion_tokens":1000,"total_tokens":5000}}`)
	})
	e := setup(t, func(c *Config) { c.Keys[0].TPM = 1000 }, up)
	e.post(t, alphaKey, hello)
	resp := e.post(t, alphaKey, hello)
	if resp.StatusCode != 429 || errorCode(t, resp) != "rate_limit_exceeded" {
		t.Fatalf("a key that used 5x its budget must wait; got %d", resp.StatusCode)
	}
}

// --- cache ---

func TestCacheServesDeterministicRequestsPerKey(t *testing.T) {
	up := newFake(t, completion("cached answer"))
	e := setup(t, func(c *Config) { c.Keys[1].Routes = []string{"*"}; c.Keys[1].MaxTokens = 0 }, up)
	det := `{"model":"a","temperature":0,"messages":[{"role":"user","content":"2+2?"}]}`
	if r := e.post(t, alphaKey, det); r.Header.Get("X-Gateway-Cache") != "miss" {
		t.Fatal("first call is a miss")
	}
	r := e.post(t, alphaKey, det)
	if r.Header.Get("X-Gateway-Cache") != "hit" || !strings.Contains(readAll(t, r), "cached answer") || up.calls.Load() != 1 {
		t.Fatal("second identical call should be served from cache")
	}
	if r := e.post(t, betaKey, det); r.Header.Get("X-Gateway-Cache") != "miss" {
		t.Fatal("cache entries must not be shared between keys")
	}
	e.post(t, alphaKey, `{"model":"a","temperature":0.7,"messages":[{"role":"user","content":"2+2?"}]}`)
	e.post(t, alphaKey, det, "Cache-Control", "no-store")
	if up.calls.Load() != 4 {
		t.Fatalf("non-deterministic and no-store requests bypass the cache; calls = %d", up.calls.Load())
	}
}

func TestCacheExpires(t *testing.T) {
	up := newFake(t, completion("x"))
	e := setup(t, func(c *Config) { c.Cache.TTLSeconds = 60 }, up)
	det := `{"model":"a","temperature":0,"messages":[{"role":"user","content":"q"}]}`
	e.post(t, alphaKey, det)
	e.now = e.now.Add(61 * time.Second)
	e.post(t, alphaKey, det)
	if up.calls.Load() != 2 {
		t.Fatal("expired entry should be refetched")
	}
}

func TestConcurrentIdenticalRequestsShareOneUpstreamCall(t *testing.T) {
	up := newFake(t, func(w http.ResponseWriter, r *http.Request, b map[string]any) {
		time.Sleep(300 * time.Millisecond)
		completion("once")(w, r, b)
	})
	e := setup(t, nil, up)
	det := `{"model":"a","temperature":0,"messages":[{"role":"user","content":"same"}]}`
	var wg sync.WaitGroup
	var ok atomic.Int32
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if r := e.post(t, alphaKey, det); r.StatusCode == 200 {
				ok.Add(1)
			}
		}()
	}
	wg.Wait()
	if ok.Load() != 10 || up.calls.Load() != 1 {
		t.Fatalf("%d ok, %d upstream calls", ok.Load(), up.calls.Load())
	}
}

func TestLRUEvictsOldest(t *testing.T) {
	c := newCache(2, time.Minute, time.Now)
	c.put("a", cached{})
	c.put("b", cached{})
	c.get("a")
	c.put("c", cached{})
	if _, ok := c.get("b"); ok {
		t.Fatal("b was least recently used")
	}
	if _, ok := c.get("a"); !ok {
		t.Fatal("a was used recently and should stay")
	}
}

// --- streaming ---

type sseResult struct {
	text   string
	events []string
}

func readSSE(t *testing.T, r *http.Response) sseResult {
	var out sseResult
	sc := bufio.NewScanner(r.Body)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		out.events = append(out.events, data)
		var c struct {
			Choices []struct {
				Delta struct{ Content string } `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &c) == nil && len(c.Choices) > 0 {
			out.text += c.Choices[0].Delta.Content
		}
	}
	return out
}

const streamReq = `{"model":"a","stream":true,"messages":[{"role":"user","content":"hi"}]}`

func TestStreamPassthroughCountsTokensButHidesUsageChunk(t *testing.T) {
	e := setup(t, nil, newFake(t, sseStream("Hel", "lo")))
	resp := e.post(t, alphaKey, streamReq)
	got := readSSE(t, resp)
	if got.text != "Hello" || got.events[len(got.events)-1] != "[DONE]" {
		t.Fatalf("%+v", got)
	}
	for _, ev := range got.events {
		if strings.Contains(ev, `"usage":{`) {
			t.Fatal("the client did not ask for usage; the gateway's own usage chunk must not leak")
		}
	}
	if !strings.Contains(e.metrics(), `gateway_tokens_total{key="alpha",provider="p1",kind="completion"} 3`) {
		t.Fatal("stream tokens must still be accounted")
	}
}

func TestStreamIncludesUsageWhenAsked(t *testing.T) {
	e := setup(t, nil, newFake(t, sseStream("x")))
	resp := e.post(t, alphaKey, `{"model":"a","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`)
	if got := readSSE(t, resp); !strings.Contains(strings.Join(got.events, "\n"), `"total_tokens":10`) {
		t.Fatal("usage chunk expected")
	}
}

func TestStreamFailsOverBeforeFirstChunk(t *testing.T) {
	p1, p2 := newFake(t, failing(529)), newFake(t, sseStream("from p2"))
	e := setup(t, nil, p1, p2)
	resp := e.post(t, alphaKey, streamReq)
	if got := readSSE(t, resp); got.text != "from p2" || resp.Header.Get("X-Gateway-Provider") != "p2" {
		t.Fatalf("%+v", got)
	}
}

func TestStreamFirstChunkTimeoutFailsOver(t *testing.T) {
	hang := newFake(t, func(w http.ResponseWriter, r *http.Request, _ map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.(http.Flusher).Flush() // headers sent, then nothing
		<-r.Context().Done()
	})
	e := setup(t, func(c *Config) { c.Providers[0].TimeoutSeconds = 0.2 }, hang, newFake(t, sseStream("ok")))
	if got := readSSE(t, e.post(t, alphaKey, streamReq)); got.text != "ok" {
		t.Fatalf("%+v", got)
	}
}

func TestMidStreamFailureEndsWithErrorAndNoFailover(t *testing.T) {
	broken := newFake(t, func(w http.ResponseWriter, _ *http.Request, _ map[string]any) {
		io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n")
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler) // connection drops mid-answer
	})
	p2 := newFake(t, sseStream("second answer"))
	e := setup(t, nil, broken, p2)
	got := readSSE(t, e.post(t, alphaKey, streamReq))
	last := got.events[len(got.events)-1]
	if got.text != "partial" || !strings.Contains(last, "upstream_stream_error") || p2.calls.Load() != 0 {
		t.Fatalf("after the first token, a failure must not splice in another provider's answer: %+v", got)
	}
}

func TestClientDisconnectCancelsUpstream(t *testing.T) {
	cancelled := make(chan struct{})
	up := newFake(t, func(w http.ResponseWriter, r *http.Request, _ map[string]any) {
		for i := 0; ; i++ {
			if _, err := fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%d \"}}]}\n\n", i); err != nil {
				break
			}
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
				close(cancelled)
				return
			case <-time.After(20 * time.Millisecond):
			}
		}
	})
	e := setup(t, nil, up)
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", e.srv.URL+"/v1/chat/completions", strings.NewReader(streamReq))
	req.Header.Set("Authorization", "Bearer "+alphaKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	bufio.NewReader(resp.Body).ReadString('\n')
	cancel()
	select {
	case <-cancelled:
	case <-time.After(3 * time.Second):
		t.Fatal("the upstream request should be cancelled when the client leaves")
	}
}

// --- capability-aware routing ---

func TestRequestsAProviderCantServeGoElsewhere(t *testing.T) {
	anth := newFake(t, completion("x"))
	oai := newFake(t, completion("tools ok"))
	e := setup(t, func(c *Config) { c.Providers[0].Kind = "anthropic" }, anth, oai)
	tools := `{"model":"a","tools":[{"type":"function","function":{"name":"f"}}],"messages":[{"role":"user","content":"hi"}]}`
	resp := e.post(t, alphaKey, tools)
	if resp.StatusCode != 200 || anth.calls.Load() != 0 {
		t.Fatalf("got %d, anthropic calls %d", resp.StatusCode, anth.calls.Load())
	}
	e2 := setup(t, func(c *Config) { c.Providers[0].Kind = "anthropic" }, anth)
	if r := e2.post(t, alphaKey, tools); r.StatusCode != 400 || errorCode(t, r) != "unsupported_request" {
		t.Fatalf("got %d", r.StatusCode)
	}
}

// --- config ---

func TestConfigValidation(t *testing.T) {
	t.Setenv("SET_KEY", "x")
	ok := `{"providers":[{"name":"p","kind":"openai","base_url":"https://x/v1","api_key_env":"SET_KEY"}],
	        "routes":{"r":{"targets":[{"provider":"p","model":"m"}]}},
	        "keys":[{"id":"k","sha256":"` + sha("k") + `","routes":["r"]}]}`
	if _, err := ParseConfig([]byte(ok)); err != nil {
		t.Fatal(err)
	}
	bad := map[string]string{
		"typo in a limit":  strings.Replace(ok, `"routes":["r"]`, `"routes":["r"],"rmp":5`, 1),
		"unknown provider": strings.Replace(ok, `"provider":"p"`, `"provider":"q"`, 1),
		"missing env var":  strings.Replace(ok, "SET_KEY", "UNSET_KEY_XYZ", 1),
		"bad hash":         strings.Replace(ok, sha("k"), "abc", 1),
		"unknown route":    strings.Replace(ok, `"routes":["r"]`, `"routes":["nope"]`, 1),
		"bad kind":         strings.Replace(ok, `"kind":"openai"`, `"kind":"gemini"`, 1),
	}
	for name, raw := range bad {
		if _, err := ParseConfig([]byte(raw)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

// --- overhead ---

func BenchmarkGatewayOverhead(b *testing.B) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}],"usage":{"total_tokens":3}}`)
	}))
	defer up.Close()
	cfg := &Config{
		Providers: []ProviderConfig{{Name: "p", Kind: "openai", BaseURL: up.URL, TimeoutSeconds: 5}},
		Routes:    map[string]RouteConfig{"a": {Targets: []Target{{"p", "m"}}}},
		Keys:      []KeyConfig{{ID: "k", SHA256: sha(alphaKey), Routes: []string{"*"}, RPM: 1 << 30, TPM: 1 << 30}},
	}
	cfg.defaults()
	g := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	srv := httptest.NewServer(g.Handler())
	defer srv.Close()

	run := func(b *testing.B, url string) {
		for b.Loop() {
			req, _ := http.NewRequest("POST", url, strings.NewReader(hello))
			req.Header.Set("Authorization", "Bearer "+alphaKey)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				b.Fatal(err)
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
	}
	b.Run("direct", func(b *testing.B) { run(b, up.URL+"/chat/completions") })
	b.Run("via-gateway", func(b *testing.B) { run(b, srv.URL+"/v1/chat/completions") })
}
