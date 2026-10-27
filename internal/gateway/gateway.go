// Package gateway routes OpenAI-compatible chat requests across LLM providers with failover,
// circuit breaking, per-key auth and rate limits, response caching and metrics.
package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/wshuv-o/llm-gateway/internal/api"
	"github.com/wshuv-o/llm-gateway/internal/provider"
)

type Gateway struct {
	cfg       *Config
	routes    map[string]*route
	keys      map[string]*key // by SHA-256 of the client key
	upstreams []*upstream
	cache     *cache // nil when disabled
	flights   flights
	metrics   *metrics
	log       *slog.Logger
	now       func() time.Time
}

type upstream struct {
	p       provider.Provider
	br      *breaker
	timeout time.Duration
}

type target struct {
	up    *upstream
	model string
}

type route struct {
	targets    []target
	roundRobin bool
	next       atomic.Uint64
}

// order is the sequence to try targets in: as configured (priority), or rotated per request.
func (r *route) order() []target {
	if !r.roundRobin {
		return r.targets
	}
	n := len(r.targets)
	start := int(r.next.Add(1)-1) % n
	return append(r.targets[start:n:n], r.targets[:start]...)
}

type key struct {
	cfg KeyConfig
	lim *limiter
}

func New(cfg *Config, log *slog.Logger) *Gateway {
	g := &Gateway{cfg: cfg, routes: map[string]*route{}, keys: map[string]*key{}, metrics: newMetrics(), log: log, now: time.Now}
	clock := func() time.Time { return g.now() } // tests swap g.now
	client := &http.Client{Transport: &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		MaxIdleConnsPerHost: 64,
		IdleConnTimeout:     90 * time.Second,
		ForceAttemptHTTP2:   true,
	}} // no client-wide timeout: streams can run for minutes; each call gets its own deadline

	byName := map[string]*upstream{}
	for _, pc := range cfg.Providers {
		apiKey := os.Getenv(pc.APIKeyEnv)
		var p provider.Provider
		switch pc.Kind {
		case "anthropic":
			p = provider.NewAnthropic(pc.Name, pc.BaseURL, apiKey, pc.DefaultMaxTokens, client)
		default:
			p = provider.NewOpenAI(pc.Name, pc.BaseURL, apiKey, client)
		}
		up := &upstream{p, newBreaker(cfg.Breaker.Failures, seconds(cfg.Breaker.CooldownSeconds), clock), seconds(pc.TimeoutSeconds)}
		byName[pc.Name] = up
		g.upstreams = append(g.upstreams, up)
	}
	for alias, rc := range cfg.Routes {
		r := &route{roundRobin: rc.Strategy == "round_robin"}
		for _, t := range rc.Targets {
			r.targets = append(r.targets, target{byName[t.Provider], t.Model})
		}
		g.routes[alias] = r
	}
	for _, kc := range cfg.Keys {
		g.keys[strings.ToLower(kc.SHA256)] = &key{kc, newLimiter(kc.RPM, kc.TPM, time.Now())}
	}
	if cfg.Cache.MaxEntries > 0 {
		g.cache = newCache(cfg.Cache.MaxEntries, time.Duration(cfg.Cache.TTLSeconds)*time.Second, clock)
	}
	return g
}

func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

type attempt struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
	Outcome  string `json:"outcome"` // ok | error | client_error | cancelled | breaker_open | unsupported
	Status   int    `json:"status,omitempty"`
	Ms       int64  `json:"ms"`
}

// routeError is a failure the gateway itself reports (as opposed to an upstream rejection).
type routeError struct {
	status     int
	code, msg  string
	retryAfter time.Duration
}

func (e *routeError) Error() string { return e.msg }

// tryTargets calls fn on each target in turn until one succeeds. It moves on after outages,
// overloads and timeouts, but not after a malformed request (that would fail everywhere) or
// after the client has gone away.
func (g *Gateway) tryTargets(ctx context.Context, rt *route, req *api.ChatRequest,
	fn func(context.Context, target) error) (*target, []attempt, error) {
	var attempts []attempt
	var lastErr error
	var unsupported []string
	for _, t := range rt.order() {
		a := attempt{Provider: t.up.p.Name(), Model: t.model}
		if err := t.up.p.Supports(req); err != nil {
			a.Outcome = "unsupported"
			attempts = append(attempts, a)
			unsupported = append(unsupported, fmt.Sprintf("%s: %v", a.Provider, err))
			continue
		}
		if !t.up.br.allow() {
			a.Outcome = "breaker_open"
			attempts = append(attempts, a)
			continue
		}
		start := time.Now()
		err := fn(ctx, t)
		a.Ms = time.Since(start).Milliseconds()
		var pe *provider.Error
		switch {
		case err == nil:
			t.up.br.success()
			a.Outcome = "ok"
		case ctx.Err() != nil:
			t.up.br.abandon()
			a.Outcome = "cancelled"
		case errors.As(err, &pe) && !pe.Failover():
			t.up.br.success() // the provider answered; the request was bad
			a.Outcome, a.Status = "client_error", pe.Status
		default:
			var retryAfter time.Duration
			if errors.As(err, &pe) {
				a.Status, retryAfter = pe.Status, pe.RetryAfter
			}
			t.up.br.failure(retryAfter)
			a.Outcome = "error"
			lastErr = err
		}
		attempts = append(attempts, a)
		g.metrics.upstream.add(1, a.Provider, a.Outcome)
		if a.Outcome != "error" {
			if err == nil {
				return &t, attempts, nil
			}
			return nil, attempts, err
		}
		g.log.Warn("upstream failed, trying next target", "provider", a.Provider, "model", a.Model, "error", err)
	}
	switch {
	case lastErr != nil:
		return nil, attempts, &routeError{status: http.StatusBadGateway, code: "upstream_error",
			msg: "all providers for this model failed; last error: " + lastErr.Error()}
	case len(unsupported) == len(rt.targets):
		return nil, attempts, &routeError{status: http.StatusBadRequest, code: "unsupported_request",
			msg: "no provider for this model can serve this request (" + strings.Join(unsupported, "; ") + ")"}
	default:
		return nil, attempts, &routeError{status: http.StatusServiceUnavailable, code: "providers_unavailable",
			msg: "every provider for this model is failing; retry shortly", retryAfter: seconds(g.cfg.Breaker.CooldownSeconds)}
	}
}
