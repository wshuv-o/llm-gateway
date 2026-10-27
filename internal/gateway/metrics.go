package gateway

import (
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"
)

// A minimal Prometheus text-format registry, to keep the gateway dependency-free.

type series struct {
	labels []string
	value  float64
	counts []uint64 // histograms only
	sum    float64
}

type family struct {
	name, help, kind string
	labelNames       []string
	buckets          []float64
	mu               sync.Mutex
	series           map[string]*series
}

func (f *family) get(lv []string) *series {
	key := strings.Join(lv, "\xff")
	s, ok := f.series[key]
	if !ok {
		s = &series{labels: slices.Clone(lv)}
		if f.kind == "histogram" {
			s.counts = make([]uint64, len(f.buckets))
		}
		f.series[key] = s
	}
	return s
}

func (f *family) add(v float64, lv ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.get(lv).value += v
}

func (f *family) set(v float64, lv ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.get(lv).value = v
}

func (f *family) observe(v float64, lv ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	s := f.get(lv)
	for i, b := range f.buckets {
		if v <= b {
			s.counts[i]++
		}
	}
	s.value++ // count
	s.sum += v
}

type registry struct{ families []*family }

func (r *registry) new(kind, name, help string, buckets []float64, labels ...string) *family {
	f := &family{name: name, help: help, kind: kind, labelNames: labels, buckets: buckets, series: map[string]*series{}}
	r.families = append(r.families, f)
	return f
}

var labelEscaper = strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)

func labelString(names, values []string, extra ...string) string {
	var parts []string
	for i, n := range names {
		parts = append(parts, fmt.Sprintf(`%s="%s"`, n, labelEscaper.Replace(values[i])))
	}
	parts = append(parts, extra...)
	if len(parts) == 0 {
		return ""
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func (r *registry) write(w io.Writer) {
	for _, f := range r.families {
		f.mu.Lock()
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n", f.name, f.help, f.name, f.kind)
		keys := make([]string, 0, len(f.series))
		for k := range f.series {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		for _, k := range keys {
			s := f.series[k]
			if f.kind != "histogram" {
				fmt.Fprintf(w, "%s%s %g\n", f.name, labelString(f.labelNames, s.labels), s.value)
				continue
			}
			for i, b := range f.buckets {
				fmt.Fprintf(w, "%s_bucket%s %d\n", f.name, labelString(f.labelNames, s.labels, fmt.Sprintf(`le="%g"`, b)), s.counts[i])
			}
			fmt.Fprintf(w, "%s_bucket%s %g\n", f.name, labelString(f.labelNames, s.labels, `le="+Inf"`), s.value)
			fmt.Fprintf(w, "%s_sum%s %g\n%s_count%s %g\n", f.name, labelString(f.labelNames, s.labels), s.sum,
				f.name, labelString(f.labelNames, s.labels), s.value)
		}
		f.mu.Unlock()
	}
}

type metrics struct {
	reg         registry
	requests    *family
	upstream    *family
	latency     *family
	ttft        *family
	tokens      *family
	cache       *family
	ratelimited *family
	breaker     *family
}

func newMetrics() *metrics {
	m := &metrics{}
	lat := []float64{.05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60}
	m.requests = m.reg.new("counter", "gateway_requests_total", "Client requests by route and HTTP status.", nil, "route", "status")
	m.upstream = m.reg.new("counter", "gateway_upstream_attempts_total", "Upstream attempts by provider and outcome.", nil, "provider", "outcome")
	m.latency = m.reg.new("histogram", "gateway_upstream_latency_seconds", "Time to a complete upstream response.", lat, "provider")
	m.ttft = m.reg.new("histogram", "gateway_time_to_first_token_seconds", "Time to the first streamed chunk.", lat, "provider")
	m.tokens = m.reg.new("counter", "gateway_tokens_total", "Tokens by key, provider and kind.", nil, "key", "provider", "kind")
	m.cache = m.reg.new("counter", "gateway_cache_total", "Cache lookups by result.", nil, "result")
	m.ratelimited = m.reg.new("counter", "gateway_ratelimited_total", "Requests rejected by rate limits.", nil, "key", "limit")
	m.breaker = m.reg.new("gauge", "gateway_breaker_state", "Circuit breaker: 0 closed, 1 open, 2 half-open.", nil, "provider")
	return m
}
