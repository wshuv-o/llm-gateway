# llm-gateway

An OpenAI-compatible gateway in Go that sits between your apps and LLM providers. Apps call one endpoint with one key. The gateway:
- picks the provider;
- fails over when a provider is down;
- stops calling a provider that keeps failing;
- enforces per-key limits;
- caches deterministic answers;
- records what every request cost.

It uses only the Go standard library, so there are no dependencies to audit.

```
app ──► POST /v1/chat/completions  (any OpenAI SDK; model = a route alias like "smart")
           │ auth: hashed per-app keys → allowed routes, max_tokens cap
           │ rate limits: requests/min and tokens/min per key (reserve, then settle on real usage)
           │ cache: temperature-0 requests, scoped per key, concurrent duplicates collapsed
           ▼
        route "smart": anthropic/claude-sonnet-5-5 → openai/gpt-5      (priority or round-robin)
           │ skip targets that can't serve the request (e.g. tools on a text-only adapter)
           │ skip targets whose circuit breaker is open
           │ timeout / 5xx / 429 → next target;  400 → straight back to the app
           ▼
        provider adapters: OpenAI-compatible (OpenAI, Groq, vLLM, Ollama…) · Anthropic (translated, incl. streaming)
```

## Behaviour worth knowing

| Situation | What happens | Why |
|---|---|---|
| Provider returns 5xx, 429, times out, or rejects our provider key | Next target in the route | Another provider may well succeed |
| Provider returns 400, 413 or 422 | Error goes straight back to the app | A malformed request fails everywhere; retrying only adds latency and cost |
| Provider fails 3 times in a row | Circuit opens for 30 s (or the provider's `Retry-After`, if longer); one probe request then decides | Stops queueing requests behind a dead provider |
| Probe request abandoned by its client | Probe slot released | Otherwise the breaker would stay half-open forever |
| Stream fails **before** the first chunk | Fails over like any other request | The app hasn't seen anything yet |
| Stream fails **after** the first chunk | Stream ends with an error event; no failover | Splicing a second model's answer onto the first would produce text no model wrote |
| Streaming timeout | Applies to the first chunk only | A long answer may legitimately stream for minutes |
| Client disconnects | Upstream request is cancelled | No paying for tokens nobody reads |
| Request uses tools, images or `n>1` and the target is the Anthropic adapter | That target is skipped | It translates text chat only, and says so rather than guessing |

**Keys and policy.**
- Client keys are stored only as SHA-256 hashes (`llm-gateway new-key` prints a key and its hash).
- Client keys are never forwarded upstream; provider keys come from environment variables.
- Each key has the routes it may use, `rpm`, `tpm` and a `max_tokens` cap. The cap is injected when the client omits it, so it can't be bypassed by leaving it out.
- Unknown config fields are rejected at startup: a typo like `"rmp"` would otherwise silently mean "unlimited".

**Token limits** can't be known until the answer arrives. Each request reserves an estimate of prompt size plus `max_tokens`. Afterwards it settles against the provider's reported usage: an overestimate is refunded, and an underestimate becomes debt the key waits out.

**Cache.**
- Only requests with `temperature: 0` are cached, and never when the client sends `Cache-Control: no-store`.
- Entries are scoped per key, so one tenant can never receive, or time, another tenant's response.
- When ten identical requests arrive at once, one upstream call serves all ten.

**Audit log.** One JSON line per request:
- key and route;
- the provider that answered and every attempt with its outcome;
- token counts, cache result and latency.

Prompt and completion text are never logged.

**Metrics** (Prometheus, on a separate internal listener):
- requests by route and status;
- upstream attempts by outcome;
- latency and time to first token;
- tokens by key and provider;
- cache hits;
- rate-limit rejections;
- breaker state.

## Tests

```
go test ./...          # 34 tests, incl. fake upstreams for every failure mode above
go test -race ./...    # run in CI
```

The tests start real HTTP servers as fake providers. The fakes can:
- go down or rate-limit;
- hang before the first byte;
- drop the connection mid-stream;
- report more tokens than estimated.

I checked that the tests catch regressions by breaking the code on purpose:
- removing per-key cache scoping;
- making the breaker never open;
- leaking the gateway's own usage chunk;
- retrying 400s elsewhere.

Each of those changes makes a test fail.

**Overhead.** On a laptop over loopback, `go test -bench GatewayOverhead` measured about 0.3 ms per request, including the extra HTTP hop. That's negligible next to model latency.

## Run it

```bash
go run ./cmd/llm-gateway new-key                 # put the sha256 in gateway.json, give the key to the app
cp gateway.example.json gateway.json             # edit providers, routes, keys
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... go run ./cmd/llm-gateway -config gateway.json
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-gw-...")
client.chat.completions.create(model="smart", messages=[{"role": "user", "content": "Hi"}])
```

Docker: `docker build -t llm-gateway . && docker run -v $PWD:/config -p 8080:8080 llm-gateway`. The image is distroless, runs as non-root, and works with a read-only root filesystem.

## Limits, and what I'd do next

- Rate limits and cache are per instance. With several replicas, move both to Redis (or route each key to one replica). The bucket and LRU interfaces are small for that reason.
- The Anthropic adapter translates text chat only. Tool calls would be the next translation to add.
- No semantic cache. Exact-match caching is predictable, and a similarity threshold means sometimes answering a different question.
- Policy is static config. The natural next step is OPA or a policy webhook for per-request decisions (data classification, PII routing to the local model).
