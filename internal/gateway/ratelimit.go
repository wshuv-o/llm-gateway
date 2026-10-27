package gateway

import (
	"math"
	"sync"
	"time"
)

// bucket is a token bucket that refills continuously up to its capacity.
type bucket struct {
	capacity, tokens, perSec float64
	last                     time.Time
}

func newBucket(perMinute int, now time.Time) *bucket {
	c := float64(perMinute)
	return &bucket{capacity: c, tokens: c, perSec: c / 60, last: now}
}

func (b *bucket) refill(now time.Time) {
	b.tokens = math.Min(b.capacity, b.tokens+now.Sub(b.last).Seconds()*b.perSec)
	b.last = now
}

func (b *bucket) wait(n float64) time.Duration {
	return time.Duration((n - b.tokens) / b.perSec * float64(time.Second))
}

// limiter enforces requests-per-minute and tokens-per-minute for one key. Token use is only
// known after the response, so a request reserves an estimate up front and settles the
// difference afterwards. Underestimates become debt that the next requests wait for.
type limiter struct {
	mu       sync.Mutex
	requests *bucket // nil = unlimited
	tokens   *bucket
}

func newLimiter(rpm, tpm int, now time.Time) *limiter {
	l := &limiter{}
	if rpm > 0 {
		l.requests = newBucket(rpm, now)
	}
	if tpm > 0 {
		l.tokens = newBucket(tpm, now)
	}
	return l
}

type reservation struct {
	Tokens            float64
	RemainingRequests int // -1 = unlimited
	RemainingTokens   int
}

// reserve takes one request and `tokens` tokens, or neither. If it can't, it returns how long
// until it could and which limit was hit.
func (l *limiter) reserve(tokens float64, now time.Time) (r reservation, retry time.Duration, limit string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	r = reservation{RemainingRequests: -1, RemainingTokens: -1}
	if l.requests != nil {
		l.requests.refill(now)
		if l.requests.tokens < 1 {
			return r, l.requests.wait(1), "requests"
		}
	}
	if l.tokens != nil {
		l.tokens.refill(now)
		tokens = math.Min(tokens, l.tokens.capacity) // one request bigger than the whole budget can still run when it's full
		if l.tokens.tokens < tokens {
			return r, l.tokens.wait(tokens), "tokens"
		}
		l.tokens.tokens -= tokens
		r.Tokens = tokens
		r.RemainingTokens = int(l.tokens.tokens)
	}
	if l.requests != nil {
		l.requests.tokens--
		r.RemainingRequests = int(l.requests.tokens)
	}
	return r, 0, ""
}

// settle replaces the reserved estimate with what the request actually used.
func (l *limiter) settle(reserved float64, used int) {
	if l.tokens == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.tokens.tokens = math.Min(l.tokens.capacity, l.tokens.tokens+reserved-float64(used))
}
