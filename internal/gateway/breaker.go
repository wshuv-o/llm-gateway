package gateway

import (
	"sync"
	"time"
)

const (
	breakerClosed = iota
	breakerOpen
	breakerHalfOpen
)

// breaker stops sending traffic to a provider that keeps failing. After `threshold` consecutive
// failures it opens for `cooldown` (or the upstream's Retry-After, if longer). Then one probe
// request is let through: success closes it, failure opens it again.
type breaker struct {
	mu        sync.Mutex
	threshold int
	cooldown  time.Duration
	now       func() time.Time
	failures  int
	openUntil time.Time
	probing   bool
}

func newBreaker(threshold int, cooldown time.Duration, now func() time.Time) *breaker {
	return &breaker{threshold: threshold, cooldown: cooldown, now: now}
}

// allow reports whether a request may go to the provider now.
func (b *breaker) allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.stateLocked() {
	case breakerClosed:
		return true
	case breakerHalfOpen:
		if !b.probing {
			b.probing = true
			return true
		}
	}
	return false
}

func (b *breaker) success() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures, b.probing = 0, false
}

func (b *breaker) failure(retryAfter time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures++
	b.probing = false
	if b.failures >= b.threshold {
		b.openUntil = b.now().Add(max(b.cooldown, retryAfter))
	}
}

// abandon releases a probe whose request ended without a verdict (the client went away).
// Without it, a cancelled probe would leave the breaker half-open with no way to close.
func (b *breaker) abandon() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.probing = false
}

func (b *breaker) state() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.stateLocked()
}

func (b *breaker) stateLocked() int {
	if b.failures < b.threshold {
		return breakerClosed
	}
	if b.now().Before(b.openUntil) {
		return breakerOpen
	}
	return breakerHalfOpen
}
