package gateway

import (
	"container/list"
	"sync"
	"time"

	"github.com/wshuv-o/llm-gateway/internal/provider"
)

type cached struct {
	resp     *provider.Response
	provider string
	model    string
}

// cache is an LRU with a TTL for deterministic (temperature 0) completions.
type cache struct {
	mu    sync.Mutex
	max   int
	ttl   time.Duration
	now   func() time.Time
	order *list.List // front = most recent
	items map[string]*list.Element
}

type cacheEntry struct {
	key     string
	val     cached
	expires time.Time
}

func newCache(max int, ttl time.Duration, now func() time.Time) *cache {
	return &cache{max: max, ttl: ttl, now: now, order: list.New(), items: map[string]*list.Element{}}
}

func (c *cache) get(key string) (cached, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[key]
	if !ok {
		return cached{}, false
	}
	e := el.Value.(*cacheEntry)
	if c.now().After(e.expires) {
		c.order.Remove(el)
		delete(c.items, key)
		return cached{}, false
	}
	c.order.MoveToFront(el)
	return e.val, true
}

func (c *cache) put(key string, val cached) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		el.Value = &cacheEntry{key, val, c.now().Add(c.ttl)}
		c.order.MoveToFront(el)
		return
	}
	c.items[key] = c.order.PushFront(&cacheEntry{key, val, c.now().Add(c.ttl)})
	for c.order.Len() > c.max {
		old := c.order.Back()
		c.order.Remove(old)
		delete(c.items, old.Value.(*cacheEntry).key)
	}
}

// flights collapses concurrent identical requests into one upstream call (like x/sync/singleflight).
type flights struct {
	mu sync.Mutex
	m  map[string]*flight
}

type flight struct {
	done chan struct{}
	val  cached
	err  error
}

func (f *flights) do(key string, fn func() (cached, error)) (val cached, err error, shared bool) {
	f.mu.Lock()
	if f.m == nil {
		f.m = map[string]*flight{}
	}
	if fl, ok := f.m[key]; ok {
		f.mu.Unlock()
		<-fl.done
		return fl.val, fl.err, true
	}
	fl := &flight{done: make(chan struct{})}
	f.m[key] = fl
	f.mu.Unlock()

	fl.val, fl.err = fn()
	f.mu.Lock()
	delete(f.m, key)
	f.mu.Unlock()
	close(fl.done)
	return fl.val, fl.err, false
}
