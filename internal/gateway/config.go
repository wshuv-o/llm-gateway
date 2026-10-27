package gateway

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"slices"
)

type Config struct {
	Listen        string                 `json:"listen"`
	MetricsListen string                 `json:"metrics_listen"`
	MaxBodyBytes  int64                  `json:"max_body_bytes"`
	Providers     []ProviderConfig       `json:"providers"`
	Routes        map[string]RouteConfig `json:"routes"`
	Keys          []KeyConfig            `json:"keys"`
	Cache         CacheConfig            `json:"cache"`
	Breaker       BreakerConfig          `json:"breaker"`
}

type ProviderConfig struct {
	Name             string  `json:"name"`
	Kind             string  `json:"kind"` // openai | anthropic
	BaseURL          string  `json:"base_url"`
	APIKeyEnv        string  `json:"api_key_env"`
	TimeoutSeconds   float64 `json:"timeout_seconds"` // whole response, or first chunk for streams
	DefaultMaxTokens int     `json:"default_max_tokens"`
}

type RouteConfig struct {
	Strategy string   `json:"strategy"` // priority (default) | round_robin
	Targets  []Target `json:"targets"`
}

type Target struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// KeyConfig is a client API key. Only its SHA-256 is stored; the gateway never sees it in plain
// text at rest, and client keys are never forwarded upstream.
type KeyConfig struct {
	ID        string   `json:"id"`
	SHA256    string   `json:"sha256"`
	Routes    []string `json:"routes"` // model aliases this key may use; "*" for all
	RPM       int      `json:"rpm"`    // requests per minute, 0 = unlimited
	TPM       int      `json:"tpm"`    // tokens per minute, 0 = unlimited
	MaxTokens int      `json:"max_tokens"`
}

type CacheConfig struct {
	MaxEntries int `json:"max_entries"` // 0 disables the cache
	TTLSeconds int `json:"ttl_seconds"`
}

type BreakerConfig struct {
	Failures        int     `json:"failures"`
	CooldownSeconds float64 `json:"cooldown_seconds"`
}

func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseConfig(raw)
}

func ParseConfig(raw []byte) (*Config, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields() // a typo in a limit must fail loudly, not silently mean "unlimited"
	var c Config
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	c.defaults()
	return &c, c.validate()
}

func (c *Config) defaults() {
	if c.Listen == "" {
		c.Listen = ":8080"
	}
	if c.MaxBodyBytes == 0 {
		c.MaxBodyBytes = 1 << 20
	}
	if c.Breaker.Failures == 0 {
		c.Breaker.Failures = 3
	}
	if c.Breaker.CooldownSeconds == 0 {
		c.Breaker.CooldownSeconds = 30
	}
	if c.Cache.TTLSeconds == 0 {
		c.Cache.TTLSeconds = 300
	}
	for i := range c.Providers {
		if c.Providers[i].TimeoutSeconds == 0 {
			c.Providers[i].TimeoutSeconds = 60
		}
	}
}

func (c *Config) validate() error {
	var errs []error
	names := map[string]bool{}
	for _, p := range c.Providers {
		if p.Name == "" || names[p.Name] {
			errs = append(errs, fmt.Errorf("provider name %q is empty or duplicated", p.Name))
		}
		names[p.Name] = true
		if p.Kind != "openai" && p.Kind != "anthropic" {
			errs = append(errs, fmt.Errorf("provider %s: kind must be openai or anthropic", p.Name))
		}
		if u, err := url.Parse(p.BaseURL); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			errs = append(errs, fmt.Errorf("provider %s: base_url must be an http(s) URL", p.Name))
		}
		if p.APIKeyEnv != "" && os.Getenv(p.APIKeyEnv) == "" {
			errs = append(errs, fmt.Errorf("provider %s: environment variable %s is not set", p.Name, p.APIKeyEnv))
		}
	}
	if len(c.Routes) == 0 {
		errs = append(errs, errors.New("at least one route is required"))
	}
	for alias, r := range c.Routes {
		if r.Strategy != "" && r.Strategy != "priority" && r.Strategy != "round_robin" {
			errs = append(errs, fmt.Errorf("route %s: strategy must be priority or round_robin", alias))
		}
		if len(r.Targets) == 0 {
			errs = append(errs, fmt.Errorf("route %s: no targets", alias))
		}
		for _, t := range r.Targets {
			if !names[t.Provider] || t.Model == "" {
				errs = append(errs, fmt.Errorf("route %s: target %s/%s has an unknown provider or no model", alias, t.Provider, t.Model))
			}
		}
	}
	ids, hashes := map[string]bool{}, map[string]bool{}
	for _, k := range c.Keys {
		if k.ID == "" || ids[k.ID] {
			errs = append(errs, fmt.Errorf("key id %q is empty or duplicated", k.ID))
		}
		ids[k.ID] = true
		if b, err := hex.DecodeString(k.SHA256); err != nil || len(b) != 32 || hashes[k.SHA256] {
			errs = append(errs, fmt.Errorf("key %s: sha256 must be 64 hex characters and unique", k.ID))
		}
		hashes[k.SHA256] = true
		for _, r := range k.Routes {
			if _, ok := c.Routes[r]; !ok && r != "*" {
				errs = append(errs, fmt.Errorf("key %s: unknown route %s", k.ID, r))
			}
		}
		if k.RPM < 0 || k.TPM < 0 || k.MaxTokens < 0 {
			errs = append(errs, fmt.Errorf("key %s: limits must not be negative", k.ID))
		}
	}
	if len(c.Keys) == 0 {
		errs = append(errs, errors.New("at least one key is required"))
	}
	return errors.Join(errs...)
}

func (k *KeyConfig) allows(route string) bool {
	return slices.Contains(k.Routes, "*") || slices.Contains(k.Routes, route)
}
