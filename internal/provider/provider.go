// Package provider adapts upstream LLM APIs to one OpenAI-compatible interface.
package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/wshuv-o/llm-gateway/internal/api"
)

type Provider interface {
	Name() string
	// Supports returns why this provider can't serve req, or nil.
	Supports(req *api.ChatRequest) error
	Complete(ctx context.Context, model string, req *api.ChatRequest) (*Response, error)
	// Stream returns once the upstream has accepted the request; chunks follow from Next.
	Stream(ctx context.Context, model string, req *api.ChatRequest) (Stream, error)
}

// Response is a complete chat.completion in OpenAI format.
type Response struct {
	Body  []byte
	Usage api.Usage
}

// Chunk is one chat.completion.chunk in OpenAI format. Data is nil for chunks the
// client didn't ask for (the usage chunk the gateway requests for its own accounting).
type Chunk struct {
	Data  []byte
	Usage *api.Usage
}

type Stream interface {
	Next() (Chunk, error) // io.EOF after the last chunk
	Close() error
}

// Error is a failed upstream call. Status is 0 for transport errors and timeouts.
type Error struct {
	Provider   string
	Status     int
	Message    string
	RetryAfter time.Duration
	Err        error
}

func (e *Error) Error() string {
	if e.Status == 0 {
		return fmt.Sprintf("%s: %v", e.Provider, e.Err)
	}
	return fmt.Sprintf("%s: HTTP %d: %s", e.Provider, e.Status, e.Message)
}

func (e *Error) Unwrap() error { return e.Err }

// Failover reports whether another provider could succeed where this one failed. A malformed
// request fails everywhere, so it goes straight back to the client. Anything else (outage,
// overload, timeout, this provider's auth or model config) is worth trying elsewhere.
func (e *Error) Failover() bool {
	switch e.Status {
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		return false
	}
	return true
}

const maxResponseBytes = 32 << 20

// errorFrom turns a non-200 upstream response into an *Error and closes the body.
func errorFrom(name string, resp *http.Response) *Error {
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	e := &Error{Provider: name, Status: resp.StatusCode, Message: upstreamMessage(body)}
	if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s > 0 {
		e.RetryAfter = time.Duration(s) * time.Second
	}
	return e
}

// upstreamMessage pulls error.message out of an OpenAI- or Anthropic-style error body.
func upstreamMessage(body []byte) string {
	var b struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &b) == nil && b.Error.Message != "" {
		return b.Error.Message
	}
	if len(body) > 200 {
		body = body[:200]
	}
	return string(bytes.TrimSpace(body))
}

func post(ctx context.Context, client *http.Client, name, url string, body []byte, header http.Header) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, &Error{Provider: name, Err: err}
	}
	req.Header = header
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, &Error{Provider: name, Err: err}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errorFrom(name, resp)
	}
	return resp, nil
}

// sseReader reads Server-Sent Events.
type sseReader struct{ sc *bufio.Scanner }

type sseEvent struct {
	Event string
	Data  []byte
}

func newSSEReader(r io.Reader) *sseReader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	return &sseReader{sc}
}

func (r *sseReader) Next() (sseEvent, error) {
	var ev sseEvent
	var data [][]byte
	for r.sc.Scan() {
		line := bytes.TrimSuffix(r.sc.Bytes(), []byte("\r"))
		if len(line) == 0 {
			if len(data) > 0 {
				ev.Data = bytes.Join(data, []byte("\n"))
				return ev, nil
			}
			ev.Event = ""
			continue
		}
		if line[0] == ':' {
			continue // comment / keep-alive
		}
		field, value, _ := bytes.Cut(line, []byte(":"))
		value = bytes.TrimPrefix(value, []byte(" "))
		switch string(field) {
		case "event":
			ev.Event = string(value)
		case "data":
			data = append(data, bytes.Clone(value))
		}
	}
	if err := r.sc.Err(); err != nil {
		return ev, err
	}
	if len(data) > 0 {
		ev.Data = bytes.Join(data, []byte("\n"))
		return ev, nil
	}
	return ev, io.EOF
}
