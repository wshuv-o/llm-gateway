package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/wshuv-o/llm-gateway/internal/api"
)

func parse(t *testing.T, body string) *api.ChatRequest {
	t.Helper()
	r, err := api.ParseChatRequest([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestAnthropicTranslatesRequestAndResponse(t *testing.T) {
	var got map[string]any
	var header http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header = r.Header
		json.NewDecoder(r.Body).Decode(&got)
		io.WriteString(w, `{"id":"msg_1","model":"claude-haiku-5-5","content":[{"type":"text","text":"Hi "},{"type":"text","text":"there"}],
			"stop_reason":"max_tokens","usage":{"input_tokens":12,"output_tokens":4}}`)
	}))
	defer srv.Close()
	p := NewAnthropic("anthropic", srv.URL, "secret", 0, srv.Client())
	req := parse(t, `{"model":"fast","temperature":1.5,"stop":"END","messages":[
		{"role":"system","content":"Be brief."},
		{"role":"user","content":[{"type":"text","text":"Hello"}]}]}`)

	resp, err := p.Complete(context.Background(), "claude-haiku-5-5", req)
	if err != nil {
		t.Fatal(err)
	}
	if header.Get("x-api-key") != "secret" || header.Get("anthropic-version") == "" {
		t.Fatal("auth headers")
	}
	if got["system"] != "Be brief." || got["max_tokens"] != float64(1024) || got["temperature"] != float64(1) {
		t.Fatalf("translated request: %v", got)
	}
	if msgs := got["messages"].([]any); len(msgs) != 1 || msgs[0].(map[string]any)["content"] != "Hello" {
		t.Fatalf("messages: %v", got["messages"])
	}
	if stops := got["stop_sequences"].([]any); stops[0] != "END" {
		t.Fatal("stop")
	}
	var out struct {
		Choices []struct {
			Message      struct{ Content string } `json:"message"`
			FinishReason string                   `json:"finish_reason"`
		} `json:"choices"`
	}
	json.Unmarshal(resp.Body, &out)
	if out.Choices[0].Message.Content != "Hi there" || out.Choices[0].FinishReason != "length" || resp.Usage.TotalTokens != 16 {
		t.Fatalf("translated response: %s", resp.Body)
	}
}

const anthropicStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-haiku-5-5","usage":{"input_tokens":9,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`

func TestAnthropicStreamBecomesOpenAIChunks(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, anthropicStream)
	}))
	defer srv.Close()
	p := NewAnthropic("anthropic", srv.URL, "k", 0, srv.Client())
	s, err := p.Stream(context.Background(), "claude-haiku-5-5",
		parse(t, `{"model":"fast","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var text, finish string
	var usage *api.Usage
	for {
		c, err := s.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if c.Usage != nil {
			usage = c.Usage
		}
		if c.Data == nil {
			continue
		}
		var chunk struct {
			Object  string `json:"object"`
			Choices []struct {
				Delta        struct{ Content string } `json:"delta"`
				FinishReason *string                  `json:"finish_reason"`
			} `json:"choices"`
		}
		json.Unmarshal(c.Data, &chunk)
		if chunk.Object != "chat.completion.chunk" {
			t.Fatalf("chunk %s", c.Data)
		}
		text += chunk.Choices[0].Delta.Content
		if f := chunk.Choices[0].FinishReason; f != nil {
			finish = *f
		}
	}
	if text != "Hello" || finish != "stop" || usage == nil || usage.TotalTokens != 11 {
		t.Fatalf("text %q finish %q usage %+v", text, finish, usage)
	}
}

func TestAnthropicMidStreamOverloadIsRetryableError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n")
	}))
	defer srv.Close()
	p := NewAnthropic("anthropic", srv.URL, "k", 0, srv.Client())
	s, _ := p.Stream(context.Background(), "m", parse(t, `{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	_, err := s.Next()
	pe, ok := err.(*Error)
	if !ok || pe.Status != 503 || !pe.Failover() {
		t.Fatalf("got %v", err)
	}
}

func TestAnthropicRefusesWhatItCannotTranslate(t *testing.T) {
	p := NewAnthropic("anthropic", "https://x", "k", 0, nil)
	for _, body := range []string{
		`{"model":"x","tools":[{"type":"function"}],"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}`,
		`{"model":"x","messages":[{"role":"tool","content":"result"}]}`,
		`{"model":"x","n":2,"messages":[{"role":"user","content":"hi"}]}`,
	} {
		if p.Supports(parse(t, body)) == nil {
			t.Errorf("should refuse %s", body)
		}
	}
	if err := p.Supports(parse(t, `{"model":"x","messages":[{"role":"user","content":"hi"}]}`)); err != nil {
		t.Fatal(err)
	}
}

func TestSSEReader(t *testing.T) {
	r := newSSEReader(strings.NewReader(": keep-alive\r\nevent: a\r\ndata: line1\r\ndata: line2\r\n\r\ndata: last"))
	ev, _ := r.Next()
	if ev.Event != "a" || string(ev.Data) != "line1\nline2" {
		t.Fatalf("%+v", ev)
	}
	ev, _ = r.Next()
	if string(ev.Data) != "last" {
		t.Fatal("an unterminated final event is still delivered")
	}
	if _, err := r.Next(); err != io.EOF {
		t.Fatal(err)
	}
}

func TestErrorClassification(t *testing.T) {
	for status, want := range map[int]bool{0: true, 400: false, 401: true, 404: true, 413: false, 422: false, 429: true, 500: true, 529: true} {
		if got := (&Error{Status: status}).Failover(); got != want {
			t.Errorf("status %d: failover %v, want %v", status, got, want)
		}
	}
}
