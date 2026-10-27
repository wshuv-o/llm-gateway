package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/wshuv-o/llm-gateway/internal/api"
)

// Anthropic translates OpenAI-format chat requests to the Anthropic Messages API and back,
// including streams. It handles text chat; requests it can't translate faithfully
// (tools, images, n > 1) are refused by Supports, so the router sends them to another target.
type Anthropic struct {
	name             string
	baseURL          string // e.g. https://api.anthropic.com
	apiKey           string
	defaultMaxTokens int // Anthropic requires max_tokens; OpenAI clients often omit it
	client           *http.Client
}

func NewAnthropic(name, baseURL, apiKey string, defaultMaxTokens int, client *http.Client) *Anthropic {
	if defaultMaxTokens <= 0 {
		defaultMaxTokens = 1024
	}
	return &Anthropic{name, strings.TrimRight(baseURL, "/"), apiKey, defaultMaxTokens, client}
}

func (p *Anthropic) Name() string { return p.name }

func (p *Anthropic) Supports(req *api.ChatRequest) error {
	if len(req.Tools) > 0 && string(req.Tools) != "null" {
		return errors.New("tool calling is not translated for this provider")
	}
	if req.N != nil && *req.N > 1 {
		return errors.New("n > 1 is not supported by this provider")
	}
	for _, m := range req.Messages {
		if m.Role != "system" && m.Role != "user" && m.Role != "assistant" {
			return fmt.Errorf("message role %q is not supported by this provider", m.Role)
		}
		if _, ok := api.Text(m.Content); !ok {
			return errors.New("non-text content is not translated for this provider")
		}
	}
	return nil
}

type anthMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthRequest struct {
	Model         string        `json:"model"`
	MaxTokens     int           `json:"max_tokens"`
	System        string        `json:"system,omitempty"`
	Messages      []anthMessage `json:"messages"`
	Temperature   *float64      `json:"temperature,omitempty"`
	TopP          *float64      `json:"top_p,omitempty"`
	StopSequences []string      `json:"stop_sequences,omitempty"`
	Stream        bool          `json:"stream,omitempty"`
}

func (p *Anthropic) translate(model string, req *api.ChatRequest, stream bool) ([]byte, error) {
	out := anthRequest{Model: model, MaxTokens: req.MaxOutput(), TopP: req.TopP, Stream: stream}
	if out.MaxTokens <= 0 {
		out.MaxTokens = p.defaultMaxTokens
	}
	if t := req.Temperature; t != nil {
		v := min(*t, 1) // OpenAI allows 0-2, Anthropic 0-1
		out.Temperature = &v
	}
	var system []string
	for _, m := range req.Messages {
		text, _ := api.Text(m.Content)
		if m.Role == "system" {
			system = append(system, text)
			continue
		}
		out.Messages = append(out.Messages, anthMessage{m.Role, text})
	}
	out.System = strings.Join(system, "\n\n")
	if len(req.Stop) > 0 && string(req.Stop) != "null" {
		var one string
		if json.Unmarshal(req.Stop, &one) == nil {
			out.StopSequences = []string{one}
		} else if err := json.Unmarshal(req.Stop, &out.StopSequences); err != nil {
			return nil, errors.New("stop must be a string or an array of strings")
		}
	}
	return json.Marshal(out)
}

func (p *Anthropic) header() http.Header {
	h := http.Header{}
	h.Set("x-api-key", p.apiKey)
	h.Set("anthropic-version", "2023-06-01")
	return h
}

var finishReasons = map[string]string{
	"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter",
}

func finishReason(stop string) string {
	if r, ok := finishReasons[stop]; ok {
		return r
	}
	return "stop"
}

type anthUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func (p *Anthropic) Complete(ctx context.Context, model string, req *api.ChatRequest) (*Response, error) {
	body, err := p.translate(model, req, false)
	if err != nil {
		return nil, &Error{Provider: p.name, Status: http.StatusBadRequest, Message: err.Error()}
	}
	resp, err := post(ctx, p.client, p.name, p.baseURL+"/v1/messages", body, p.header())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var r struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string    `json:"stop_reason"`
		Usage      anthUsage `json:"usage"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxResponseBytes)).Decode(&r); err != nil {
		return nil, &Error{Provider: p.name, Status: http.StatusBadGateway, Message: "upstream returned invalid JSON"}
	}
	var text strings.Builder
	for _, c := range r.Content {
		if c.Type == "text" {
			text.WriteString(c.Text)
		}
	}
	usage := api.Usage{PromptTokens: r.Usage.InputTokens, CompletionTokens: r.Usage.OutputTokens,
		TotalTokens: r.Usage.InputTokens + r.Usage.OutputTokens}
	out, _ := json.Marshal(map[string]any{
		"id": r.ID, "object": "chat.completion", "created": time.Now().Unix(), "model": r.Model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": text.String()},
			"finish_reason": finishReason(r.StopReason),
		}},
		"usage": usage,
	})
	return &Response{Body: out, Usage: usage}, nil
}

func (p *Anthropic) Stream(ctx context.Context, model string, req *api.ChatRequest) (Stream, error) {
	body, err := p.translate(model, req, true)
	if err != nil {
		return nil, &Error{Provider: p.name, Status: http.StatusBadRequest, Message: err.Error()}
	}
	resp, err := post(ctx, p.client, p.name, p.baseURL+"/v1/messages", body, p.header())
	if err != nil {
		return nil, err
	}
	wantUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
	return &anthStream{name: p.name, body: resp.Body, sse: newSSEReader(resp.Body), wantUsage: wantUsage,
		created: time.Now().Unix()}, nil
}

// anthStream turns Anthropic stream events into OpenAI chat.completion.chunk objects.
type anthStream struct {
	name      string
	body      io.ReadCloser
	sse       *sseReader
	wantUsage bool
	created   int64
	id, model string
	usage     anthUsage
	done      bool
}

func (s *anthStream) chunk(delta map[string]any, finish any) []byte {
	b, _ := json.Marshal(map[string]any{
		"id": s.id, "object": "chat.completion.chunk", "created": s.created, "model": s.model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	})
	return b
}

func (s *anthStream) Next() (Chunk, error) {
	for {
		if s.done {
			return Chunk{}, io.EOF
		}
		ev, err := s.sse.Next()
		if err == io.EOF {
			return Chunk{}, io.ErrUnexpectedEOF // a complete stream ends with message_stop
		}
		if err != nil {
			return Chunk{}, &Error{Provider: s.name, Err: err}
		}
		var e struct {
			Type    string `json:"type"`
			Message struct {
				ID    string    `json:"id"`
				Model string    `json:"model"`
				Usage anthUsage `json:"usage"`
			} `json:"message"`
			Delta struct {
				Type       string `json:"type"`
				Text       string `json:"text"`
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage *anthUsage `json:"usage"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(ev.Data, &e); err != nil {
			return Chunk{}, &Error{Provider: s.name, Status: http.StatusBadGateway, Message: "invalid event from upstream"}
		}
		switch e.Type {
		case "message_start":
			s.id, s.model, s.usage = e.Message.ID, e.Message.Model, e.Message.Usage
			return Chunk{Data: s.chunk(map[string]any{"role": "assistant", "content": ""}, nil)}, nil
		case "content_block_delta":
			if e.Delta.Type == "text_delta" {
				return Chunk{Data: s.chunk(map[string]any{"content": e.Delta.Text}, nil)}, nil
			}
		case "message_delta":
			if e.Usage != nil {
				s.usage.OutputTokens = e.Usage.OutputTokens
			}
			if e.Delta.StopReason != "" {
				return Chunk{Data: s.chunk(map[string]any{}, finishReason(e.Delta.StopReason))}, nil
			}
		case "message_stop":
			s.done = true
			u := api.Usage{PromptTokens: s.usage.InputTokens, CompletionTokens: s.usage.OutputTokens,
				TotalTokens: s.usage.InputTokens + s.usage.OutputTokens}
			c := Chunk{Usage: &u}
			if s.wantUsage {
				c.Data, _ = json.Marshal(map[string]any{"id": s.id, "object": "chat.completion.chunk",
					"created": s.created, "model": s.model, "choices": []any{}, "usage": u})
			}
			return c, nil
		case "error":
			status := http.StatusBadGateway
			if e.Error.Type == "overloaded_error" {
				status = http.StatusServiceUnavailable
			}
			return Chunk{}, &Error{Provider: s.name, Status: status, Message: e.Error.Message}
		}
		// ping, content_block_start/stop: nothing for the client
	}
}

func (s *anthStream) Close() error { return s.body.Close() }
