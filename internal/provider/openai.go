package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/wshuv-o/llm-gateway/internal/api"
)

// OpenAI talks to any OpenAI-compatible API: OpenAI, Groq, Together, vLLM, Ollama, LiteLLM.
// Requests are forwarded as sent (tools, response_format...), with only the model replaced.
type OpenAI struct {
	name    string
	baseURL string // e.g. https://api.openai.com/v1
	apiKey  string
	client  *http.Client
}

func NewOpenAI(name, baseURL, apiKey string, client *http.Client) *OpenAI {
	return &OpenAI{name, strings.TrimRight(baseURL, "/"), apiKey, client}
}

func (p *OpenAI) Name() string                        { return p.name }
func (p *OpenAI) Supports(req *api.ChatRequest) error { return nil }

func (p *OpenAI) header() http.Header {
	h := http.Header{}
	if p.apiKey != "" {
		h.Set("Authorization", "Bearer "+p.apiKey)
	}
	return h
}

func (p *OpenAI) Complete(ctx context.Context, model string, req *api.ChatRequest) (*Response, error) {
	body, err := req.Encode(model, false)
	if err != nil {
		return nil, &Error{Provider: p.name, Status: http.StatusBadRequest, Message: err.Error()}
	}
	resp, err := post(ctx, p.client, p.name, p.baseURL+"/chat/completions", body, p.header())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return nil, &Error{Provider: p.name, Err: err}
	}
	var parsed struct {
		Usage *api.Usage `json:"usage"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		return nil, &Error{Provider: p.name, Status: http.StatusBadGateway, Message: "upstream returned invalid JSON"}
	}
	r := &Response{Body: out}
	if parsed.Usage != nil {
		r.Usage = *parsed.Usage
	}
	return r, nil
}

func (p *OpenAI) Stream(ctx context.Context, model string, req *api.ChatRequest) (Stream, error) {
	body, err := req.Encode(model, true)
	if err != nil {
		return nil, &Error{Provider: p.name, Status: http.StatusBadRequest, Message: err.Error()}
	}
	resp, err := post(ctx, p.client, p.name, p.baseURL+"/chat/completions", body, p.header())
	if err != nil {
		return nil, err
	}
	wantUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
	return &openaiStream{p.name, resp.Body, newSSEReader(resp.Body), wantUsage}, nil
}

type openaiStream struct {
	name      string
	body      io.ReadCloser
	sse       *sseReader
	wantUsage bool
}

func (s *openaiStream) Next() (Chunk, error) {
	ev, err := s.sse.Next()
	if err == io.EOF {
		return Chunk{}, io.ErrUnexpectedEOF // a complete stream ends with [DONE]
	}
	if err != nil {
		return Chunk{}, &Error{Provider: s.name, Err: err}
	}
	if string(ev.Data) == "[DONE]" {
		return Chunk{}, io.EOF
	}
	var c struct {
		Choices []json.RawMessage `json:"choices"`
		Usage   *api.Usage        `json:"usage"`
		Error   *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(ev.Data, &c); err != nil {
		return Chunk{}, &Error{Provider: s.name, Status: http.StatusBadGateway, Message: "invalid chunk from upstream"}
	}
	if c.Error != nil {
		return Chunk{}, &Error{Provider: s.name, Status: http.StatusBadGateway, Message: c.Error.Message}
	}
	chunk := Chunk{Data: ev.Data, Usage: c.Usage}
	if len(c.Choices) == 0 && c.Usage != nil && !s.wantUsage {
		chunk.Data = nil // we asked for usage for accounting; the client didn't
	}
	return chunk, nil
}

func (s *openaiStream) Close() error { return s.body.Close() }
