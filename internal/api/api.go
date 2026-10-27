// Package api holds the OpenAI-compatible wire format the gateway speaks to clients.
package api

import (
	"encoding/json"
	"errors"
	"net/http"
)

type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatRequest keeps the fields the gateway reads, plus every field as sent (Raw),
// so OpenAI-compatible upstreams receive tools, response_format etc. untouched.
type ChatRequest struct {
	Model               string          `json:"model"`
	Messages            []Message       `json:"messages"`
	Stream              bool            `json:"stream"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Stop                json.RawMessage `json:"stop"`
	N                   *int            `json:"n"`
	Tools               json.RawMessage `json:"tools"`
	StreamOptions       *StreamOptions  `json:"stream_options"`

	Raw map[string]json.RawMessage `json:"-"`
}

func ParseChatRequest(body []byte) (*ChatRequest, error) {
	var r ChatRequest
	if err := json.Unmarshal(body, &r.Raw); err != nil {
		return nil, errors.New("request body is not a JSON object")
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, err
	}
	if r.Model == "" {
		return nil, errors.New("model is required")
	}
	if len(r.Messages) == 0 {
		return nil, errors.New("messages must not be empty")
	}
	return &r, nil
}

// MaxOutput is the output-token cap the client asked for, or 0.
func (r *ChatRequest) MaxOutput() int {
	switch {
	case r.MaxCompletionTokens != nil:
		return *r.MaxCompletionTokens
	case r.MaxTokens != nil:
		return *r.MaxTokens
	}
	return 0
}

// SetMaxTokens enforces an output cap on the request, in both the struct and Raw.
func (r *ChatRequest) SetMaxTokens(n int) {
	r.MaxTokens = &n
	r.Raw["max_tokens"], _ = json.Marshal(n)
}

// Encode returns the body for an OpenAI-compatible upstream: the client's request with the
// upstream model name, and usage reporting switched on for streams so tokens can be accounted.
func (r *ChatRequest) Encode(model string, stream bool) ([]byte, error) {
	out := make(map[string]json.RawMessage, len(r.Raw)+1)
	for k, v := range r.Raw {
		out[k] = v
	}
	out["model"], _ = json.Marshal(model)
	if stream {
		out["stream"] = json.RawMessage("true")
		out["stream_options"] = json.RawMessage(`{"include_usage":true}`)
	}
	return json.Marshal(out)
}

// Text returns a message's content as plain text. ok is false if it has non-text parts (images, audio).
func Text(content json.RawMessage) (text string, ok bool) {
	if len(content) == 0 || string(content) == "null" {
		return "", true
	}
	if err := json.Unmarshal(content, &text); err == nil {
		return text, true
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &parts); err != nil {
		return "", false
	}
	for _, p := range parts {
		if p.Type != "text" {
			return "", false
		}
		text += p.Text
	}
	return text, true
}

// PromptChars approximates the prompt size, used to reserve tokens before the upstream reports usage.
func (r *ChatRequest) PromptChars() int {
	n := 0
	for _, m := range r.Messages {
		n += len(m.Content) + len(m.Role)
	}
	return n
}

type errorBody struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}

// WriteError writes an error in the shape OpenAI SDKs parse.
func WriteError(w http.ResponseWriter, status int, code, message string) {
	var b errorBody
	b.Error.Message, b.Error.Code = message, code
	b.Error.Type = map[bool]string{true: "invalid_request_error", false: "api_error"}[status < 500]
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(b)
}
