// Package dazagentsdk provides a provider-agnostic AI library with
// tier-based routing and automatic fallback.
package dazagentsdk

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/google/uuid"
)

// Tier represents a logical model quality tier.
// Config maps these to concrete provider:model pairs.
type Tier string

const (
	TierVeryHigh     Tier = "very_high"
	TierHigh         Tier = "high"
	TierMedium       Tier = "medium"
	TierLow          Tier = "low"
	TierFreeFast     Tier = "free_fast"
	TierFreeThinking Tier = "free_thinking"
)

// Capability describes what kind of work a model can do.
type Capability string

const (
	CapabilityText       Capability = "text"
	CapabilityStructured Capability = "structured"
	CapabilityAgentic    Capability = "agentic"
	CapabilityImage      Capability = "image"
	CapabilityTTS        Capability = "tts"
	CapabilitySTT        Capability = "stt"
)

// ErrorKind classifies errors for fallback decision making.
type ErrorKind string

const (
	ErrorRateLimit      ErrorKind = "rate_limit"
	ErrorAuth           ErrorKind = "auth"
	ErrorTimeout        ErrorKind = "timeout"
	ErrorInvalidRequest ErrorKind = "invalid_request"
	ErrorNotAvailable   ErrorKind = "not_available"
	ErrorInternal       ErrorKind = "internal"
)

// ModelInfo describes a concrete model offered by a provider.
type ModelInfo struct {
	Provider              string       `json:"provider"`
	ModelID               string       `json:"model_id"`
	DisplayName           string       `json:"display_name"`
	Capabilities          []Capability `json:"capabilities"`
	Tier                  Tier         `json:"tier"`
	SupportsStreaming      bool         `json:"supports_streaming"`
	SupportsStructured    bool         `json:"supports_structured"`
	SupportsConversation  bool         `json:"supports_conversation"`
	SupportsTools         bool         `json:"supports_tools"`
	MaxContext            *int         `json:"max_context,omitempty"`
}

// QualifiedName returns the provider:model_id format used in config files
// and logging.
func (m ModelInfo) QualifiedName() string {
	return m.Provider + ":" + m.ModelID
}

// Message represents a single message in a conversation history.
type Message struct {
	Role     string         `json:"role"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ToDict serialises the message for logging and provider APIs.
// Metadata is only included when non-empty.
func (m Message) ToDict() map[string]any {
	result := map[string]any{
		"role":    m.Role,
		"content": m.Content,
	}
	if len(m.Metadata) > 0 {
		result["metadata"] = m.Metadata
	}
	return result
}

// Response is the result of an ask/say call.
type Response struct {
	Text           string         `json:"text"`
	ModelUsed      ModelInfo      `json:"model_used"`
	ConversationID uuid.UUID      `json:"conversation_id"`
	TurnID         uuid.UUID      `json:"turn_id"`
	Usage          map[string]any `json:"usage,omitempty"`
}

// StructuredResponse is the result when a schema was provided.
type StructuredResponse struct {
	Response
	Parsed any `json:"parsed,omitempty"`
}

// ImageResult is the result of an image generation call.
type ImageResult struct {
	Path           string    `json:"path"`
	ModelUsed      ModelInfo `json:"model_used"`
	ConversationID uuid.UUID `json:"conversation_id"`
	Prompt         string    `json:"prompt"`
	Width          int       `json:"width"`
	Height         int       `json:"height"`
}

// AudioResult is the result of a TTS call.
type AudioResult struct {
	Path            string    `json:"path"`
	ModelUsed       ModelInfo `json:"model_used"`
	ConversationID  uuid.UUID `json:"conversation_id"`
	Text            string    `json:"text"`
	Voice           string    `json:"voice"`
	DurationSeconds *float64  `json:"duration_seconds,omitempty"`
}

// AgentError is the base error for all agent-sdk failures.
// It carries structured context for debugging including all provider attempts.
type AgentError struct {
	Message  string         `json:"error"`
	Kind     ErrorKind      `json:"kind"`
	Attempts []map[string]any `json:"attempts"`
}

// Error implements the error interface.
func (e *AgentError) Error() string {
	return e.Message
}

// ToDict returns a structured representation for logging.
func (e *AgentError) ToDict() map[string]any {
	return map[string]any{
		"error":    e.Message,
		"kind":     string(e.Kind),
		"attempts": e.Attempts,
	}
}

// NewAgentError creates a new AgentError.
func NewAgentError(message string, kind ErrorKind, attempts []map[string]any) *AgentError {
	if attempts == nil {
		attempts = []map[string]any{}
	}
	return &AgentError{
		Message:  message,
		Kind:     kind,
		Attempts: attempts,
	}
}

// fenceRe matches markdown code fences containing JSON anywhere in text.
var fenceRe = regexp.MustCompile("(?s)```(?:json)?\\s*\n(.*?)\n```")

// ParseJSONFromLLM extracts JSON from LLM response text.
// It handles raw JSON, markdown-fenced JSON, and JSON embedded at the end
// of a prose response.
func ParseJSONFromLLM(text string) (map[string]any, error) {
	text = strings.TrimSpace(text)

	// Strip leading markdown fence
	if strings.HasPrefix(text, "```") {
		lines := strings.Split(text, "\n")
		lines = lines[1:]
		if len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "```" {
			lines = lines[:len(lines)-1]
		}
		text = strings.TrimSpace(strings.Join(lines, "\n"))
	}

	// Try direct parse first
	var result map[string]any
	if err := json.Unmarshal([]byte(text), &result); err == nil {
		return result, nil
	}

	// Try extracting JSON from markdown fences anywhere in the text
	if matches := fenceRe.FindStringSubmatch(text); len(matches) > 1 {
		var fenced map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSpace(matches[1])), &fenced); err == nil {
			return fenced, nil
		}
	}

	// Try finding the last JSON object in the text (scan backwards for '}')
	for i := len(text) - 1; i >= 0; i-- {
		if text[i] == '}' {
			depth := 0
			for j := i; j >= 0; j-- {
				if text[j] == '}' {
					depth++
				} else if text[j] == '{' {
					depth--
					if depth == 0 {
						candidate := text[j : i+1]
						var obj map[string]any
						if err := json.Unmarshal([]byte(candidate), &obj); err == nil {
							return obj, nil
						}
						break
					}
				}
			}
			break
		}
	}

	return nil, fmt.Errorf("no valid JSON found in text")
}
