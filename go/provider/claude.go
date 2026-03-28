// Package provider contains concrete AI provider implementations.
//
// ClaudeProvider wraps the Claude Code CLI (ambient subscription login).
// No API key required — uses the same auth as `claude` on the command line.
package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/google/uuid"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// claudeModels is the static catalog of supported Claude models.
var claudeModels = []sdk.ModelInfo{
	{
		Provider:             "claude",
		ModelID:              "claude-opus-4-6",
		DisplayName:          "Claude Opus 4.6",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured, sdk.CapabilityAgentic},
		Tier:                 sdk.TierHigh,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "claude",
		ModelID:              "claude-sonnet-4-6",
		DisplayName:          "Claude Sonnet 4.6",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured, sdk.CapabilityAgentic},
		Tier:                 sdk.TierMedium,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
	{
		Provider:             "claude",
		ModelID:              "claude-haiku-4-5-20251001",
		DisplayName:          "Claude Haiku 4.5",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierLow,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        true,
	},
}

// ClaudeProvider wraps the Claude Code CLI for text generation, streaming,
// and structured output. Uses the ambient Claude subscription login —
// no API key required.
type ClaudeProvider struct {
	permissionMode string
}

// NewClaudeProvider returns a new ClaudeProvider with bypassPermissions mode.
func NewClaudeProvider() *ClaudeProvider {
	return &ClaudeProvider{permissionMode: "bypassPermissions"}
}

// Name returns "claude".
func (c *ClaudeProvider) Name() string {
	return "claude"
}

// findClaudeCLI locates the claude binary on PATH or in common locations.
func findClaudeCLI() (string, error) {
	if p, err := exec.LookPath("claude"); err == nil {
		return p, nil
	}
	home, _ := os.UserHomeDir()
	candidates := []string{
		home + "/.npm-global/bin/claude",
		"/usr/local/bin/claude",
		home + "/.local/bin/claude",
		home + "/node_modules/.bin/claude",
		home + "/.claude/local/claude",
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("claude CLI not found — install with: npm install -g @anthropic-ai/claude-code")
}

// Available returns true when the claude CLI is installed and reachable.
func (c *ClaudeProvider) Available(_ context.Context) (bool, error) {
	_, err := findClaudeCLI()
	return err == nil, nil
}

// ListModels returns the static Claude model catalog.
func (c *ClaudeProvider) ListModels(_ context.Context) ([]sdk.ModelInfo, error) {
	result := make([]sdk.ModelInfo, len(claudeModels))
	copy(result, claudeModels)
	return result, nil
}

// buildPrompt combines message history into a single prompt string for the CLI.
func buildPrompt(messages []sdk.Message) (string, string) {
	var systemParts []string
	var userParts []string
	for _, m := range messages {
		switch m.Role {
		case "system":
			systemParts = append(systemParts, m.Content)
		case "user":
			userParts = append(userParts, m.Content)
		case "assistant":
			userParts = append(userParts, "[Previous assistant response]\n"+m.Content)
		}
	}
	return strings.Join(systemParts, "\n"), strings.Join(userParts, "\n\n")
}

// claudeStreamMessage represents a JSONL message from the claude CLI stream-json output.
type claudeStreamMessage struct {
	Type    string `json:"type"`
	Subtype string `json:"subtype,omitempty"`
	// For assistant messages
	Role    string              `json:"role,omitempty"`
	Content []claudeContentItem `json:"content,omitempty"`
	// For result messages
	Result          *string `json:"result,omitempty"`
	StopReason      string  `json:"stop_reason,omitempty"`
	TotalCostUSD    float64 `json:"total_cost_usd,omitempty"`
	IsError         bool    `json:"is_error,omitempty"`
	StructuredOutput any    `json:"structured_output,omitempty"`
	// Usage
	Usage map[string]any `json:"usage,omitempty"`
}

// claudeContentItem represents a content block in a claude response.
type claudeContentItem struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`
}

// stripClaudeCodeEnv returns a copy of os.Environ() with CLAUDECODE removed.
func stripClaudeCodeEnv() []string {
	var env []string
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, "CLAUDECODE=") {
			env = append(env, e)
		}
	}
	return env
}

// classifyClaudeError maps error messages to ErrorKind.
func classifyClaudeError(err error) sdk.ErrorKind {
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "rate_limit") || strings.Contains(msg, "429") || strings.Contains(msg, "overloaded") {
		return sdk.ErrorRateLimit
	}
	if strings.Contains(msg, "401") || strings.Contains(msg, "403") || strings.Contains(msg, "auth") {
		return sdk.ErrorAuth
	}
	if strings.Contains(msg, "timeout") || strings.Contains(msg, "timed out") {
		return sdk.ErrorTimeout
	}
	if strings.Contains(msg, "400") || strings.Contains(msg, "invalid") {
		return sdk.ErrorInvalidRequest
	}
	if strings.Contains(msg, "not found") || strings.Contains(msg, "not installed") {
		return sdk.ErrorNotAvailable
	}
	return sdk.ErrorInternal
}

// Complete sends a prompt to the claude CLI and collects the full response.
// Uses ambient subscription login — no API key needed.
func (c *ClaudeProvider) Complete(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.CompleteOpts) (*sdk.Response, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	cliPath, err := findClaudeCLI()
	if err != nil {
		return nil, sdk.NewAgentError(err.Error(), sdk.ErrorNotAvailable, nil)
	}

	system, prompt := buildPrompt(messages)

	// Build command
	args := []string{"--output-format", "stream-json", "--verbose"}
	if system != "" {
		args = append(args, "--system-prompt", system)
	}
	if model.ModelID != "claude-opus-4-6" {
		args = append(args, "--model", model.ModelID)
	}
	args = append(args, "--permission-mode", c.permissionMode)
	args = append(args, "--max-turns", "1")

	if opts.MaxTurns > 0 {
		// Override max-turns
		args[len(args)-1] = fmt.Sprint(opts.MaxTurns)
	}

	// Structured output via --json-schema
	if opts.Schema != nil {
		schemaBytes, err := json.Marshal(opts.Schema)
		if err != nil {
			return nil, sdk.NewAgentError(fmt.Sprintf("marshaling schema: %v", err), sdk.ErrorInvalidRequest, nil)
		}
		args = append(args, "--json-schema", string(schemaBytes))
	}

	// Pass prompt via -p flag
	args = append(args, "-p", prompt)

	cmd := exec.CommandContext(ctx, cliPath, args...)
	cmd.Env = stripClaudeCodeEnv()

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, sdk.NewAgentError(
				fmt.Sprintf("claude request timed out after %.0fs", timeout),
				sdk.ErrorTimeout, nil,
			)
		}
		errMsg := stderr.String()
		if errMsg == "" {
			errMsg = err.Error()
		}
		kind := classifyClaudeError(fmt.Errorf("%s", errMsg))
		return nil, sdk.NewAgentError(fmt.Sprintf("claude CLI error: %s", errMsg), kind, nil)
	}

	// Parse stream-json output
	var textParts []string
	var resultText string
	var structuredOutput any
	var usage map[string]any

	scanner := bufio.NewScanner(&stdout)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var msg claudeStreamMessage
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			continue // skip unparseable lines
		}

		switch msg.Type {
		case "result":
			if msg.Result != nil {
				resultText = *msg.Result
			}
			if msg.StructuredOutput != nil {
				structuredOutput = msg.StructuredOutput
			}
			usage = msg.Usage
		case "assistant":
			for _, block := range msg.Content {
				if block.Type == "text" && block.Text != "" {
					textParts = append(textParts, block.Text)
				}
				if block.Type == "tool_use" && block.Name == "StructuredOutput" {
					structuredOutput = block.Input
				}
			}
		}
	}

	// Prefer result text over intermediate text
	text := resultText
	if text == "" {
		text = strings.Join(textParts, "")
	}

	resp := &sdk.Response{
		Text:           strings.TrimSpace(text),
		ModelUsed:      model,
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
		Usage:          usage,
	}

	// If structured output was captured and schema was requested, attach it
	if structuredOutput != nil && opts.Schema != nil {
		return resp, nil
	}

	return resp, nil
}

// Stream sends a prompt to the claude CLI and yields response chunks.
func (c *ClaudeProvider) Stream(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.StreamOpts) (<-chan sdk.StreamChunk, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 300.0
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))

	cliPath, err := findClaudeCLI()
	if err != nil {
		cancel()
		return nil, sdk.NewAgentError(err.Error(), sdk.ErrorNotAvailable, nil)
	}

	system, prompt := buildPrompt(messages)

	args := []string{"--output-format", "stream-json", "--verbose"}
	if system != "" {
		args = append(args, "--system-prompt", system)
	}
	if model.ModelID != "claude-opus-4-6" {
		args = append(args, "--model", model.ModelID)
	}
	args = append(args, "--permission-mode", c.permissionMode)
	args = append(args, "--max-turns", "1")
	args = append(args, "-p", prompt)

	cmd := exec.CommandContext(ctx, cliPath, args...)
	cmd.Env = stripClaudeCodeEnv()

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("creating stdout pipe: %v", err), sdk.ErrorInternal, nil)
	}

	if err := cmd.Start(); err != nil {
		cancel()
		kind := classifyClaudeError(err)
		return nil, sdk.NewAgentError(fmt.Sprintf("starting claude CLI: %v", err), kind, nil)
	}

	ch := make(chan sdk.StreamChunk)
	go func() {
		defer close(ch)
		defer cancel()
		defer cmd.Wait() //nolint:errcheck

		scanner := bufio.NewScanner(stdoutPipe)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var msg claudeStreamMessage
			if err := json.Unmarshal([]byte(line), &msg); err != nil {
				continue
			}
			if msg.Type == "assistant" {
				for _, block := range msg.Content {
					if block.Type == "text" && block.Text != "" {
						ch <- sdk.StreamChunk{Text: block.Text}
					}
				}
			}
			if msg.Type == "result" && msg.Result != nil && *msg.Result != "" {
				ch <- sdk.StreamChunk{Text: *msg.Result}
			}
		}
	}()

	return ch, nil
}

// Compile-time check that ClaudeProvider satisfies Provider.
var _ sdk.Provider = (*ClaudeProvider)(nil)
