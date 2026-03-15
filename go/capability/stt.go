package capability

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/google/uuid"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// defaultModelSize is the whisper model size when the caller does not specify.
const defaultModelSize = "small"

// localSTTModelInfo describes the local Whisper subprocess tool.
var localSTTModelInfo = agentsdk.ModelInfo{
	Provider:     "local",
	ModelID:      "whisper",
	DisplayName:  "Local Whisper STT",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilitySTT},
	Tier:         agentsdk.TierHigh,
}

// TranscribeOpts holds optional parameters for Transcribe.
type TranscribeOpts struct {
	// ModelSize is the whisper model variant (base, small, large-v3-turbo).
	// Empty means "small".
	ModelSize string
	// Language is the expected language code. Empty means auto-detect.
	Language string
	// Timeout is the subprocess timeout. Zero means 120 seconds.
	Timeout time.Duration
	// Logger is an optional conversation logger for event recording.
	Logger *agentsdk.ConversationLogger
	// ConversationID ties the result to a conversation. Zero means generate one.
	ConversationID uuid.UUID
}

// buildSTTCommand constructs the whisper subprocess argument list.
func buildSTTCommand(audioPath string, modelSize string, language string) []string {
	cmd := []string{
		"whisper",
		audioPath,
		"--model", modelSize,
		"--output_format", "txt",
	}
	if language != "" {
		cmd = append(cmd, "--language", language)
	}
	return cmd
}

// Transcribe converts audio to text using the local whisper subprocess tool.
// It returns the transcribed text.
func Transcribe(ctx context.Context, audioPath string, opts TranscribeOpts) (string, error) {
	// Verify audio file exists
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		return "", agentsdk.NewAgentError(
			fmt.Sprintf("audio file does not exist: %s", audioPath),
			agentsdk.ErrorInvalidRequest, nil,
		)
	}

	modelSize := opts.ModelSize
	if modelSize == "" {
		modelSize = defaultModelSize
	}

	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	if opts.Logger != nil {
		opts.Logger.LogEvent("stt_request", map[string]any{
			"audio":      audioPath,
			"model_size": modelSize,
			"language":   opts.Language,
		})
	}

	args := buildSTTCommand(audioPath, modelSize, opts.Language)
	stdout, err := runSubprocessWithOutput(ctx, args, timeout, "whisper")
	if err != nil {
		return "", err
	}

	text := strings.TrimSpace(stdout)

	if opts.Logger != nil {
		opts.Logger.LogEvent("stt_complete", map[string]any{
			"audio":       audioPath,
			"text_length": len(text),
		})
	}

	return text, nil
}

// runSubprocessWithOutput executes a command with a timeout, capturing
// both stdout and stderr. Returns stdout text on success, or an AgentError
// on failure or timeout.
func runSubprocessWithOutput(ctx context.Context, args []string, timeout time.Duration, label string) (string, error) {
	cmdCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, args[0], args[1:]...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		if cmdCtx.Err() == context.DeadlineExceeded {
			return "", agentsdk.NewAgentError(
				fmt.Sprintf("%s timed out after %s", label, timeout),
				agentsdk.ErrorTimeout, nil,
			)
		}
		if execErr, ok := err.(*exec.Error); ok {
			return "", agentsdk.NewAgentError(
				fmt.Sprintf("%s could not be started: %v", label, execErr),
				agentsdk.ErrorNotAvailable, nil,
			)
		}
		stderrText := stderr.String()
		exitCode := -1
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		}
		return "", agentsdk.NewAgentError(
			fmt.Sprintf("%s failed (exit %d): %s", label, exitCode, stderrText),
			agentsdk.ErrorInternal, nil,
		)
	}
	return stdout.String(), nil
}
