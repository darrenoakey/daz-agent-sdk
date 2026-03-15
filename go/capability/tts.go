package capability

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"

	"github.com/google/uuid"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// defaultVoice is used when the caller does not specify a voice.
const defaultVoice = "aiden"

// defaultTTSSuffix is the audio format produced by the local tts tool.
const defaultTTSSuffix = ".wav"

// localTTSModelInfo describes the local TTS subprocess tool.
var localTTSModelInfo = agentsdk.ModelInfo{
	Provider:     "local",
	ModelID:      "tts",
	DisplayName:  "Local TTS",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilityTTS},
	Tier:         agentsdk.TierHigh,
}

// SpeakOpts holds optional parameters for SynthesizeSpeech.
type SpeakOpts struct {
	// Voice is the TTS voice name. Empty means "aiden".
	Voice string
	// Output is the file path to write the audio to.
	// When empty, a temp file is created.
	Output string
	// Speed is the speech rate multiplier. Zero means 1.0.
	Speed float64
	// Timeout is the subprocess timeout. Zero means 120 seconds.
	Timeout time.Duration
	// Logger is an optional conversation logger for event recording.
	Logger *agentsdk.ConversationLogger
	// ConversationID ties the result to a conversation. Zero means generate one.
	ConversationID uuid.UUID
}

// buildTTSCommand constructs the tts subprocess argument list.
func buildTTSCommand(text string, voice string, output string, speed float64) []string {
	return []string{
		"tts",
		"tts",
		"--text", text,
		"--voice", voice,
		"--output", output,
		"--speed", strconv.FormatFloat(speed, 'f', -1, 64),
	}
}

// SynthesizeSpeech converts text to speech audio using the local tts
// subprocess tool. It writes the result to opts.Output (or a temporary file)
// and returns an AudioResult with the file path.
func SynthesizeSpeech(ctx context.Context, text string, opts SpeakOpts) (*agentsdk.AudioResult, error) {
	voice := opts.Voice
	if voice == "" {
		voice = defaultVoice
	}

	speed := opts.Speed
	if speed == 0 {
		speed = 1.0
	}

	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	convID := opts.ConversationID
	if convID == uuid.Nil {
		convID = uuid.New()
	}

	// Determine output path
	outputPath := opts.Output
	if outputPath == "" {
		tmpFile, err := os.CreateTemp("", "agent_sdk_tts_*"+defaultTTSSuffix)
		if err != nil {
			return nil, fmt.Errorf("creating temp file: %w", err)
		}
		outputPath = tmpFile.Name()
		tmpFile.Close()
	} else {
		dir := filepath.Dir(outputPath)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("creating output directory: %w", err)
		}
	}

	if opts.Logger != nil {
		opts.Logger.LogEvent("tts_request", map[string]any{
			"text_length": len(text),
			"voice":       voice,
			"output":      outputPath,
			"speed":       speed,
		})
	}

	args := buildTTSCommand(text, voice, outputPath, speed)
	if err := runSubprocess(ctx, args, timeout, "tts"); err != nil {
		return nil, err
	}

	if opts.Logger != nil {
		opts.Logger.LogEvent("tts_complete", map[string]any{
			"path":  outputPath,
			"voice": voice,
		})
	}

	return &agentsdk.AudioResult{
		Path:           outputPath,
		ModelUsed:      localTTSModelInfo,
		ConversationID: convID,
		Text:           text,
		Voice:          voice,
	}, nil
}

// runSubprocess executes a command with a timeout, capturing stderr for
// error messages. Returns an AgentError on failure or timeout.
func runSubprocess(ctx context.Context, args []string, timeout time.Duration, label string) error {
	cmdCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, args[0], args[1:]...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		if cmdCtx.Err() == context.DeadlineExceeded {
			return agentsdk.NewAgentError(
				fmt.Sprintf("%s timed out after %s", label, timeout),
				agentsdk.ErrorTimeout, nil,
			)
		}
		// Check if the binary was not found
		if execErr, ok := err.(*exec.Error); ok {
			return agentsdk.NewAgentError(
				fmt.Sprintf("%s could not be started: %v", label, execErr),
				agentsdk.ErrorNotAvailable, nil,
			)
		}
		stderrText := stderr.String()
		exitCode := -1
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		}
		return agentsdk.NewAgentError(
			fmt.Sprintf("%s failed (exit %d): %s", label, exitCode, stderrText),
			agentsdk.ErrorInternal, nil,
		)
	}
	return nil
}
