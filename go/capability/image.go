// Package capability implements image generation, text-to-speech, and
// speech-to-text as subprocess/HTTP wrappers around external tools.
package capability

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// ollamaImageModel is the default Ollama model used for image generation.
const ollamaImageModel = "x/z-image-turbo"

// ollamaImageModelInfo describes the Ollama image model for result metadata.
var ollamaImageModelInfo = agentsdk.ModelInfo{
	Provider:     "ollama",
	ModelID:      "z-image-turbo",
	DisplayName:  "Ollama z-image-turbo",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilityImage},
	Tier:         agentsdk.TierHigh,
}

// ImageOpts holds optional parameters for GenerateImage.
type ImageOpts struct {
	// Width is the requested image width in pixels (logged only; Ollama
	// does not expose a width parameter for image models).
	Width int
	// Height is the requested image height in pixels (logged only).
	Height int
	// Output is the file path to write the generated image to.
	// When empty, a temp file is created.
	Output string
	// Tier selects the quality tier (used to look up step count for logging).
	Tier agentsdk.Tier
	// Transparent requests background removal. Not yet supported in Go;
	// a warning is logged if set.
	Transparent bool
	// Timeout is the HTTP request timeout. Zero means 120 seconds.
	Timeout time.Duration
	// Config is the agent-sdk config. nil means load defaults.
	Config *agentsdk.Config
	// Logger is an optional conversation logger for event recording.
	Logger *agentsdk.ConversationLogger
	// ConversationID ties the result to a conversation. Zero means generate one.
	ConversationID uuid.UUID
}

// ollamaGenerateRequest is the JSON body for POST /api/generate.
type ollamaGenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

// ollamaGenerateResponse is the JSON body returned by POST /api/generate
// for image models.
type ollamaGenerateResponse struct {
	Model string `json:"model"`
	Image string `json:"image"`
	Done  bool   `json:"done"`
}

// ollamaBaseURL returns the Ollama base URL from config or the default.
func ollamaBaseURL(cfg *agentsdk.Config) string {
	if cfg != nil {
		if p, ok := cfg.Providers["ollama"]; ok {
			if u, ok := p["base_url"].(string); ok && u != "" {
				return u
			}
		}
	}
	return "http://localhost:11434"
}

// ollamaImageModelName returns the model name to use, prefixed with "x/"
// for Ollama's namespace convention.
func ollamaImageModelName(cfg *agentsdk.Config) string {
	if cfg != nil && cfg.Image.Model != "" {
		return "x/" + cfg.Image.Model
	}
	return ollamaImageModel
}

// GenerateImage generates an image using an Ollama image model.
// The image is written to opts.Output (or a temporary file) and the result
// includes the file path and model metadata.
func GenerateImage(ctx context.Context, prompt string, opts ImageOpts) (*agentsdk.ImageResult, error) {
	cfg := opts.Config
	if cfg == nil {
		var err error
		cfg, err = agentsdk.LoadConfig()
		if err != nil {
			return nil, fmt.Errorf("loading config: %w", err)
		}
	}

	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	tier := opts.Tier
	if tier == "" {
		tier = agentsdk.TierHigh
	}

	convID := opts.ConversationID
	if convID == uuid.Nil {
		convID = uuid.New()
	}

	model := ollamaImageModelName(cfg)
	steps := agentsdk.GetImageSteps(tier, cfg)
	baseURL := ollamaBaseURL(cfg)

	// Log the request
	if opts.Logger != nil {
		opts.Logger.LogEvent("image_request", map[string]any{
			"prompt":      prompt,
			"width":       opts.Width,
			"height":      opts.Height,
			"model":       model,
			"tier":        string(tier),
			"steps":       steps,
			"transparent": opts.Transparent,
		})
	}

	if opts.Width != 0 || opts.Height != 0 {
		// Ollama image models don't expose width/height params; log it
		if opts.Logger != nil {
			opts.Logger.LogEvent("image_dimensions_ignored", map[string]any{
				"width":  opts.Width,
				"height": opts.Height,
				"reason": "Ollama image models generate at their default resolution",
			})
		}
	}

	// Build the HTTP request
	reqBody := ollamaGenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	url := baseURL + "/api/generate"
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("creating HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		if reqCtx.Err() == context.DeadlineExceeded {
			return nil, agentsdk.NewAgentError(
				fmt.Sprintf("image generation timed out after %s", timeout),
				agentsdk.ErrorTimeout, nil,
			)
		}
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("Ollama request failed: %v", err),
			agentsdk.ErrorNotAvailable, nil,
		)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("Ollama returned HTTP %d: %s", resp.StatusCode, string(respBytes)),
			agentsdk.ErrorInternal, nil,
		)
	}

	var genResp ollamaGenerateResponse
	if err := json.Unmarshal(respBytes, &genResp); err != nil {
		return nil, fmt.Errorf("unmarshaling response: %w", err)
	}

	if genResp.Image == "" {
		return nil, agentsdk.NewAgentError(
			"Ollama returned no image data",
			agentsdk.ErrorInternal, nil,
		)
	}

	// Decode base64 image data
	imageData, err := base64.StdEncoding.DecodeString(genResp.Image)
	if err != nil {
		return nil, fmt.Errorf("decoding base64 image data: %w", err)
	}

	// Determine output path
	outputPath := opts.Output
	if outputPath == "" {
		tmpFile, err := os.CreateTemp("", "agent_sdk_img_*.png")
		if err != nil {
			return nil, fmt.Errorf("creating temp file: %w", err)
		}
		outputPath = tmpFile.Name()
		tmpFile.Close()
	} else {
		// Ensure parent directory exists
		dir := filepath.Dir(outputPath)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("creating output directory: %w", err)
		}
	}

	if err := os.WriteFile(outputPath, imageData, 0o644); err != nil {
		return nil, fmt.Errorf("writing image to %s: %w", outputPath, err)
	}

	// Transparent background removal is not supported in Go
	if opts.Transparent {
		if opts.Logger != nil {
			opts.Logger.LogEvent("image_transparent_unsupported", map[string]any{
				"reason": "background removal requires Python BiRefNet; not available in Go",
			})
		}
	}

	if opts.Logger != nil {
		opts.Logger.LogEvent("image_complete", map[string]any{
			"path": outputPath,
		})
	}

	width := opts.Width
	height := opts.Height

	return &agentsdk.ImageResult{
		Path:           outputPath,
		ModelUsed:      ollamaImageModelInfo,
		ConversationID: convID,
		Prompt:         prompt,
		Width:          width,
		Height:         height,
	}, nil
}
