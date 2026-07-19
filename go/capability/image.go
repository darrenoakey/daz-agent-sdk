// Package capability implements image generation, text-to-speech, and
// speech-to-text clients for the SDK's durable capability services.
package capability

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

const legacyImageProviderError = "image provider %q is actively disabled; use the Mac mini Codex image service"

func durableImageContext(parent context.Context) context.Context {
	return context.WithoutCancel(parent)
}

// ImageOpts holds optional parameters for GenerateImage. Provider, Model,
// Steps, and Timeout remain for source compatibility, but disabled selectors
// fail closed before service I/O.
type ImageOpts struct {
	Provider       string
	Model          string
	Image          string
	Images         []string
	Width          int
	Height         int
	Output         string
	Steps          int
	Tier           agentsdk.Tier
	Transparent    bool
	Timeout        time.Duration
	Config         *agentsdk.Config
	Logger         *agentsdk.ConversationLogger
	ConversationID uuid.UUID
	IdempotencyKey string
	StatePath      string
}

// inputImages returns input paths in caller order without duplicates.
func (options ImageOpts) inputImages() []string {
	paths := make([]string, 0, len(options.Images)+1)
	seen := make(map[string]struct{}, len(options.Images)+1)
	for _, path := range append([]string{options.Image}, options.Images...) {
		if path == "" {
			continue
		}
		if _, exists := seen[path]; exists {
			continue
		}
		seen[path] = struct{}{}
		paths = append(paths, path)
	}
	return paths
}

// resolveOutputPath prepares a stable destination for a completed artifact.
func resolveOutputPath(output string) (string, error) {
	if output == "" {
		file, err := os.CreateTemp("", "agent_sdk_img_*.png")
		if err != nil {
			return "", fmt.Errorf("creating image destination: %w", err)
		}
		path := file.Name()
		if err := file.Close(); err != nil {
			return "", fmt.Errorf("closing image destination: %w", err)
		}
		if err := os.Remove(path); err != nil {
			return "", fmt.Errorf("preparing image destination: %w", err)
		}
		return path, nil
	}
	if err := os.MkdirAll(filepath.Dir(output), 0o755); err != nil {
		return "", fmt.Errorf("creating image output directory: %w", err)
	}
	return output, nil
}

// validateImageRoute rejects every legacy backend selector before any I/O.
func validateImageRoute(options ImageOpts) error {
	if err := options.Config.ValidateImageConfig(); err != nil {
		return err
	}
	provider := strings.ToLower(strings.TrimSpace(options.Provider))
	if provider != "" && provider != "codex" {
		return agentsdk.NewAgentError(fmt.Sprintf(legacyImageProviderError, options.Provider), agentsdk.ErrorInvalidRequest, nil)
	}
	if strings.TrimSpace(options.Model) != "" {
		return agentsdk.NewAgentError(
			fmt.Sprintf("image model %q is actively disabled; the Mac mini Codex image service owns model selection", options.Model),
			agentsdk.ErrorInvalidRequest, nil,
		)
	}
	if options.Steps != 0 {
		return agentsdk.NewAgentError(
			"image step overrides are actively disabled; the Mac mini Codex image service owns inference settings",
			agentsdk.ErrorInvalidRequest, nil,
		)
	}
	if err := agentsdk.ValidateImageTimeout(options.Timeout); err != nil {
		return err
	}
	return nil
}

// GenerateImage submits exactly one durable job and waits for its downloaded artifact.
func GenerateImage(parent context.Context, prompt string, options ImageOpts) (*agentsdk.ImageResult, error) {
	if err := validateImageRoute(options); err != nil {
		return nil, err
	}
	if strings.TrimSpace(prompt) == "" || options.Width <= 0 || options.Height <= 0 {
		return nil, agentsdk.NewAgentError("image prompt and positive width/height are required", agentsdk.ErrorInvalidRequest, nil)
	}
	if options.ConversationID == uuid.Nil {
		options.ConversationID = uuid.New()
	}
	logImageRequest(options, prompt)
	sources, err := encodeSourceImages(options.inputImages())
	if err != nil {
		return nil, err
	}
	body, err := encodeImageJobBody(prompt, options, sources)
	if err != nil {
		return nil, err
	}
	operationContext := durableImageContext(parent)
	result, err := completeImageOperation(operationContext, prompt, options, body)
	if err == nil && options.Logger != nil {
		options.Logger.LogEvent("image_complete", map[string]any{
			"job_id": result.JobID, "path": result.Path, "provider": "codex", "status": result.Status,
		})
	}
	return result, err
}

func logImageRequest(options ImageOpts, prompt string) {
	if options.Logger == nil {
		return
	}
	options.Logger.LogEvent("image_request", map[string]any{
		"prompt": prompt, "provider": "codex", "chain": []string{"codex"},
		"width": options.Width, "height": options.Height, "transparent": options.Transparent,
	})
}
