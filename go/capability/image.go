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
	"google.golang.org/genai"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// ollamaImageModel is the default Ollama model used for image generation.
const ollamaImageModel = "x/z-image-turbo"

// sparkImageModel is the default Spark model used for image generation.
const sparkImageModel = "z-image-turbo"

// nanoBananaImageModel is the Gemini model for nano-banana-2 image generation.
const nanoBananaImageModel = "gemini-3.1-flash-image-preview"

// ollamaImageModelInfo describes the Ollama image model for result metadata.
var ollamaImageModelInfo = agentsdk.ModelInfo{
	Provider:     "ollama",
	ModelID:      "z-image-turbo",
	DisplayName:  "Ollama z-image-turbo",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilityImage},
	Tier:         agentsdk.TierHigh,
}

// sparkModelInfo describes the Spark image model for result metadata.
var sparkModelInfo = agentsdk.ModelInfo{
	Provider:     "spark",
	ModelID:      "z-image-turbo",
	DisplayName:  "Spark Z-Image Turbo (FLUX.1-schnell)",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilityImage},
	Tier:         agentsdk.TierHigh,
}

// nanoBananaModelInfo describes the Nano Banana 2 (Gemini) model for result metadata.
var nanoBananaModelInfo = agentsdk.ModelInfo{
	Provider:     "gemini",
	ModelID:      nanoBananaImageModel,
	DisplayName:  "Nano Banana 2",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilityImage},
	Tier:         agentsdk.TierHigh,
}

// ImageOpts holds optional parameters for GenerateImage.
type ImageOpts struct {
	// Provider selects the backend: "spark", "ollama", "nano-banana-2".
	// Empty means "spark" (default).
	Provider string
	// Model selects the image model on the backend. For spark, valid values
	// are "z-image-turbo" and "flux-schnell". Empty means use config default
	// or "z-image-turbo".
	Model string
	// Image is the input image file path for image-to-image generation.
	// Empty means text-to-image.
	Image string
	// Width is the requested image width in pixels.
	Width int
	// Height is the requested image height in pixels.
	Height int
	// Output is the file path to write the generated image to.
	// When empty, a temp file is created.
	Output string
	// Steps overrides the inference step count for spark. Zero means
	// derive from Tier via config (typically 4 for high, 2 for low).
	Steps int
	// Tier selects the quality tier (used to look up step count when Steps is 0).
	Tier agentsdk.Tier
	// Transparent requests background removal via spark arbiter.
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

// arbiterJobRequest is the JSON body for POST /v1/jobs.
type arbiterJobRequest struct {
	Type   string         `json:"type"`
	Params map[string]any `json:"params"`
}

// arbiterJobResponse is returned by POST /v1/jobs.
type arbiterJobResponse struct {
	JobID string `json:"job_id"`
}

// arbiterJobStatus is returned by GET /v1/jobs/{job_id}.
type arbiterJobStatus struct {
	Status string         `json:"status"`
	Result map[string]any `json:"result"`
	Error  string         `json:"error,omitempty"`
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

// arbiterBaseURL returns the arbiter job server URL from config, env, or default.
// All spark image generation goes through the arbiter — there is no direct
// spark:8100 endpoint anymore.
func arbiterBaseURL(cfg *agentsdk.Config) string {
	if cfg != nil {
		if p, ok := cfg.Providers["arbiter"]; ok {
			if u, ok := p["base_url"].(string); ok && u != "" {
				return u
			}
		}
	}
	if u := os.Getenv("ARBITER_URL"); u != "" {
		return u
	}
	return "http://spark:8400"
}

// ollamaImageModelName returns the model name to use, prefixed with "x/"
// for Ollama's namespace convention.
func ollamaImageModelName(cfg *agentsdk.Config) string {
	if cfg != nil && cfg.Image.Model != "" {
		return "x/" + cfg.Image.Model
	}
	return ollamaImageModel
}

// sparkModelName resolves the image model to use on spark. Priority:
// 1. opts.Model (explicit override)
// 2. cfg.Image.Model (config file)
// 3. sparkImageModel constant ("z-image-turbo")
func sparkModelName(opts ImageOpts, cfg *agentsdk.Config) string {
	if opts.Model != "" {
		return opts.Model
	}
	if cfg != nil && cfg.Image.Model != "" {
		return cfg.Image.Model
	}
	return sparkImageModel
}

// sparkModelInfoFor returns a ModelInfo reflecting the actual model used on spark.
func sparkModelInfoFor(modelName string) agentsdk.ModelInfo {
	info := sparkModelInfo
	info.ModelID = modelName
	if modelName == "flux-schnell" {
		info.DisplayName = "Spark FLUX.1-schnell"
	} else {
		info.DisplayName = "Spark Z-Image Turbo (FLUX.1-schnell)"
	}
	return info
}

// closestAspectRatio maps width and height to the nearest standard aspect ratio
// string supported by the Gemini image API.
func closestAspectRatio(width, height int) string {
	if width <= 0 || height <= 0 {
		return "1:1"
	}
	ratio := float64(width) / float64(height)

	// Standard ratios and their float values
	type candidate struct {
		label string
		value float64
	}
	candidates := []candidate{
		{"1:1", 1.0},
		{"4:3", 4.0 / 3.0},
		{"3:4", 3.0 / 4.0},
		{"16:9", 16.0 / 9.0},
		{"9:16", 9.0 / 16.0},
	}

	best := candidates[0].label
	bestDiff := abs64(ratio - candidates[0].value)
	for _, c := range candidates[1:] {
		if d := abs64(ratio - c.value); d < bestDiff {
			bestDiff = d
			best = c.label
		}
	}
	return best
}

// abs64 returns the absolute value of a float64.
func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// imageSize maps the maximum dimension to a size bucket label.
func imageSize(width, height int) string {
	max := width
	if height > max {
		max = height
	}
	switch {
	case max <= 512:
		return "0.5K"
	case max <= 1024:
		return "1K"
	case max <= 2048:
		return "2K"
	default:
		return "4K"
	}
}

// resolveOutputPath determines the output file path and ensures the parent
// directory exists. Returns the resolved path or an error.
func resolveOutputPath(output string) (string, error) {
	if output == "" {
		tmpFile, err := os.CreateTemp("", "agent_sdk_img_*.png")
		if err != nil {
			return "", fmt.Errorf("creating temp file: %w", err)
		}
		p := tmpFile.Name()
		tmpFile.Close()
		return p, nil
	}
	dir := filepath.Dir(output)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("creating output directory: %w", err)
	}
	return output, nil
}

// writeImageData writes raw image bytes to a file and returns an error if it fails.
func writeImageData(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}

// generateOllama performs image generation using the local Ollama server.
func generateOllama(ctx context.Context, prompt string, opts ImageOpts, cfg *agentsdk.Config, timeout time.Duration) (*agentsdk.ImageResult, error) {
	model := ollamaImageModelName(cfg)
	baseURL := ollamaBaseURL(cfg)

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

	imageData, err := base64.StdEncoding.DecodeString(genResp.Image)
	if err != nil {
		return nil, fmt.Errorf("decoding base64 image data: %w", err)
	}

	outputPath, err := resolveOutputPath(opts.Output)
	if err != nil {
		return nil, err
	}
	if err := writeImageData(outputPath, imageData); err != nil {
		return nil, fmt.Errorf("writing image to %s: %w", outputPath, err)
	}

	return &agentsdk.ImageResult{
		Path:      outputPath,
		ModelUsed: ollamaImageModelInfo,
		Prompt:    prompt,
		Width:     opts.Width,
		Height:    opts.Height,
	}, nil
}

// generateSpark submits an image generation job to the arbiter and polls
// for completion. All spark image generation goes through arbiter (spark:8400).
func generateSpark(ctx context.Context, prompt string, opts ImageOpts, cfg *agentsdk.Config, timeout time.Duration, steps int) (*agentsdk.ImageResult, error) {
	model := sparkModelName(opts, cfg)
	arbURL := arbiterBaseURL(cfg)

	// Build job params
	params := map[string]any{
		"prompt": prompt,
		"width":  opts.Width,
		"height": opts.Height,
		"steps":  steps,
	}

	jobType := "image-generate"
	if opts.Image != "" {
		jobType = "image-edit"
		inputData, err := os.ReadFile(opts.Image)
		if err != nil {
			return nil, fmt.Errorf("reading input image %s: %w", opts.Image, err)
		}
		params["image"] = base64.StdEncoding.EncodeToString(inputData)
	}

	imageData, err := submitArbiterImageJob(ctx, arbURL, jobType, model, params, timeout)
	if err != nil {
		return nil, err
	}

	outputPath, err := resolveOutputPath(opts.Output)
	if err != nil {
		return nil, err
	}
	if err := writeImageData(outputPath, imageData); err != nil {
		return nil, fmt.Errorf("writing image to %s: %w", outputPath, err)
	}

	if opts.Transparent {
		newPath, err := removeBackgroundSpark(ctx, outputPath, cfg, timeout)
		if err != nil {
			// Log but do not fail — return the non-transparent image
			_ = err
		} else {
			outputPath = newPath
		}
	}

	return &agentsdk.ImageResult{
		Path:      outputPath,
		ModelUsed: sparkModelInfoFor(model),
		Prompt:    prompt,
		Width:     opts.Width,
		Height:    opts.Height,
	}, nil
}

// arbiterJobRequestWithModel extends arbiterJobRequest with a top-level model field.
type arbiterJobRequestWithModel struct {
	Type   string         `json:"type"`
	Model  string         `json:"model,omitempty"`
	Params map[string]any `json:"params"`
}

// submitArbiterImageJob submits a job to the arbiter, polls for completion,
// and returns the base64-decoded image data from the result.
func submitArbiterImageJob(ctx context.Context, arbURL, jobType, model string, params map[string]any, timeout time.Duration) ([]byte, error) {
	jobReq := arbiterJobRequestWithModel{
		Type:   jobType,
		Model:  model,
		Params: params,
	}
	jobBytes, err := json.Marshal(jobReq)
	if err != nil {
		return nil, fmt.Errorf("marshaling arbiter job request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	submitReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, arbURL+"/v1/jobs", bytes.NewReader(jobBytes))
	if err != nil {
		return nil, fmt.Errorf("creating arbiter submit request: %w", err)
	}
	submitReq.Header.Set("Content-Type", "application/json")

	submitResp, err := http.DefaultClient.Do(submitReq)
	if err != nil {
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("arbiter unreachable at %s: %v", arbURL, err),
			agentsdk.ErrorNotAvailable, nil,
		)
	}
	defer submitResp.Body.Close()

	submitBody, err := io.ReadAll(submitResp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading arbiter submit response: %w", err)
	}
	if submitResp.StatusCode != http.StatusOK && submitResp.StatusCode != http.StatusCreated {
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("arbiter submit returned HTTP %d: %s", submitResp.StatusCode, string(submitBody)),
			agentsdk.ErrorInternal, nil,
		)
	}

	var jobResp arbiterJobResponse
	if err := json.Unmarshal(submitBody, &jobResp); err != nil {
		return nil, fmt.Errorf("unmarshaling arbiter job response: %w", err)
	}
	if jobResp.JobID == "" {
		return nil, agentsdk.NewAgentError("arbiter returned no job_id", agentsdk.ErrorInternal, nil)
	}

	// Poll for completion
	pollInterval := 500 * time.Millisecond
	for {
		select {
		case <-reqCtx.Done():
			return nil, agentsdk.NewAgentError(
				fmt.Sprintf("arbiter %s job timed out", jobType),
				agentsdk.ErrorTimeout, nil,
			)
		case <-time.After(pollInterval):
		}

		pollReq, err := http.NewRequestWithContext(reqCtx, http.MethodGet,
			fmt.Sprintf("%s/v1/jobs/%s", arbURL, jobResp.JobID), nil)
		if err != nil {
			return nil, fmt.Errorf("creating arbiter poll request: %w", err)
		}

		pollResp, err := http.DefaultClient.Do(pollReq)
		if err != nil {
			continue // transient; keep polling
		}
		pollBody, err := io.ReadAll(pollResp.Body)
		pollResp.Body.Close()
		if err != nil {
			continue
		}

		var status arbiterJobStatus
		if err := json.Unmarshal(pollBody, &status); err != nil {
			continue
		}

		switch status.Status {
		case "completed":
			var resultB64 string
			if v, ok := status.Result["image"].(string); ok {
				resultB64 = v
			} else if v, ok := status.Result["data"].(string); ok {
				resultB64 = v
			}
			if resultB64 == "" {
				return nil, agentsdk.NewAgentError(
					fmt.Sprintf("arbiter %s job completed but no image in result", jobType),
					agentsdk.ErrorInternal, nil,
				)
			}
			imageData, err := base64.StdEncoding.DecodeString(resultB64)
			if err != nil {
				return nil, fmt.Errorf("decoding arbiter result image: %w", err)
			}
			return imageData, nil

		case "failed":
			return nil, agentsdk.NewAgentError(
				fmt.Sprintf("arbiter %s job failed: %s", jobType, status.Error),
				agentsdk.ErrorInternal, nil,
			)
		}
		// still running — keep polling
	}
}

// removeBackgroundSpark// removeBackgroundSpark submits a background-remove job to the arbiter and
// waits for completion, returning the path to the result image file.
func removeBackgroundSpark(ctx context.Context, imagePath string, cfg *agentsdk.Config, timeout time.Duration) (string, error) {
	rawImage, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("reading image for background removal: %w", err)
	}

	b64Image := base64.StdEncoding.EncodeToString(rawImage)
	arbURL := arbiterBaseURL(cfg)

	jobReq := arbiterJobRequest{
		Type: "background-remove",
		Params: map[string]any{
			"image": b64Image,
		},
	}
	jobBytes, err := json.Marshal(jobReq)
	if err != nil {
		return "", fmt.Errorf("marshaling arbiter job request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	submitReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, arbURL+"/v1/jobs", bytes.NewReader(jobBytes))
	if err != nil {
		return "", fmt.Errorf("creating arbiter submit request: %w", err)
	}
	submitReq.Header.Set("Content-Type", "application/json")

	submitResp, err := http.DefaultClient.Do(submitReq)
	if err != nil {
		return "", agentsdk.NewAgentError(
			fmt.Sprintf("arbiter submit failed: %v", err),
			agentsdk.ErrorNotAvailable, nil,
		)
	}
	defer submitResp.Body.Close()

	submitBody, err := io.ReadAll(submitResp.Body)
	if err != nil {
		return "", fmt.Errorf("reading arbiter submit response: %w", err)
	}
	if submitResp.StatusCode != http.StatusOK && submitResp.StatusCode != http.StatusCreated {
		return "", agentsdk.NewAgentError(
			fmt.Sprintf("arbiter submit returned HTTP %d: %s", submitResp.StatusCode, string(submitBody)),
			agentsdk.ErrorInternal, nil,
		)
	}

	var jobResp arbiterJobResponse
	if err := json.Unmarshal(submitBody, &jobResp); err != nil {
		return "", fmt.Errorf("unmarshaling arbiter job response: %w", err)
	}
	if jobResp.JobID == "" {
		return "", agentsdk.NewAgentError("arbiter returned no job_id", agentsdk.ErrorInternal, nil)
	}

	// Poll for completion
	pollInterval := 500 * time.Millisecond
	for {
		select {
		case <-reqCtx.Done():
			return "", agentsdk.NewAgentError("background removal timed out", agentsdk.ErrorTimeout, nil)
		case <-time.After(pollInterval):
		}

		pollReq, err := http.NewRequestWithContext(reqCtx, http.MethodGet,
			fmt.Sprintf("%s/v1/jobs/%s", arbURL, jobResp.JobID), nil)
		if err != nil {
			return "", fmt.Errorf("creating arbiter poll request: %w", err)
		}

		pollResp, err := http.DefaultClient.Do(pollReq)
		if err != nil {
			continue // transient; keep polling
		}
		pollBody, err := io.ReadAll(pollResp.Body)
		pollResp.Body.Close()
		if err != nil {
			continue
		}

		var status arbiterJobStatus
		if err := json.Unmarshal(pollBody, &status); err != nil {
			continue
		}

		switch status.Status {
		case "completed":
			// Extract result image from result.image or result.data
			var resultB64 string
			if v, ok := status.Result["image"].(string); ok {
				resultB64 = v
			} else if v, ok := status.Result["data"].(string); ok {
				resultB64 = v
			}
			if resultB64 == "" {
				return "", agentsdk.NewAgentError("arbiter job completed but no image in result", agentsdk.ErrorInternal, nil)
			}
			resultData, err := base64.StdEncoding.DecodeString(resultB64)
			if err != nil {
				return "", fmt.Errorf("decoding arbiter result image: %w", err)
			}
			// Overwrite the same path with the transparent image
			if err := os.WriteFile(imagePath, resultData, 0o644); err != nil {
				return "", fmt.Errorf("writing transparent image: %w", err)
			}
			return imagePath, nil

		case "failed":
			return "", agentsdk.NewAgentError(
				fmt.Sprintf("arbiter background-remove job failed: %s", status.Error),
				agentsdk.ErrorInternal, nil,
			)
		}
		// still running — keep polling
	}
}

// generateNanoBanana performs image generation using the Gemini flash image model.
func generateNanoBanana(ctx context.Context, prompt string, opts ImageOpts, timeout time.Duration) (*agentsdk.ImageResult, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		return nil, agentsdk.NewAgentError(
			"GEMINI_API_KEY or GOOGLE_API_KEY must be set for nano-banana-2",
			agentsdk.ErrorAuth, nil,
		)
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	client, err := genai.NewClient(reqCtx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("creating Gemini client: %v", err),
			agentsdk.ErrorNotAvailable, nil,
		)
	}

	aspectRatio := closestAspectRatio(opts.Width, opts.Height)

	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{"IMAGE", "TEXT"},
		ImageConfig: &genai.ImageConfig{
			AspectRatio: aspectRatio,
		},
	}

	// Build contents — for i2i prepend the input image as InlineData
	var contents []*genai.Content
	if opts.Image != "" {
		inputData, err := os.ReadFile(opts.Image)
		if err != nil {
			return nil, fmt.Errorf("reading input image %s: %w", opts.Image, err)
		}
		contents = []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					genai.NewPartFromBytes(inputData, "image/png"),
					genai.NewPartFromText(prompt),
				},
			},
		}
	} else {
		contents = genai.Text(prompt)
	}

	result, err := client.Models.GenerateContent(reqCtx, nanoBananaImageModel, contents, config)
	if err != nil {
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("Gemini GenerateContent failed: %v", err),
			agentsdk.ErrorInternal, nil,
		)
	}

	// Extract the first inline image from the response candidates
	var imageData []byte
	for _, cand := range result.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part.InlineData != nil && len(part.InlineData.Data) > 0 {
				imageData = part.InlineData.Data
				break
			}
		}
		if imageData != nil {
			break
		}
	}

	if imageData == nil {
		return nil, agentsdk.NewAgentError(
			"nano-banana-2 returned no image data",
			agentsdk.ErrorInternal, nil,
		)
	}

	outputPath, err := resolveOutputPath(opts.Output)
	if err != nil {
		return nil, err
	}
	if err := writeImageData(outputPath, imageData); err != nil {
		return nil, fmt.Errorf("writing image to %s: %w", outputPath, err)
	}

	return &agentsdk.ImageResult{
		Path:      outputPath,
		ModelUsed: nanoBananaModelInfo,
		Prompt:    prompt,
		Width:     opts.Width,
		Height:    opts.Height,
	}, nil
}

// buildFallbackChain constructs the ordered list of providers to try.
// The primary provider is first; config fallbacks follow, with the primary
// excluded from the fallback list to prevent duplicates.
func buildFallbackChain(primary string, cfg *agentsdk.Config) []string {
	chain := []string{primary}
	for _, fb := range cfg.Image.Fallback {
		if fb != primary {
			chain = append(chain, fb)
		}
	}
	return chain
}

// GenerateImage generates an image using a provider backend with automatic
// fallback. The provider is selected via opts.Provider (default: "spark").
// On failure, the config fallback chain is tried in order.
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

	steps := opts.Steps
	if steps <= 0 {
		steps = agentsdk.GetImageSteps(tier, cfg)
	}

	primary := opts.Provider
	if primary == "" {
		primary = "spark"
	}

	chain := buildFallbackChain(primary, cfg)

	if opts.Logger != nil {
		opts.Logger.LogEvent("image_request", map[string]any{
			"prompt":      prompt,
			"provider":    primary,
			"chain":       chain,
			"width":       opts.Width,
			"height":      opts.Height,
			"tier":        string(tier),
			"steps":       steps,
			"transparent": opts.Transparent,
		})
	}

	var lastErr error
	for _, provider := range chain {
		var result *agentsdk.ImageResult
		var err error

		switch provider {
		case "spark":
			result, err = generateSpark(ctx, prompt, opts, cfg, timeout, steps)
		case "ollama":
			result, err = generateOllama(ctx, prompt, opts, cfg, timeout)
		case "nano-banana-2":
			result, err = generateNanoBanana(ctx, prompt, opts, timeout)
		default:
			err = agentsdk.NewAgentError(
				fmt.Sprintf("unknown image provider: %s", provider),
				agentsdk.ErrorInvalidRequest, nil,
			)
		}

		if err == nil {
			result.ConversationID = convID
			if opts.Logger != nil {
				opts.Logger.LogEvent("image_complete", map[string]any{
					"path":     result.Path,
					"provider": provider,
				})
			}
			return result, nil
		}

		lastErr = err
		if opts.Logger != nil {
			opts.Logger.LogEvent("image_provider_failed", map[string]any{
				"provider": provider,
				"error":    err.Error(),
			})
		}
	}

	return nil, lastErr
}
