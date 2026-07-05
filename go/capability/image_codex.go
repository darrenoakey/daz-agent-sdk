package capability

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// codexImageDir is where codex writes generated PNGs, one folder per session.
func codexImageDir() string {
	return filepath.Join(os.Getenv("HOME"), ".codex", "generated_images")
}

const imageServiceURL = "http://10.0.0.46:8830"

// codexModelInfo describes the shared mac mini image service for result metadata.
var codexModelInfo = agentsdk.ModelInfo{
	Provider:     "codex",
	ModelID:      "macmini-image-service",
	DisplayName:  "Mac mini image generation service",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilityImage},
	Tier:         agentsdk.TierHigh,
}

// codexImagePrompt returns the prompt that reliably gets the codex CLI to
// invoke its built-in image_gen.imagegen tool — never the skill, the arbiter,
// or an external shell script.
func codexImagePrompt(prompt string, width, height int, hasInputs bool, numInputs int) string {
	aspectHint := " (square)"
	if width > height {
		aspectHint = " (landscape orientation)"
	} else if height > width {
		aspectHint = " (portrait orientation)"
	}

	if hasInputs {
		action := "EDIT the attached image"
		if numInputs > 1 {
			action = fmt.Sprintf("COMBINE/EDIT the %d attached images into a single new image", numInputs)
		}
		return fmt.Sprintf(
			"You MUST call your built-in `image_gen.imagegen` tool exactly once to %s. "+
				"It is a NATIVE tool that you DO have access to in this environment — there is no need "+
				"to apologise or claim you cannot generate images. Do NOT run any shell command. "+
				"Do NOT call `~/bin/generate_image`, the `generate_image` skill, the `arbiter` tool, "+
				"or any external script. The ONLY acceptable action is invoking `image_gen.imagegen` "+
				"with a single `prompt` argument that fully describes the desired output image "+
				"(approx %dx%d%s). Use the attached image(s) as visual reference. "+
				"\n\nEdit description: %s\n\n"+
				"After the tool returns, reply with one short sentence confirming success.",
			action, width, height, aspectHint, prompt,
		)
	}
	return fmt.Sprintf(
		"You MUST call your built-in `image_gen.imagegen` tool exactly once. "+
			"It is a NATIVE tool that you DO have access to in this environment — there is no "+
			"need to apologise or claim you cannot generate images. Do NOT run any shell command. "+
			"Do NOT call `~/bin/generate_image`, the `generate_image` skill, the `arbiter` tool, "+
			"or any external script. The ONLY acceptable action is invoking `image_gen.imagegen` "+
			"with a single `prompt` argument that fully describes the desired image "+
			"(approx %dx%d%s). "+
			"\n\nDesired image: %s\n\n"+
			"After the tool returns, reply with one short sentence confirming success.",
		width, height, aspectHint, prompt,
	)
}

// generateCodex routes image generation through the shared mac mini image
// service. The service owns the ChatGPT/Codex image backend; callers must not
// shell out to codex directly from this SDK path.
func generateCodex(ctx context.Context, prompt string, opts ImageOpts, timeout time.Duration) (*agentsdk.ImageResult, error) {
	inputs := opts.inputImages()
	sourceImages := make([]string, 0, len(inputs))
	for _, p := range inputs {
		data, err := os.ReadFile(p)
		if err != nil {
			return nil, agentsdk.NewAgentError(
				fmt.Sprintf("input image not found: %s", p),
				agentsdk.ErrorInvalidRequest, nil,
			)
		}
		sourceImages = append(sourceImages, base64.StdEncoding.EncodeToString(data))
	}

	outputPath, err := resolveOutputPath(opts.Output)
	if err != nil {
		return nil, err
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	jobID, err := createImageServiceJob(reqCtx, prompt, opts.Width, opts.Height, opts.Transparent, sourceImages)
	if err != nil {
		return nil, err
	}
	if err := waitImageServiceJob(reqCtx, jobID); err != nil {
		return nil, err
	}
	data, err := fetchImageServiceImage(reqCtx, jobID)
	if err != nil {
		return nil, err
	}
	if err := writeServiceImage(data, outputPath, opts.Transparent); err != nil {
		return nil, err
	}

	return &agentsdk.ImageResult{
		Path:      outputPath,
		ModelUsed: codexModelInfo,
		Prompt:    prompt,
		Width:     opts.Width,
		Height:    opts.Height,
	}, nil
}

func createImageServiceJob(ctx context.Context, prompt string, width, height int, transparent bool, sourceImages []string) (string, error) {
	body := map[string]any{
		"prompt":      prompt,
		"width":       width,
		"height":      height,
		"transparent": transparent,
	}
	if len(sourceImages) > 0 {
		body["source_images"] = sourceImages
	}
	var response struct {
		ID string `json:"id"`
	}
	if err := imageServiceJSON(ctx, http.MethodPost, "/jobs", body, &response); err != nil {
		return "", err
	}
	if response.ID == "" {
		return "", agentsdk.NewAgentError("image service returned no job id", agentsdk.ErrorInternal, nil)
	}
	return response.ID, nil
}

func waitImageServiceJob(ctx context.Context, jobID string) error {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		var status struct {
			Status   string `json:"status"`
			Attempts int    `json:"attempts"`
			Error    string `json:"error"`
		}
		if err := imageServiceJSON(ctx, http.MethodGet, "/jobs/"+jobID, nil, &status); err != nil {
			return err
		}
		switch status.Status {
		case "done":
			return nil
		case "failed":
			return agentsdk.NewAgentError(
				fmt.Sprintf("image service job %s failed after %d attempts: %s", jobID, status.Attempts, status.Error),
				agentsdk.ErrorInternal, nil,
			)
		case "queued", "running":
		default:
			return agentsdk.NewAgentError(
				fmt.Sprintf("image service job %s returned unknown status %q", jobID, status.Status),
				agentsdk.ErrorInternal, nil,
			)
		}
		select {
		case <-ctx.Done():
			return agentsdk.NewAgentError(
				fmt.Sprintf("image service job %s timed out: %v", jobID, ctx.Err()),
				agentsdk.ErrorTimeout, nil,
			)
		case <-ticker.C:
		}
	}
}

func imageServiceJSON(ctx context.Context, method, path string, payload any, target any) error {
	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		body = bytes.NewReader(data)
	}
	request, err := http.NewRequestWithContext(ctx, method, imageServiceURL+path, body)
	if err != nil {
		return err
	}
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return agentsdk.NewAgentError(fmt.Sprintf("image service request failed: %v", err), agentsdk.ErrorInternal, nil)
	}
	defer func() { _ = response.Body.Close() }()
	data, _ := io.ReadAll(response.Body)
	if response.StatusCode >= 400 {
		return agentsdk.NewAgentError(
			fmt.Sprintf("image service %s %s returned HTTP %d: %s", method, path, response.StatusCode, string(data)),
			agentsdk.ErrorInternal, nil,
		)
	}
	if err := json.Unmarshal(data, target); err != nil {
		return agentsdk.NewAgentError(
			fmt.Sprintf("image service %s %s returned invalid JSON: %v", method, path, err),
			agentsdk.ErrorInternal, nil,
		)
	}
	return nil
}

func fetchImageServiceImage(ctx context.Context, jobID string) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, imageServiceURL+"/jobs/"+jobID+"/image", nil)
	if err != nil {
		return nil, err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, agentsdk.NewAgentError(fmt.Sprintf("image service image fetch failed: %v", err), agentsdk.ErrorInternal, nil)
	}
	defer func() { _ = response.Body.Close() }()
	data, _ := io.ReadAll(response.Body)
	if response.StatusCode >= 400 {
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("image service GET /jobs/%s/image returned HTTP %d: %s", jobID, response.StatusCode, string(data)),
			agentsdk.ErrorInternal, nil,
		)
	}
	if !bytes.HasPrefix(data, []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}) {
		return nil, agentsdk.NewAgentError("image service returned non-PNG data", agentsdk.ErrorInternal, nil)
	}
	return data, nil
}

func writeServiceImage(data []byte, outputPath string, transparent bool) error {
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	ext := strings.ToLower(filepath.Ext(outputPath))
	if ext == ".jpg" || ext == ".jpeg" {
		if transparent {
			return agentsdk.NewAgentError("transparent image output must be PNG, not JPEG", agentsdk.ErrorInvalidRequest, nil)
		}
		img, _, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			return fmt.Errorf("decoding service PNG: %w", err)
		}
		out := image.NewRGBA(img.Bounds())
		draw.Draw(out, out.Bounds(), &image.Uniform{C: color.White}, image.Point{}, draw.Src)
		draw.Draw(out, out.Bounds(), img, img.Bounds().Min, draw.Over)
		file, err := os.Create(outputPath)
		if err != nil {
			return err
		}
		defer func() { _ = file.Close() }()
		return jpeg.Encode(file, out, &jpeg.Options{Quality: 92})
	}
	return os.WriteFile(outputPath, data, 0o644)
}

// findCodexImages returns generated codex images sorted newest-first, RESTRICTED
// to those modified at/after `since` (so only images produced by the current
// call qualify — never a stale one from a prior generation). When threadID is
// set, it scans that session's folder; otherwise it falls back to every session
// folder, but the `since` gate still guarantees freshness.
func findCodexImages(threadID string, since time.Time) []string {
	root := codexImageDir()
	dirs := []string{}
	if threadID != "" {
		dirs = append(dirs, filepath.Join(root, threadID))
	} else {
		entries, err := os.ReadDir(root)
		if err != nil {
			return nil
		}
		for _, e := range entries {
			if e.IsDir() {
				dirs = append(dirs, filepath.Join(root, e.Name()))
			}
		}
	}
	type fileInfo struct {
		path    string
		modTime time.Time
	}
	var files []fileInfo
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for _, e := range entries {
			name := e.Name()
			if !strings.HasPrefix(name, "ig_") || !strings.HasSuffix(name, ".png") {
				continue
			}
			info, err := e.Info()
			if err != nil {
				continue
			}
			// Freshness gate: drop any image written before this call started.
			if info.ModTime().Before(since) {
				continue
			}
			files = append(files, fileInfo{path: filepath.Join(d, name), modTime: info.ModTime()})
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].modTime.After(files[j].modTime) })
	out := make([]string, 0, len(files))
	for _, f := range files {
		out = append(out, f.path)
	}
	return out
}

func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

// codexModel returns the configured codex model, defaulting to gpt-5.5.
func codexModel(cfg *agentsdk.Config) string {
	if cfg != nil && cfg.Image.CodexModel != "" {
		return cfg.Image.CodexModel
	}
	return "gpt-5.5"
}
