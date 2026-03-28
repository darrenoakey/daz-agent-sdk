package capability

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// ── Helper ─────────────────────────────────────────────────────────────────

// createTestPNG returns a minimal valid 1x1 red PNG file.
func createTestPNG() []byte {
	// Minimal 1x1 red pixel PNG (hand-crafted)
	return []byte{
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
		0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
		0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
		0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
		0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
		0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
		0x44, 0xAE, 0x42, 0x60, 0x82,
	}
}

// ollamaCfg builds a minimal Config pointing Ollama at the given server URL.
func ollamaCfg(serverURL string) *agentsdk.Config {
	cfg := &agentsdk.Config{}
	cfg.Image.Model = "z-image-turbo"
	cfg.Image.Tiers = map[string]agentsdk.ImageTierConfig{
		"high": {Steps: 3},
	}
	cfg.Providers = map[string]map[string]any{
		"ollama": {"base_url": serverURL},
	}
	return cfg
}

// sparkCfg builds a minimal Config pointing Spark at the given server URL.
func sparkCfg(serverURL string) *agentsdk.Config {
	cfg := &agentsdk.Config{}
	cfg.Image.Model = "z-image-turbo"
	cfg.Image.Tiers = map[string]agentsdk.ImageTierConfig{
		"high": {Steps: 3},
	}
	cfg.Providers = map[string]map[string]any{
		"spark": {"base_url": serverURL},
	}
	return cfg
}

// ── Unit: closestAspectRatio ───────────────────────────────────────────────

func TestClosestAspectRatio_Square(t *testing.T) {
	if got := closestAspectRatio(512, 512); got != "1:1" {
		t.Errorf("expected 1:1, got %s", got)
	}
}

func TestClosestAspectRatio_Landscape(t *testing.T) {
	// 1920×1080 → 16:9
	if got := closestAspectRatio(1920, 1080); got != "16:9" {
		t.Errorf("expected 16:9, got %s", got)
	}
}

func TestClosestAspectRatio_Portrait(t *testing.T) {
	// 1080×1920 → 9:16
	if got := closestAspectRatio(1080, 1920); got != "9:16" {
		t.Errorf("expected 9:16, got %s", got)
	}
}

func TestClosestAspectRatio_FourThree(t *testing.T) {
	// 800×600 → 4:3
	if got := closestAspectRatio(800, 600); got != "4:3" {
		t.Errorf("expected 4:3, got %s", got)
	}
}

func TestClosestAspectRatio_ThreeFour(t *testing.T) {
	// 600×800 → 3:4
	if got := closestAspectRatio(600, 800); got != "3:4" {
		t.Errorf("expected 3:4, got %s", got)
	}
}

func TestClosestAspectRatio_ExtremeWide(t *testing.T) {
	// 3840×1080 → 16:9 (closest standard wide ratio)
	got := closestAspectRatio(3840, 1080)
	if got != "16:9" {
		t.Errorf("expected 16:9 for 3840×1080, got %s", got)
	}
}

func TestClosestAspectRatio_ExtremeTall(t *testing.T) {
	// 1080×3840 → 9:16
	got := closestAspectRatio(1080, 3840)
	if got != "9:16" {
		t.Errorf("expected 9:16 for 1080×3840, got %s", got)
	}
}

func TestClosestAspectRatio_ZeroDimension(t *testing.T) {
	// Zero dimensions → safe default of 1:1
	if got := closestAspectRatio(0, 0); got != "1:1" {
		t.Errorf("expected 1:1 for zero dims, got %s", got)
	}
}

// ── Unit: imageSize ────────────────────────────────────────────────────────

func TestImageSize_Tiny(t *testing.T) {
	if got := imageSize(256, 256); got != "0.5K" {
		t.Errorf("expected 0.5K, got %s", got)
	}
}

func TestImageSize_Medium(t *testing.T) {
	if got := imageSize(512, 768); got != "1K" {
		t.Errorf("expected 1K, got %s", got)
	}
}

func TestImageSize_Large(t *testing.T) {
	if got := imageSize(1920, 1080); got != "2K" {
		t.Errorf("expected 2K, got %s", got)
	}
}

func TestImageSize_Huge(t *testing.T) {
	if got := imageSize(4096, 2048); got != "4K" {
		t.Errorf("expected 4K, got %s", got)
	}
}

func TestImageSize_Exact512(t *testing.T) {
	if got := imageSize(512, 512); got != "0.5K" {
		t.Errorf("expected 0.5K for 512×512, got %s", got)
	}
}

func TestImageSize_Exact1024(t *testing.T) {
	if got := imageSize(1024, 512); got != "1K" {
		t.Errorf("expected 1K for 1024×512, got %s", got)
	}
}

// ── Unit: sparkBaseURL ─────────────────────────────────────────────────────

func TestSparkBaseURL_Default(t *testing.T) {
	t.Setenv("SPARK_IMAGE_URL", "")
	if got := sparkBaseURL(nil); got != "http://spark:8100" {
		t.Errorf("expected default, got %s", got)
	}
}

func TestSparkBaseURL_FromEnv(t *testing.T) {
	t.Setenv("SPARK_IMAGE_URL", "http://myhost:9000")
	if got := sparkBaseURL(nil); got != "http://myhost:9000" {
		t.Errorf("expected env value, got %s", got)
	}
}

func TestSparkBaseURL_FromConfig(t *testing.T) {
	t.Setenv("SPARK_IMAGE_URL", "")
	cfg := &agentsdk.Config{
		Providers: map[string]map[string]any{
			"spark": {"base_url": "http://cfg-spark:1234"},
		},
	}
	if got := sparkBaseURL(cfg); got != "http://cfg-spark:1234" {
		t.Errorf("expected config value, got %s", got)
	}
}

func TestSparkBaseURL_ConfigOverridesEnv(t *testing.T) {
	t.Setenv("SPARK_IMAGE_URL", "http://env-spark:9999")
	cfg := &agentsdk.Config{
		Providers: map[string]map[string]any{
			"spark": {"base_url": "http://cfg-spark:1234"},
		},
	}
	if got := sparkBaseURL(cfg); got != "http://cfg-spark:1234" {
		t.Errorf("expected config to override env, got %s", got)
	}
}

// ── Unit: arbiterBaseURL ───────────────────────────────────────────────────

func TestArbiterBaseURL_Default(t *testing.T) {
	t.Setenv("ARBITER_URL", "")
	if got := arbiterBaseURL(nil); got != "http://spark:8400" {
		t.Errorf("expected default, got %s", got)
	}
}

func TestArbiterBaseURL_FromEnv(t *testing.T) {
	t.Setenv("ARBITER_URL", "http://arbiter:8888")
	if got := arbiterBaseURL(nil); got != "http://arbiter:8888" {
		t.Errorf("expected env value, got %s", got)
	}
}

func TestArbiterBaseURL_FromConfig(t *testing.T) {
	t.Setenv("ARBITER_URL", "")
	cfg := &agentsdk.Config{
		Providers: map[string]map[string]any{
			"arbiter": {"base_url": "http://cfg-arbiter:5678"},
		},
	}
	if got := arbiterBaseURL(cfg); got != "http://cfg-arbiter:5678" {
		t.Errorf("expected config value, got %s", got)
	}
}

// ── Unit: buildFallbackChain ───────────────────────────────────────────────

func TestBuildFallbackChain_NilFallback(t *testing.T) {
	cfg := &agentsdk.Config{}
	chain := buildFallbackChain("spark", cfg)
	if len(chain) != 1 || chain[0] != "spark" {
		t.Errorf("expected [spark], got %v", chain)
	}
}

func TestBuildFallbackChain_WithFallbacks(t *testing.T) {
	cfg := &agentsdk.Config{
		Image: agentsdk.ImageConfig{
			Fallback: []string{"ollama", "nano-banana-2"},
		},
	}
	chain := buildFallbackChain("spark", cfg)
	if len(chain) != 3 {
		t.Fatalf("expected 3 entries, got %v", chain)
	}
	if chain[0] != "spark" || chain[1] != "ollama" || chain[2] != "nano-banana-2" {
		t.Errorf("unexpected chain order: %v", chain)
	}
}

func TestBuildFallbackChain_PrimaryExcludedFromFallbacks(t *testing.T) {
	cfg := &agentsdk.Config{
		Image: agentsdk.ImageConfig{
			Fallback: []string{"spark", "ollama"},
		},
	}
	chain := buildFallbackChain("spark", cfg)
	// spark must not appear twice
	count := 0
	for _, p := range chain {
		if p == "spark" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("spark should appear once, got %d times in %v", count, chain)
	}
}

func TestBuildFallbackChain_AlternatePrimary(t *testing.T) {
	cfg := &agentsdk.Config{
		Image: agentsdk.ImageConfig{
			Fallback: []string{"spark"},
		},
	}
	chain := buildFallbackChain("ollama", cfg)
	if len(chain) != 2 || chain[0] != "ollama" || chain[1] != "spark" {
		t.Errorf("expected [ollama spark], got %v", chain)
	}
}

// ── Unit: multipart form building ─────────────────────────────────────────

func TestSparkI2I_MultipartForm(t *testing.T) {
	pngData := createTestPNG()
	// Write input image to temp file
	inputFile, err := os.CreateTemp("", "input_*.png")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(inputFile.Name())
	if _, err := inputFile.Write(pngData); err != nil {
		t.Fatal(err)
	}
	inputFile.Close()

	var capturedContentType string
	var capturedFields = map[string]string{}
	var capturedImageData []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/edit" {
			t.Errorf("expected /v1/images/edit, got %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		capturedContentType = r.Header.Get("Content-Type")
		mediaType, params, err := mime.ParseMediaType(capturedContentType)
		if err != nil || !strings.HasPrefix(mediaType, "multipart/") {
			t.Errorf("expected multipart content-type, got %s", capturedContentType)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		mr := multipart.NewReader(r.Body, params["boundary"])
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Errorf("reading multipart: %v", err)
				break
			}
			data, _ := io.ReadAll(part)
			name := part.FormName()
			if name == "image" {
				capturedImageData = data
			} else {
				capturedFields[name] = string(data)
			}
		}

		b64Image := base64.StdEncoding.EncodeToString(createTestPNG())
		resp := sparkGenerateResponse{Image: b64Image}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "out.png")

	cfg := sparkCfg(server.URL)
	_, err = GenerateImage(context.Background(), "edit this image", ImageOpts{
		Provider: "spark",
		Image:    inputFile.Name(),
		Width:    512,
		Height:   512,
		Output:   outputPath,
		Config:   cfg,
		Timeout:  10 * time.Second,
	})
	if err != nil {
		t.Fatalf("GenerateImage failed: %v", err)
	}

	// Verify multipart boundary is present
	if !strings.Contains(capturedContentType, "boundary=") {
		t.Errorf("content-type missing boundary: %s", capturedContentType)
	}

	// Verify key fields
	if capturedFields["prompt"] != "edit this image" {
		t.Errorf("expected prompt field, got %q", capturedFields["prompt"])
	}
	if capturedFields["model"] != sparkImageModel {
		t.Errorf("expected model=%s, got %q", sparkImageModel, capturedFields["model"])
	}
	if len(capturedImageData) != len(pngData) {
		t.Errorf("expected %d image bytes, got %d", len(pngData), len(capturedImageData))
	}
}

// ── Spark text-to-image via httptest ──────────────────────────────────────

func TestGenerateImage_SparkT2I(t *testing.T) {
	pngData := createTestPNG()
	b64Image := base64.StdEncoding.EncodeToString(pngData)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generate" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req sparkGenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("failed to decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if req.Prompt != "a red circle" {
			t.Errorf("unexpected prompt: %s", req.Prompt)
		}
		if req.Model != sparkImageModel {
			t.Errorf("unexpected model: %s", req.Model)
		}

		resp := sparkGenerateResponse{Image: b64Image}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "spark_out.png")
	cfg := sparkCfg(server.URL)

	result, err := GenerateImage(context.Background(), "a red circle", ImageOpts{
		Provider: "spark",
		Width:    512,
		Height:   512,
		Output:   outputPath,
		Tier:     agentsdk.TierHigh,
		Config:   cfg,
		Timeout:  10 * time.Second,
	})
	if err != nil {
		t.Fatalf("GenerateImage failed: %v", err)
	}

	if result.Path != outputPath {
		t.Errorf("expected path %s, got %s", outputPath, result.Path)
	}
	if result.ModelUsed.Provider != "spark" {
		t.Errorf("expected provider spark, got %s", result.ModelUsed.Provider)
	}
	if result.Prompt != "a red circle" {
		t.Errorf("expected prompt, got %q", result.Prompt)
	}

	written, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if len(written) != len(pngData) {
		t.Errorf("expected %d bytes, got %d", len(pngData), len(written))
	}
}

func TestGenerateImage_SparkNoImageData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := sparkGenerateResponse{Image: ""}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := sparkCfg(server.URL)
	_, err := GenerateImage(context.Background(), "test", ImageOpts{
		Provider: "spark",
		Config:   cfg,
		Timeout:  5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error for missing image data")
	}
	agentErr, ok := err.(*agentsdk.AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T", err)
	}
	if agentErr.Kind != agentsdk.ErrorInternal {
		t.Errorf("expected ErrorInternal, got %s", agentErr.Kind)
	}
}

func TestGenerateImage_SparkHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer server.Close()

	cfg := sparkCfg(server.URL)
	_, err := GenerateImage(context.Background(), "test", ImageOpts{
		Provider: "spark",
		Config:   cfg,
		Timeout:  5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error for HTTP 500")
	}
	agentErr, ok := err.(*agentsdk.AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T", err)
	}
	if agentErr.Kind != agentsdk.ErrorInternal {
		t.Errorf("expected ErrorInternal, got %s", agentErr.Kind)
	}
}

func TestGenerateImage_SparkNotRunning(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := listener.Addr().String()
	listener.Close()

	cfg := &agentsdk.Config{
		Providers: map[string]map[string]any{
			"spark": {"base_url": "http://" + addr},
		},
		Image: agentsdk.ImageConfig{
			Tiers: map[string]agentsdk.ImageTierConfig{"high": {Steps: 3}},
		},
	}

	_, err = GenerateImage(context.Background(), "test", ImageOpts{
		Provider: "spark",
		Config:   cfg,
		Timeout:  2 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error when spark is not running")
	}
}

// ── Ollama via httptest ────────────────────────────────────────────────────

func TestGenerateImage_MockOllama(t *testing.T) {
	pngData := createTestPNG()
	b64Image := base64.StdEncoding.EncodeToString(pngData)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ollamaGenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("failed to decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		if req.Prompt != "a red circle" {
			t.Errorf("unexpected prompt: %s", req.Prompt)
		}
		if req.Stream {
			t.Error("expected stream=false")
		}

		resp := ollamaGenerateResponse{
			Model: req.Model,
			Image: b64Image,
			Done:  true,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "test_output.png")
	cfg := ollamaCfg(server.URL)

	result, err := GenerateImage(context.Background(), "a red circle", ImageOpts{
		Provider: "ollama",
		Width:    512,
		Height:   512,
		Output:   outputPath,
		Tier:     agentsdk.TierHigh,
		Config:   cfg,
		Timeout:  10 * time.Second,
	})
	if err != nil {
		t.Fatalf("GenerateImage failed: %v", err)
	}

	if result.Path != outputPath {
		t.Errorf("expected path %s, got %s", outputPath, result.Path)
	}
	if result.Prompt != "a red circle" {
		t.Errorf("expected prompt 'a red circle', got %q", result.Prompt)
	}
	if result.ModelUsed.Provider != "ollama" {
		t.Errorf("expected provider 'ollama', got %q", result.ModelUsed.Provider)
	}

	written, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if len(written) != len(pngData) {
		t.Errorf("expected %d bytes, got %d", len(pngData), len(written))
	}
}

func TestGenerateImage_NoImageData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ollamaGenerateResponse{
			Model: "x/z-image-turbo",
			Image: "",
			Done:  true,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := ollamaCfg(server.URL)
	_, err := GenerateImage(context.Background(), "test", ImageOpts{
		Provider: "ollama",
		Config:   cfg,
		Timeout:  5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error for missing image data")
	}
	agentErr, ok := err.(*agentsdk.AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T", err)
	}
	if agentErr.Kind != agentsdk.ErrorInternal {
		t.Errorf("expected ErrorInternal, got %s", agentErr.Kind)
	}
}

func TestGenerateImage_TempFile(t *testing.T) {
	pngData := createTestPNG()
	b64Image := base64.StdEncoding.EncodeToString(pngData)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := sparkGenerateResponse{Image: b64Image}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := sparkCfg(server.URL)

	result, err := GenerateImage(context.Background(), "test prompt", ImageOpts{
		Provider: "spark",
		Config:   cfg,
		Timeout:  5 * time.Second,
	})
	if err != nil {
		t.Fatalf("GenerateImage failed: %v", err)
	}
	defer os.Remove(result.Path)

	if result.Path == "" {
		t.Error("expected non-empty path")
	}
	info, err := os.Stat(result.Path)
	if err != nil {
		t.Fatalf("temp file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("temp file is empty")
	}
}

func TestGenerateImage_OllamaNotRunning(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := listener.Addr().String()
	listener.Close()

	cfg := ollamaCfg("http://" + addr)

	_, err = GenerateImage(context.Background(), "test", ImageOpts{
		Provider: "ollama",
		Config:   cfg,
		Timeout:  2 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error when Ollama is not running")
	}
}

// ── Fallback chain ─────────────────────────────────────────────────────────

func TestGenerateImage_FallbackChain_UsesSecondOnFirstFailure(t *testing.T) {
	pngData := createTestPNG()
	b64Image := base64.StdEncoding.EncodeToString(pngData)

	// Spark server: always 500
	sparkServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "spark down", http.StatusInternalServerError)
	}))
	defer sparkServer.Close()

	// Ollama fallback server: returns valid image
	ollamaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ollamaGenerateResponse{Model: "x/z-image-turbo", Image: b64Image, Done: true}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer ollamaServer.Close()

	cfg := &agentsdk.Config{
		Image: agentsdk.ImageConfig{
			Model:    "z-image-turbo",
			Tiers:    map[string]agentsdk.ImageTierConfig{"high": {Steps: 3}},
			Fallback: []string{"ollama"},
		},
		Providers: map[string]map[string]any{
			"spark":  {"base_url": sparkServer.URL},
			"ollama": {"base_url": ollamaServer.URL},
		},
	}

	result, err := GenerateImage(context.Background(), "test fallback", ImageOpts{
		Provider: "spark",
		Config:   cfg,
		Timeout:  5 * time.Second,
	})
	if err != nil {
		t.Fatalf("expected fallback to succeed, got: %v", err)
	}
	if result.ModelUsed.Provider != "ollama" {
		t.Errorf("expected ollama as fallback provider, got %s", result.ModelUsed.Provider)
	}
}

func TestGenerateImage_FallbackChain_AllFail(t *testing.T) {
	sparkServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "spark down", http.StatusInternalServerError)
	}))
	defer sparkServer.Close()

	ollamaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "ollama down", http.StatusInternalServerError)
	}))
	defer ollamaServer.Close()

	cfg := &agentsdk.Config{
		Image: agentsdk.ImageConfig{
			Model:    "z-image-turbo",
			Tiers:    map[string]agentsdk.ImageTierConfig{"high": {Steps: 3}},
			Fallback: []string{"ollama"},
		},
		Providers: map[string]map[string]any{
			"spark":  {"base_url": sparkServer.URL},
			"ollama": {"base_url": ollamaServer.URL},
		},
	}

	_, err := GenerateImage(context.Background(), "test all fail", ImageOpts{
		Provider: "spark",
		Config:   cfg,
		Timeout:  5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error when all providers fail")
	}
}

func TestGenerateImage_UnknownProvider(t *testing.T) {
	cfg := &agentsdk.Config{
		Image: agentsdk.ImageConfig{
			Tiers: map[string]agentsdk.ImageTierConfig{"high": {Steps: 3}},
		},
		Providers: map[string]map[string]any{},
	}

	_, err := GenerateImage(context.Background(), "test", ImageOpts{
		Provider: "nonexistent",
		Config:   cfg,
		Timeout:  5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
	agentErr, ok := err.(*agentsdk.AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T", err)
	}
	if agentErr.Kind != agentsdk.ErrorInvalidRequest {
		t.Errorf("expected ErrorInvalidRequest, got %s", agentErr.Kind)
	}
}

// ── Default provider ───────────────────────────────────────────────────────

func TestGenerateImage_DefaultProviderIsSpark(t *testing.T) {
	pngData := createTestPNG()
	b64Image := base64.StdEncoding.EncodeToString(pngData)

	sparkServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/images/generate" {
			resp := sparkGenerateResponse{Image: b64Image}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer sparkServer.Close()

	cfg := sparkCfg(sparkServer.URL)

	// Provider is empty — should default to spark
	result, err := GenerateImage(context.Background(), "default provider test", ImageOpts{
		Config:  cfg,
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("GenerateImage failed: %v", err)
	}
	if result.ModelUsed.Provider != "spark" {
		t.Errorf("expected spark provider by default, got %s", result.ModelUsed.Provider)
	}
}

// ── Arbiter background removal ─────────────────────────────────────────────

func TestRemoveBackgroundSpark_Success(t *testing.T) {
	pngData := createTestPNG()
	resultB64 := base64.StdEncoding.EncodeToString(pngData)

	arbiter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/jobs":
			var req arbiterJobRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "bad request", http.StatusBadRequest)
				return
			}
			if req.Type != "background-remove" {
				http.Error(w, fmt.Sprintf("unexpected job type: %s", req.Type), http.StatusBadRequest)
				return
			}
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(arbiterJobResponse{JobID: "job-123"})

		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/job-123"):
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(arbiterJobStatus{
				Status: "completed",
				Result: map[string]any{"image": resultB64},
			})

		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))
	defer arbiter.Close()

	// Write a temp image file
	tmpImg, err := os.CreateTemp("", "test_img_*.png")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpImg.Name())
	if _, err := tmpImg.Write(pngData); err != nil {
		t.Fatal(err)
	}
	tmpImg.Close()

	cfg := &agentsdk.Config{
		Providers: map[string]map[string]any{
			"arbiter": {"base_url": arbiter.URL},
		},
	}

	resultPath, err := removeBackgroundSpark(context.Background(), tmpImg.Name(), cfg, 10*time.Second)
	if err != nil {
		t.Fatalf("removeBackgroundSpark failed: %v", err)
	}
	if resultPath != tmpImg.Name() {
		t.Errorf("expected same path, got %s", resultPath)
	}
	written, err := os.ReadFile(resultPath)
	if err != nil {
		t.Fatalf("reading result: %v", err)
	}
	if len(written) != len(pngData) {
		t.Errorf("expected %d bytes, got %d", len(pngData), len(written))
	}
}

func TestRemoveBackgroundSpark_JobFailed(t *testing.T) {
	arbiter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/jobs":
			json.NewEncoder(w).Encode(arbiterJobResponse{JobID: "job-fail"})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/job-fail"):
			json.NewEncoder(w).Encode(arbiterJobStatus{
				Status: "failed",
				Error:  "model crashed",
			})
		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))
	defer arbiter.Close()

	tmpImg, err := os.CreateTemp("", "test_img_*.png")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpImg.Name())
	tmpImg.Write(createTestPNG())
	tmpImg.Close()

	cfg := &agentsdk.Config{
		Providers: map[string]map[string]any{
			"arbiter": {"base_url": arbiter.URL},
		},
	}

	_, err = removeBackgroundSpark(context.Background(), tmpImg.Name(), cfg, 10*time.Second)
	if err == nil {
		t.Fatal("expected error when arbiter job fails")
	}
}

// ── Integration: real Ollama ───────────────────────────────────────────────

func TestGenerateImage_RealOllama(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real Ollama test in short mode")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://localhost:11434/", nil)
	if err != nil {
		t.Skip("cannot create request to check Ollama")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Skip("Ollama is not running at localhost:11434, skipping real test")
	}
	resp.Body.Close()

	checkCtx, checkCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer checkCancel()
	checkReq, err := http.NewRequestWithContext(checkCtx, http.MethodGet, "http://localhost:11434/api/tags", nil)
	if err != nil {
		t.Skip("cannot check available models")
	}
	tagsResp, err := http.DefaultClient.Do(checkReq)
	if err != nil {
		t.Skip("cannot list Ollama models")
	}
	defer tagsResp.Body.Close()
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(tagsResp.Body).Decode(&tags); err != nil {
		t.Skip("cannot decode Ollama models list")
	}
	found := false
	for _, m := range tags.Models {
		if m.Name == "x/z-image-turbo:latest" || m.Name == "x/z-image-turbo" {
			found = true
			break
		}
	}
	if !found {
		t.Skip("x/z-image-turbo model not available in Ollama, skipping real test")
	}

	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "real_test.png")

	result, err := GenerateImage(context.Background(), "a simple red circle on white background", ImageOpts{
		Provider: "ollama",
		Width:    256,
		Height:   256,
		Output:   outputPath,
		Tier:     agentsdk.TierHigh,
		Timeout:  120 * time.Second,
	})
	if err != nil {
		t.Fatalf("GenerateImage with real Ollama failed: %v", err)
	}

	info, err := os.Stat(result.Path)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}
	t.Logf("Generated image: %s (%d bytes)", result.Path, info.Size())
}

// ── Integration: real Spark ────────────────────────────────────────────────

func TestGenerateImage_RealSpark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real Spark test in short mode")
	}

	sparkURL := os.Getenv("SPARK_IMAGE_URL")
	if sparkURL == "" {
		sparkURL = "http://spark:8100"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sparkURL+"/health", nil)
	if err != nil {
		t.Skipf("cannot reach spark at %s", sparkURL)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Skipf("spark not available at %s: %v", sparkURL, err)
	}
	resp.Body.Close()

	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "spark_real.png")

	result, err := GenerateImage(context.Background(), "a simple red square on white background", ImageOpts{
		Provider: "spark",
		Width:    256,
		Height:   256,
		Output:   outputPath,
		Tier:     agentsdk.TierHigh,
		Timeout:  120 * time.Second,
	})
	if err != nil {
		t.Fatalf("Spark real test failed: %v", err)
	}

	info, err := os.Stat(result.Path)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}
	t.Logf("Generated spark image: %s (%d bytes)", result.Path, info.Size())
}

// ── Integration: real Nano Banana 2 ───────────────────────────────────────

func TestGenerateImage_RealNanoBanana(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real Nano Banana 2 test in short mode")
	}

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY or GOOGLE_API_KEY not set, skipping nano-banana-2 test")
	}

	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "nb2_real.png")

	result, err := GenerateImage(context.Background(), "a simple red circle on white background", ImageOpts{
		Provider: "nano-banana-2",
		Width:    512,
		Height:   512,
		Output:   outputPath,
		Tier:     agentsdk.TierHigh,
		Timeout:  120 * time.Second,
	})
	if err != nil {
		t.Fatalf("Nano Banana 2 real test failed: %v", err)
	}

	info, err := os.Stat(result.Path)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}
	t.Logf("Generated nano-banana-2 image: %s (%d bytes)", result.Path, info.Size())
}

// ── Model selection unit tests ─────────────────────────────────────────────

func TestSparkModelName_ExplicitOverride(t *testing.T) {
	got := sparkModelName(ImageOpts{Model: "flux-schnell"}, nil)
	if got != "flux-schnell" {
		t.Errorf("sparkModelName() = %q, want flux-schnell", got)
	}
}

func TestSparkModelName_ConfigOverride(t *testing.T) {
	cfg := &agentsdk.Config{Image: agentsdk.ImageConfig{Model: "flux-schnell"}}
	got := sparkModelName(ImageOpts{}, cfg)
	if got != "flux-schnell" {
		t.Errorf("sparkModelName() = %q, want flux-schnell", got)
	}
}

func TestSparkModelName_ExplicitBeatsConfig(t *testing.T) {
	cfg := &agentsdk.Config{Image: agentsdk.ImageConfig{Model: "flux-schnell"}}
	got := sparkModelName(ImageOpts{Model: "z-image-turbo"}, cfg)
	if got != "z-image-turbo" {
		t.Errorf("sparkModelName() = %q, want z-image-turbo (explicit beats config)", got)
	}
}

func TestSparkModelName_Default(t *testing.T) {
	got := sparkModelName(ImageOpts{}, nil)
	if got != "z-image-turbo" {
		t.Errorf("sparkModelName() = %q, want z-image-turbo", got)
	}
}

func TestSparkModelInfoFor_ZImageTurbo(t *testing.T) {
	info := sparkModelInfoFor("z-image-turbo")
	if info.ModelID != "z-image-turbo" {
		t.Errorf("ModelID = %q, want z-image-turbo", info.ModelID)
	}
	if info.Provider != "spark" {
		t.Errorf("Provider = %q, want spark", info.Provider)
	}
}

func TestSparkModelInfoFor_FluxSchnell(t *testing.T) {
	info := sparkModelInfoFor("flux-schnell")
	if info.ModelID != "flux-schnell" {
		t.Errorf("ModelID = %q, want flux-schnell", info.ModelID)
	}
	if info.DisplayName != "Spark FLUX.1-schnell" {
		t.Errorf("DisplayName = %q, want Spark FLUX.1-schnell", info.DisplayName)
	}
}

func TestSparkT2I_SendsModelInRequest(t *testing.T) {
	var receivedModel string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req sparkGenerateRequest
		json.Unmarshal(body, &req)
		receivedModel = req.Model
		json.NewEncoder(w).Encode(sparkGenerateResponse{
			Image: base64.StdEncoding.EncodeToString(createTestPNG()),
			Model: req.Model,
		})
	}))
	defer ts.Close()

	cfg := &agentsdk.Config{
		Providers: map[string]map[string]any{
			"spark": {"base_url": ts.URL},
		},
	}

	result, err := GenerateImage(context.Background(), "test", ImageOpts{
		Provider: "spark",
		Model:    "flux-schnell",
		Width:    512,
		Height:   512,
		Output:   filepath.Join(t.TempDir(), "out.png"),
		Config:   cfg,
	})
	if err != nil {
		t.Fatalf("GenerateImage error: %v", err)
	}

	if receivedModel != "flux-schnell" {
		t.Errorf("spark server received model=%q, want flux-schnell", receivedModel)
	}
	if result.ModelUsed.ModelID != "flux-schnell" {
		t.Errorf("result.ModelUsed.ModelID=%q, want flux-schnell", result.ModelUsed.ModelID)
	}
}

func TestSparkT2I_DefaultModel(t *testing.T) {
	var receivedModel string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req sparkGenerateRequest
		json.Unmarshal(body, &req)
		receivedModel = req.Model
		json.NewEncoder(w).Encode(sparkGenerateResponse{
			Image: base64.StdEncoding.EncodeToString(createTestPNG()),
		})
	}))
	defer ts.Close()

	cfg := &agentsdk.Config{
		Providers: map[string]map[string]any{
			"spark": {"base_url": ts.URL},
		},
	}

	_, err := GenerateImage(context.Background(), "test", ImageOpts{
		Provider: "spark",
		Width:    512,
		Height:   512,
		Output:   filepath.Join(t.TempDir(), "out.png"),
		Config:   cfg,
	})
	if err != nil {
		t.Fatalf("GenerateImage error: %v", err)
	}
	if receivedModel != "z-image-turbo" {
		t.Errorf("default model should be z-image-turbo, got %q", receivedModel)
	}
}

// ── Spark integration tests (real server) ─────────────────────────────────

func skipIfSparkUnavailable(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping real spark test in short mode")
	}
	sparkURL := os.Getenv("SPARK_IMAGE_URL")
	if sparkURL == "" {
		sparkURL = "http://spark:8100"
	}
	conn, err := net.DialTimeout("tcp", strings.TrimPrefix(strings.TrimPrefix(sparkURL, "http://"), "https://"), 2*time.Second)
	if err != nil {
		t.Skipf("spark not reachable at %s: %v", sparkURL, err)
	}
	conn.Close()
}

func TestSparkIntegration_ZImageTurbo(t *testing.T) {
	skipIfSparkUnavailable(t)

	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "spark_zit.png")

	result, err := GenerateImage(context.Background(), "a red circle on white background", ImageOpts{
		Provider: "spark",
		Model:    "z-image-turbo",
		Width:    512,
		Height:   512,
		Output:   outputPath,
		Tier:     agentsdk.TierHigh,
		Timeout:  60 * time.Second,
	})
	if err != nil {
		t.Fatalf("Spark z-image-turbo generation failed: %v", err)
	}

	info, err := os.Stat(result.Path)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}
	t.Logf("Spark z-image-turbo: model_used=%s (%s), size=%d bytes",
		result.ModelUsed.ModelID, result.ModelUsed.DisplayName, info.Size())
	if result.ModelUsed.ModelID != "z-image-turbo" {
		t.Errorf("ModelUsed.ModelID=%q, want z-image-turbo", result.ModelUsed.ModelID)
	}
}

func TestSparkIntegration_FluxSchnell(t *testing.T) {
	skipIfSparkUnavailable(t)

	outputDir := t.TempDir()
	outputPath := filepath.Join(outputDir, "spark_flux.png")

	result, err := GenerateImage(context.Background(), "a blue square on black background", ImageOpts{
		Provider: "spark",
		Model:    "flux-schnell",
		Width:    512,
		Height:   512,
		Output:   outputPath,
		Tier:     agentsdk.TierHigh,
		Timeout:  60 * time.Second,
	})
	if err != nil {
		t.Fatalf("Spark flux-schnell generation failed: %v", err)
	}

	info, err := os.Stat(result.Path)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}
	t.Logf("Spark flux-schnell: model_used=%s (%s), size=%d bytes",
		result.ModelUsed.ModelID, result.ModelUsed.DisplayName, info.Size())
	if result.ModelUsed.ModelID != "flux-schnell" {
		t.Errorf("ModelUsed.ModelID=%q, want flux-schnell", result.ModelUsed.ModelID)
	}
}
