package capability

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func TestGenerateImage_MockOllama(t *testing.T) {
	// Create a minimal 1x1 red PNG for testing
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

	cfg := &agentsdk.Config{}
	cfg.Image.Model = "z-image-turbo"
	cfg.Image.Tiers = map[string]agentsdk.ImageTierConfig{
		"high": {Steps: 3},
	}
	if cfg.Providers == nil {
		cfg.Providers = map[string]map[string]any{}
	}
	cfg.Providers["ollama"] = map[string]any{
		"base_url": server.URL,
	}

	result, err := GenerateImage(context.Background(), "a red circle", ImageOpts{
		Width:   512,
		Height:  512,
		Output:  outputPath,
		Tier:    agentsdk.TierHigh,
		Config:  cfg,
		Timeout: 10 * time.Second,
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

	// Verify the file exists and has content
	info, err := os.Stat(outputPath)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}

	// Verify the file content matches the PNG data
	written, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("reading output file: %v", err)
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

	cfg := &agentsdk.Config{}
	cfg.Image.Model = "z-image-turbo"
	cfg.Image.Tiers = map[string]agentsdk.ImageTierConfig{
		"high": {Steps: 3},
	}
	if cfg.Providers == nil {
		cfg.Providers = map[string]map[string]any{}
	}
	cfg.Providers["ollama"] = map[string]any{
		"base_url": server.URL,
	}

	_, err := GenerateImage(context.Background(), "test", ImageOpts{
		Config:  cfg,
		Timeout: 5 * time.Second,
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
		resp := ollamaGenerateResponse{
			Model: "x/z-image-turbo",
			Image: b64Image,
			Done:  true,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &agentsdk.Config{}
	cfg.Image.Model = "z-image-turbo"
	cfg.Image.Tiers = map[string]agentsdk.ImageTierConfig{
		"high": {Steps: 3},
	}
	if cfg.Providers == nil {
		cfg.Providers = map[string]map[string]any{}
	}
	cfg.Providers["ollama"] = map[string]any{
		"base_url": server.URL,
	}

	result, err := GenerateImage(context.Background(), "test prompt", ImageOpts{
		Config:  cfg,
		Timeout: 5 * time.Second,
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
	// Use a port that nothing is listening on
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := listener.Addr().String()
	listener.Close()

	cfg := &agentsdk.Config{}
	cfg.Image.Model = "z-image-turbo"
	cfg.Image.Tiers = map[string]agentsdk.ImageTierConfig{
		"high": {Steps: 3},
	}
	if cfg.Providers == nil {
		cfg.Providers = map[string]map[string]any{}
	}
	cfg.Providers["ollama"] = map[string]any{
		"base_url": "http://" + addr,
	}

	_, err = GenerateImage(context.Background(), "test", ImageOpts{
		Config:  cfg,
		Timeout: 2 * time.Second,
	})
	if err == nil {
		t.Fatal("expected error when Ollama is not running")
	}
}

func TestGenerateImage_RealOllama(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real Ollama test in short mode")
	}

	// Skip if Ollama is not running at localhost:11434
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

	// Check that the image model is available
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
		Width:   256,
		Height:  256,
		Output:  outputPath,
		Tier:    agentsdk.TierHigh,
		Timeout: 120 * time.Second,
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
