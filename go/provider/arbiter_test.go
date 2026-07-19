package provider

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// testArbiterModel returns a ModelInfo for qwen3.6-27b.
func testArbiterModel() sdk.ModelInfo {
	return sdk.ModelInfo{
		Provider:             "arbiter",
		ModelID:              "qwen3.6-27b",
		DisplayName:          "Test Arbiter Model",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierFreeThinking,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
		SupportsTools:        false,
	}
}

func requireArbiterProvider(t *testing.T) *ArbiterProvider {
	t.Helper()
	direct := NewArbiterProvider("")
	if ok, err := direct.Available(context.Background()); err == nil && ok {
		return direct
	}
	return newArbiterTunnelProvider(t)
}

func newArbiterTunnelProvider(t *testing.T) *ArbiterProvider {
	t.Helper()
	port := reserveLoopbackPort(t)
	forward := fmt.Sprintf("127.0.0.1:%d:127.0.0.1:8400", port)
	command := exec.Command("/usr/bin/ssh", "-N", "-o", "BatchMode=yes",
		"-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=10",
		"-L", forward, "darren@10.0.0.254")
	if err := command.Start(); err != nil {
		t.Fatalf("starting owned Arbiter SSH tunnel: %v", err)
	}
	t.Cleanup(func() { stopArbiterTunnel(t, command) })
	provider := NewArbiterProvider(fmt.Sprintf("http://127.0.0.1:%d", port))
	waitForArbiterTunnel(t, provider)
	return provider
}

func reserveLoopbackPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserving loopback port: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatalf("releasing reserved loopback port: %v", err)
	}
	return port
}

func waitForArbiterTunnel(t *testing.T, provider *ArbiterProvider) {
	t.Helper()
	deadline := time.NewTimer(15 * time.Second)
	ticker := time.NewTicker(50 * time.Millisecond)
	t.Cleanup(func() { deadline.Stop() })
	t.Cleanup(ticker.Stop)
	for {
		if ok, err := provider.Available(context.Background()); err == nil && ok {
			return
		}
		select {
		case <-deadline.C:
			t.Fatal("owned Arbiter SSH tunnel did not become ready")
		case <-ticker.C:
		}
	}
}

func stopArbiterTunnel(t *testing.T, command *exec.Cmd) {
	t.Helper()
	if err := command.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Errorf("stopping owned Arbiter SSH tunnel: %v", err)
	}
	if err := command.Wait(); err != nil {
		var exitError *exec.ExitError
		if !errors.As(err, &exitError) {
			t.Errorf("waiting for owned Arbiter SSH tunnel: %v", err)
		}
	}
}

func TestKnownArbiterTiers(t *testing.T) {
	if knownArbiterTiers["qwen3.6-27b"] != sdk.TierSummaries {
		t.Errorf("qwen3.6-27b should be Summaries")
	}
}

func TestNewArbiterProviderDefaults(t *testing.T) {
	p := NewArbiterProvider("")
	if p.Name() != "arbiter" {
		t.Errorf("Name() = %q, want arbiter", p.Name())
	}
	if p.baseURL != "http://10.0.0.254:8400" {
		t.Errorf("baseURL = %q, want http://10.0.0.254:8400", p.baseURL)
	}
}

func TestNewArbiterProviderCustomURL(t *testing.T) {
	p := NewArbiterProvider("http://myhost:9999/")
	if p.baseURL != "http://myhost:9999" {
		t.Errorf("baseURL = %q, want http://myhost:9999", p.baseURL)
	}
}

func TestArbiterAvailableReportsReachability(t *testing.T) {
	p := NewArbiterProvider("")
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available returned error: %v", err)
	}
	if ok {
		return
	}
	tunnelProvider := newArbiterTunnelProvider(t)
	ok, err = tunnelProvider.Available(context.Background())
	if err != nil {
		t.Fatalf("Available through owned SSH tunnel returned error: %v", err)
	}
	if !ok {
		t.Fatal("Arbiter unavailable through owned SSH tunnel")
	}
}

func TestArbiterAvailableWrongPortReturnsFalse(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	ok, err := p.Available(context.Background())
	if err != nil {
		t.Fatalf("Available returned error on unreachable: %v", err)
	}
	if ok {
		t.Error("Available should return false for unreachable arbiter")
	}
}

func TestArbiterListModels(t *testing.T) {
	p := requireArbiterProvider(t)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("ListModels returned no models")
	}

	names := map[string]bool{}
	for _, m := range models {
		if m.Provider != "arbiter" {
			t.Errorf("model %q has provider %q, want arbiter", m.ModelID, m.Provider)
		}
		if m.ModelID == "" {
			t.Errorf("model has empty ModelID")
		}
		names[m.ModelID] = true
	}
	if !names["qwen3.6-27b"] {
		t.Error("ListModels should include qwen3.6-27b")
	}
}

func TestArbiterListModelsWrongPortReturnsError(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	_, err := p.ListModels(context.Background())
	if err == nil {
		t.Error("ListModels should error when arbiter is unreachable")
	}
}

func TestArbiterCompleteSimplePrompt(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "user", Content: "What is 2+2? Reply with just the number."},
	}
	model := testArbiterModel()
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{Timeout: 180})
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if !strings.Contains(resp.Text, "4") {
		t.Errorf("response text = %q, expected to contain '4'", resp.Text)
	}
	if resp.ModelUsed.ModelID != model.ModelID {
		t.Errorf("ModelUsed = %q, want %q", resp.ModelUsed.ModelID, model.ModelID)
	}
}

func TestArbiterCompleteWithSystemMessage(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "system", Content: "You are a terse assistant. Be brief."},
		{Role: "user", Content: "What is the capital of France?"},
	}
	model := testArbiterModel()
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{Timeout: 180})
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if !strings.Contains(resp.Text, "Paris") {
		t.Errorf("response text = %q, expected to contain 'Paris'", resp.Text)
	}
}

func TestArbiterStreamYieldsChunks(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "user", Content: "Count from 1 to 5, one number per line."},
	}
	model := testArbiterModel()
	ch, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{Timeout: 180})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	var total string
	var chunkCount int
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		if chunk.Text != "" {
			chunkCount++
			total += chunk.Text
		}
	}
	if chunkCount == 0 {
		t.Error("Stream produced no non-empty chunks")
	}
	if strings.TrimSpace(total) == "" {
		t.Error("Stream produced empty full text")
	}
}

func TestArbiterStreamProducesCompleteResponse(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "user", Content: "What is 10 divided by 2? Reply with just the number."},
	}
	model := testArbiterModel()
	ch, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{Timeout: 180})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	var full string
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		full += chunk.Text
	}
	if !strings.Contains(full, "5") {
		t.Errorf("stream total = %q, expected to contain '5'", full)
	}
}

func TestArbiterCompleteWrongPortRaisesAgentError(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	messages := []sdk.Message{{Role: "user", Content: "hello"}}
	model := testArbiterModel()
	_, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{Timeout: 5})
	if err == nil {
		t.Fatal("Complete should error against unreachable arbiter")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("err is not *sdk.AgentError: %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorNotAvailable {
		t.Errorf("Kind = %v, want ErrorNotAvailable", agentErr.Kind)
	}
}

func TestArbiterStreamWrongPortRaisesAgentError(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	messages := []sdk.Message{{Role: "user", Content: "hello"}}
	model := testArbiterModel()
	_, err := p.Stream(context.Background(), messages, model, sdk.StreamOpts{Timeout: 5})
	if err == nil {
		t.Fatal("Stream should error against unreachable arbiter")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("err is not *sdk.AgentError: %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorNotAvailable {
		t.Errorf("Kind = %v, want ErrorNotAvailable", agentErr.Kind)
	}
}

// TestArbiterEmbedReturnsVectors verifies the arbiter's embed-text
// adapter submits, polls, and returns one vector per input text.
// Skipped when the arbiter is unreachable.
func TestArbiterEmbedReturnsVectors(t *testing.T) {
	p := requireArbiterProvider(t)

	texts := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Vector embeddings represent text as points in high-dimensional space.",
		"Nomic embed text v1.5 produces 768-dimensional L2-normalized vectors.",
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*60*1e9)
	defer cancel()
	res, err := p.Embed(ctx, texts, EmbedOpts{Task: "search_document", Timeout: 600})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}
	if res == nil {
		t.Fatal("Embed returned nil result")
	}
	if len(res.Embeddings) != len(texts) {
		t.Fatalf("got %d embeddings, want %d", len(res.Embeddings), len(texts))
	}
	for i, vec := range res.Embeddings {
		if len(vec) == 0 {
			t.Fatalf("embedding %d is empty", i)
		}
	}
	if res.Task != "search_document" {
		t.Errorf("Task = %q, want search_document", res.Task)
	}
	if res.ModelUsed.Provider != "arbiter" {
		t.Errorf("ModelUsed.Provider = %q, want arbiter", res.ModelUsed.Provider)
	}
}

// TestArbiterEmbedDimension verifies nomic-embed-text-v1.5 returns 768-dim
// vectors with a consistent Dimension field matching len(Embeddings[0]).
func TestArbiterEmbedDimension(t *testing.T) {
	p := requireArbiterProvider(t)

	ctx, cancel := context.WithTimeout(context.Background(), 10*60*1e9)
	defer cancel()
	res, err := p.Embed(ctx, []string{"dimension check"}, EmbedOpts{Timeout: 600})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}
	if len(res.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(res.Embeddings))
	}
	if res.Dimension != 768 {
		t.Errorf("Dimension = %d, want 768", res.Dimension)
	}
	if len(res.Embeddings[0]) != 768 {
		t.Errorf("len(Embeddings[0]) = %d, want 768", len(res.Embeddings[0]))
	}
	if res.Dimension != len(res.Embeddings[0]) {
		t.Errorf("Dimension %d != len(Embeddings[0]) %d", res.Dimension, len(res.Embeddings[0]))
	}
}

// TestArbiterEmbedEmptyTextsRaisesError verifies that calling Embed with
// no texts raises InvalidRequest rather than submitting a no-op job.
func TestArbiterEmbedEmptyTextsRaisesError(t *testing.T) {
	p := NewArbiterProvider("")
	_, err := p.Embed(context.Background(), nil, EmbedOpts{})
	if err == nil {
		t.Fatal("Embed([]) should error")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("err is not *sdk.AgentError: %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorInvalidRequest {
		t.Errorf("Kind = %v, want ErrorInvalidRequest", agentErr.Kind)
	}
}

// TestArbiterEmbedWrongPortRaisesAgentError verifies that an unreachable
// arbiter surfaces as an AgentError with kind NotAvailable (connection
// refused), not a leaked raw net/http error.
func TestArbiterEmbedWrongPortRaisesAgentError(t *testing.T) {
	p := NewArbiterProvider("http://localhost:19999")
	_, err := p.Embed(context.Background(), []string{"hi"}, EmbedOpts{Timeout: 5})
	if err == nil {
		t.Fatal("Embed should error against unreachable arbiter")
	}
	agentErr, ok := err.(*sdk.AgentError)
	if !ok {
		t.Fatalf("err is not *sdk.AgentError: %T: %v", err, err)
	}
	if agentErr.Kind != sdk.ErrorNotAvailable {
		t.Errorf("Kind = %v, want ErrorNotAvailable", agentErr.Kind)
	}
}

// TestArbiterCompleteQwenReasoning exercises qwen3.6-27b and verifies
// that the reasoning→content fallback path populates resp.Text even
// when the model is in reasoning mode. A cold qwen load can take up to 10 minutes.
func TestArbiterCompleteQwenReasoning(t *testing.T) {
	p := requireArbiterProvider(t)

	messages := []sdk.Message{
		{Role: "user", Content: "What is 2+2? Answer with just the number."},
	}
	model := sdk.ModelInfo{
		Provider:             "arbiter",
		ModelID:              "qwen3.6-27b",
		DisplayName:          "Qwen3.6 27B",
		Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
		Tier:                 sdk.TierFreeThinking,
		SupportsStreaming:    true,
		SupportsStructured:   true,
		SupportsConversation: true,
	}
	resp, err := p.Complete(context.Background(), messages, model, sdk.CompleteOpts{Timeout: 900})
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if strings.TrimSpace(resp.Text) == "" {
		t.Error("qwen response text is empty — reasoning fallback did not populate it")
	}
}

func TestArbiterLoopbackTunnelListsModels(t *testing.T) {
	provider := newArbiterTunnelProvider(t)
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels through owned SSH tunnel: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("ListModels through owned SSH tunnel returned no LLM models")
	}
}
