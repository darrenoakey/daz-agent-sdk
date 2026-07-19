package capability

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
	"github.com/google/uuid"
)

func TestImageServiceOriginIsCanonicalMacMini(t *testing.T) {
	source, err := os.ReadFile("image_codex.go")
	if err != nil {
		t.Fatalf("reading image service implementation: %v", err)
	}
	text := string(source)
	if strings.Count(text, `"http://10.0.0.46:8830"`) != 1 {
		t.Fatalf("canonical image service literal count is not one")
	}
	for _, forbidden := range []string{
		"127.0.0.1", "baseUrl", "serviceUrl", "os.Hostname", "imageServiceTestOriginKey", "imageServiceOriginForContext",
	} {
		if strings.Contains(text, forbidden) {
			t.Errorf("image service implementation retains alternate-route marker %q", forbidden)
		}
	}
}

func TestCurlImageServiceIsProxyAndRedirectImmune(t *testing.T) {
	source, err := os.ReadFile("image_codex.go")
	if err != nil {
		t.Fatalf("reading image service implementation: %v", err)
	}
	for _, argument := range []string{"--proxy", "--noproxy", "--proto", "--proto-redir", "--max-redirs"} {
		if !strings.Contains(string(source), argument) {
			t.Errorf("curl arguments omit %s", argument)
		}
	}
}

func TestGenerateImageRejectsEveryLegacyProviderBeforeNetwork(t *testing.T) {
	providers := []string{"spark", "arbiter", "flux", "mflux", "z-image-turbo", "ollama", "gemini", "nano-banana-2", "openai"}
	for _, provider := range providers {
		_, err := GenerateImage(context.Background(), "test", ImageOpts{Provider: provider, Width: 32, Height: 32})
		if err == nil || !strings.Contains(err.Error(), "actively disabled") {
			t.Errorf("provider %q returned %v, want actively-disabled error", provider, err)
		}
	}
}

func TestGenerateImageRejectsLegacyModelAndStepsPins(t *testing.T) {
	models := []string{"flux-schnell", "z-image-turbo", "gemini-3.1-flash-image-preview", "gpt-image-1"}
	for _, model := range models {
		err := validateImageRoute(ImageOpts{Model: model})
		if err == nil || !strings.Contains(err.Error(), "owns model selection") {
			t.Errorf("model %q returned %v, want service-owned selection error", model, err)
		}
	}
	if err := validateImageRoute(ImageOpts{Steps: 4}); err == nil {
		t.Fatal("expected image step pin to fail closed")
	}
}

func TestPublicImageEntrypointsRejectLegacyConfigBeforeInputOrNetwork(t *testing.T) {
	configurations := []*agentsdk.Config{
		{Image: agentsdk.ImageConfig{Model: "old-model"}},
		{Image: agentsdk.ImageConfig{CodexModel: "old-codex-model"}},
		{Image: agentsdk.ImageConfig{Tiers: map[string]agentsdk.ImageTierConfig{"high": {Steps: 4}}}},
		{Image: agentsdk.ImageConfig{Fallback: []string{"spark"}}},
		{Image: agentsdk.ImageConfig{TransparentPostProcess: "arbiter"}},
	}
	for _, configuration := range configurations {
		options := ImageOpts{
			Width: 64, Height: 64, Image: filepath.Join(t.TempDir(), "missing.png"),
			Output: filepath.Join(t.TempDir(), "output.png"), IdempotencyKey: "durable-key", Config: configuration,
		}
		result, resultError := GenerateImage(context.Background(), "config rejection", options)
		assertLegacyResultError(t, result, resultError)
		submission, submissionError := SubmitImageJob(context.Background(), "config rejection", options)
		assertLegacySubmissionError(t, submission, submissionError)
		status, statusError := GetImageJob(context.Background(), "invalid/job", configuration)
		assertLegacyStatusError(t, status, statusError)
		result, resultError = ResumeImageJob(context.Background(), "job", ImageJobOpts{Config: configuration})
		assertLegacyResultError(t, result, resultError)
		result, resultError = DownloadImageJob(context.Background(), "job", options.Output, false, configuration)
		assertLegacyResultError(t, result, resultError)
		result, resultError = ResumeImageOperation(context.Background(), filepath.Join(t.TempDir(), "missing.json"), configuration)
		assertLegacyResultError(t, result, resultError)
		submission, submissionError = RecoverImageSubmission(context.Background(), []byte(`{"prompt":"legacy"}`), "durable-key", configuration)
		assertLegacySubmissionError(t, submission, submissionError)
		submission, submissionError = WaitImageSubmission(context.Background(), []byte(`{"prompt":"legacy"}`), "durable-key", configuration)
		assertLegacySubmissionError(t, submission, submissionError)
		if _, err := os.Stat(options.Output); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("legacy image config mutated output: %v", err)
		}
	}
}

func assertLegacySubmissionError(t *testing.T, submission *agentsdk.ImageSubmission, err error) {
	t.Helper()
	if submission != nil {
		t.Fatalf("legacy image config returned submission %+v", submission)
	}
	assertLegacyError(t, err)
}

func assertLegacyStatusError(t *testing.T, status *agentsdk.ImageJobStatus, err error) {
	t.Helper()
	if status != nil {
		t.Fatalf("legacy image config returned status %+v", status)
	}
	assertLegacyError(t, err)
}

func assertLegacyResultError(t *testing.T, result *agentsdk.ImageResult, err error) {
	t.Helper()
	if result != nil {
		t.Fatalf("legacy image config returned result %+v", result)
	}
	assertLegacyError(t, err)
}

func assertLegacyError(t *testing.T, err error) {
	t.Helper()
	var agentError *agentsdk.AgentError
	if !errors.As(err, &agentError) || agentError.Kind != agentsdk.ErrorInvalidRequest || !strings.Contains(err.Error(), "legacy image configuration") {
		t.Fatalf("legacy image config error = %v", err)
	}
}

func TestParseImageSubmissionPreservesReplayAndConflictIdentity(t *testing.T) {
	key := uuid.NewString()
	accepted := []byte(`{"id":"durable-job","idempotency_key":"` + key + `","replayed":true}`)
	replay, terminal, err := parseImageSubmission(accepted, http.StatusAccepted, key)
	if err != nil || !terminal || replay.JobID != "durable-job" || replay.IdempotencyKey != key || !replay.Replayed {
		t.Fatalf("accepted submission=%+v terminal=%v error=%v", replay, terminal, err)
	}
	conflict := []byte(`{"id":"durable-job","code":"idempotency_conflict","error":"conflict"}`)
	_, terminal, err = parseImageSubmission(conflict, http.StatusConflict, key)
	var agentError *agentsdk.AgentError
	if !terminal || !errors.As(err, &agentError) || agentError.Kind != agentsdk.ErrorInvalidRequest {
		t.Fatalf("conflict terminal=%v error=%v", terminal, err)
	}
	if len(agentError.Attempts) != 1 || agentError.Attempts[0]["job_id"] != "durable-job" || agentError.Attempts[0]["code"] != "idempotency_conflict" {
		t.Fatalf("conflict metadata = %+v", agentError.Attempts)
	}
}

func TestTerminalSubmissionErrorPreservesConflictAndExpirySemantics(t *testing.T) {
	for status, code := range map[int]string{http.StatusConflict: "idempotency_conflict", http.StatusGone: "idempotency_expired"} {
		errorValue := terminalSubmissionError(status, []byte("legacy terminal response"), "durable-key")
		var agentError *agentsdk.AgentError
		if !errors.As(errorValue, &agentError) || agentError.Kind != agentsdk.ErrorInvalidRequest {
			t.Fatalf("status %d error = %v", status, errorValue)
		}
		if len(agentError.Attempts) != 1 || agentError.Attempts[0]["code"] != code || agentError.Attempts[0]["recoverable"] != false {
			t.Fatalf("status %d metadata = %+v", status, agentError.Attempts)
		}
	}
}

func TestSubmitImageJobRejectsMissingKeyBeforeNetwork(t *testing.T) {
	_, err := SubmitImageJob(context.Background(), "required key proof", ImageOpts{Width: 64, Height: 64})
	var agentError *agentsdk.AgentError
	if !errors.As(err, &agentError) || agentError.Kind != agentsdk.ErrorInvalidRequest || !strings.Contains(err.Error(), "required") {
		t.Fatalf("missing key error = %v, want invalid request", err)
	}
}
