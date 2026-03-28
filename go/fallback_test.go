package dazagentsdk

import (
	"context"
	"errors"
	"testing"

	"github.com/google/uuid"
)

func TestClassifyErrorRateLimit(t *testing.T) {
	tests := []string{
		"rate limit exceeded",
		"rate_limit error",
		"ratelimit hit",
		"HTTP 429 too many requests",
		"too many requests",
		"capacity exceeded",
		"server overloaded",
		"quota exceeded",
	}
	for _, msg := range tests {
		kind := ClassifyError(errors.New(msg))
		if kind != ErrorRateLimit {
			t.Errorf("ClassifyError(%q) = %q, want rate_limit", msg, kind)
		}
	}
}

func TestClassifyErrorAuth(t *testing.T) {
	tests := []string{
		"HTTP 401 unauthorized",
		"403 forbidden",
		"unauthorized access",
		"forbidden resource",
		"authentication failed",
		"invalid api key",
		"api_key is invalid",
		"invalid key provided",
		"invalid_api_key",
		"permission denied",
	}
	for _, msg := range tests {
		kind := ClassifyError(errors.New(msg))
		if kind != ErrorAuth {
			t.Errorf("ClassifyError(%q) = %q, want auth", msg, kind)
		}
	}
}

func TestClassifyErrorTimeout(t *testing.T) {
	tests := []string{
		"request timeout",
		"operation timed out",
		"deadline exceeded",
		"read timeout",
		"connect timeout",
	}
	for _, msg := range tests {
		kind := ClassifyError(errors.New(msg))
		if kind != ErrorTimeout {
			t.Errorf("ClassifyError(%q) = %q, want timeout", msg, kind)
		}
	}

	// Also test context.DeadlineExceeded
	kind := ClassifyError(context.DeadlineExceeded)
	if kind != ErrorTimeout {
		t.Errorf("ClassifyError(DeadlineExceeded) = %q, want timeout", kind)
	}
}

func TestClassifyErrorInvalidRequest(t *testing.T) {
	tests := []string{
		"HTTP 400 bad request",
		"invalid request format",
		"invalid_request error",
		"bad request",
		"validation error in field",
		"schema mismatch",
		"malformed input",
	}
	for _, msg := range tests {
		kind := ClassifyError(errors.New(msg))
		if kind != ErrorInvalidRequest {
			t.Errorf("ClassifyError(%q) = %q, want invalid_request", msg, kind)
		}
	}
}

func TestClassifyErrorNotAvailable(t *testing.T) {
	tests := []string{
		"connection refused",
		"service not available",
		"not_available",
		"service unavailable",
		"HTTP 503",
		"server offline",
		"host unreachable",
		"name or service not known",
		"cannot connect to host",
		"no route to host",
	}
	for _, msg := range tests {
		kind := ClassifyError(errors.New(msg))
		if kind != ErrorNotAvailable {
			t.Errorf("ClassifyError(%q) = %q, want not_available", msg, kind)
		}
	}
}

func TestClassifyErrorInternal(t *testing.T) {
	tests := []string{
		"something unexpected happened",
		"null pointer exception",
		"internal server error",
		"segfault",
	}
	for _, msg := range tests {
		kind := ClassifyError(errors.New(msg))
		if kind != ErrorInternal {
			t.Errorf("ClassifyError(%q) = %q, want internal", msg, kind)
		}
	}
}

func TestExecuteWithFallbackSuccess(t *testing.T) {
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	chain := []string{"provider_a:model1", "provider_b:model2"}
	fn := func(entry string) (*Response, error) {
		return &Response{
			Text:           "hello from " + entry,
			ConversationID: uuid.New(),
			TurnID:         uuid.New(),
		}, nil
	}

	resp, err := ExecuteWithFallback(context.Background(), "test", chain, fn, cfg, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Text != "hello from provider_a:model1" {
		t.Errorf("Text = %q, want 'hello from provider_a:model1'", resp.Text)
	}
}

func TestExecuteWithFallbackCascade(t *testing.T) {
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	chain := []string{"provider_a:model1", "provider_b:model2", "provider_c:model3"}
	callOrder := []string{}

	fn := func(entry string) (*Response, error) {
		callOrder = append(callOrder, entry)
		if entry == "provider_a:model1" {
			// connection refused → not_available → no per-provider retry, cascade immediately
			return nil, errors.New("connection refused")
		}
		if entry == "provider_b:model2" {
			// rate limit → retried up to MaxRetries (3) times before cascading
			return nil, errors.New("rate limit exceeded")
		}
		return &Response{
			Text:           "success from " + entry,
			ConversationID: uuid.New(),
			TurnID:         uuid.New(),
		}, nil
	}

	resp, err := ExecuteWithFallback(context.Background(), "test", chain, fn, cfg, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Text != "success from provider_c:model3" {
		t.Errorf("Text = %q, want 'success from provider_c:model3'", resp.Text)
	}
	// provider_a: 1 call (not_available → break)
	// provider_b: cfg.Fallback.SingleShot.MaxRetries calls (rate_limit → retry)
	// provider_c: 1 call (success)
	wantCalls := 1 + cfg.Fallback.SingleShot.MaxRetries + 1
	if len(callOrder) != wantCalls {
		t.Errorf("expected %d calls, got %d: %v", wantCalls, len(callOrder), callOrder)
	}
}

func TestExecuteWithFallbackAuthStopsImmediately(t *testing.T) {
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	chain := []string{"provider_a:model1", "provider_b:model2"}
	callCount := 0

	fn := func(entry string) (*Response, error) {
		callCount++
		return nil, errors.New("unauthorized: invalid api key")
	}

	_, err := ExecuteWithFallback(context.Background(), "test", chain, fn, cfg, false)
	if err == nil {
		t.Fatal("expected error for auth failure")
	}

	agentErr, ok := err.(*AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T", err)
	}
	if agentErr.Kind != ErrorAuth {
		t.Errorf("Kind = %q, want auth", agentErr.Kind)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call (auth stops immediately), got %d", callCount)
	}
	if len(agentErr.Attempts) != 1 {
		t.Errorf("expected 1 attempt, got %d", len(agentErr.Attempts))
	}
}

func TestExecuteWithFallbackInvalidRequestStopsImmediately(t *testing.T) {
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	chain := []string{"provider_a:model1", "provider_b:model2"}
	callCount := 0

	fn := func(entry string) (*Response, error) {
		callCount++
		return nil, errors.New("400 bad request: invalid_request")
	}

	_, err := ExecuteWithFallback(context.Background(), "test", chain, fn, cfg, false)
	if err == nil {
		t.Fatal("expected error for invalid request")
	}

	agentErr, ok := err.(*AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T", err)
	}
	// "400" matches invalid_request, but "bad request" also matches.
	// The not_available check runs first but "400" doesn't match not_available.
	// Actually "400" matches invalid_request fragments.
	if agentErr.Kind != ErrorInvalidRequest {
		t.Errorf("Kind = %q, want invalid_request", agentErr.Kind)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call (invalid_request stops immediately), got %d", callCount)
	}
}

func TestExecuteWithFallbackAllFail(t *testing.T) {
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	chain := []string{"provider_a:model1", "provider_b:model2"}

	fn := func(entry string) (*Response, error) {
		return nil, errors.New("connection refused")
	}

	_, err := ExecuteWithFallback(context.Background(), "test_tier", chain, fn, cfg, false)
	if err == nil {
		t.Fatal("expected error when all providers fail")
	}

	agentErr, ok := err.(*AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T", err)
	}
	if agentErr.Kind != ErrorNotAvailable {
		t.Errorf("Kind = %q, want not_available", agentErr.Kind)
	}
	if len(agentErr.Attempts) != 2 {
		t.Errorf("expected 2 attempts, got %d", len(agentErr.Attempts))
	}
}

func TestExecuteWithFallbackEmptyChain(t *testing.T) {
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	fn := func(entry string) (*Response, error) {
		return &Response{Text: "should not be called"}, nil
	}

	_, err := ExecuteWithFallback(context.Background(), "test", []string{}, fn, cfg, false)
	if err == nil {
		t.Fatal("expected error for empty chain")
	}

	agentErr, ok := err.(*AgentError)
	if !ok {
		t.Fatalf("expected AgentError, got %T", err)
	}
	if agentErr.Kind != ErrorNotAvailable {
		t.Errorf("Kind = %q, want not_available", agentErr.Kind)
	}
}

func TestExecuteWithFallbackSingleShotRetriesRateLimit(t *testing.T) {
	// Rate limit errors should be retried up to MaxRetries times per provider
	// before cascading. Use MaxRetries=2 to keep the test fast.
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")
	cfg.Fallback.SingleShot.MaxRetries = 2
	cfg.Fallback.SingleShot.RetryBaseSeconds = 0.001 // near-zero delay for fast tests

	callCount := 0
	chain := []string{"provider_a:model1", "provider_b:model2"}

	fn := func(entry string) (*Response, error) {
		callCount++
		if entry == "provider_a:model1" {
			return nil, errors.New("rate limit exceeded")
		}
		return &Response{
			Text:           "ok from " + entry,
			ConversationID: uuid.New(),
			TurnID:         uuid.New(),
		}, nil
	}

	resp, err := ExecuteWithFallback(context.Background(), "test", chain, fn, cfg, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Text != "ok from provider_b:model2" {
		t.Errorf("Text = %q, want 'ok from provider_b:model2'", resp.Text)
	}
	// provider_a retried MaxRetries=2 times, then provider_b succeeds on first try
	wantCalls := cfg.Fallback.SingleShot.MaxRetries + 1
	if callCount != wantCalls {
		t.Errorf("callCount = %d, want %d (maxRetries=%d + 1 success)", callCount, wantCalls, cfg.Fallback.SingleShot.MaxRetries)
	}
}

func TestExecuteWithFallbackSingleShotNoRetryOnNotAvailable(t *testing.T) {
	// NOT_AVAILABLE errors should not be retried — cascade immediately
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")
	cfg.Fallback.SingleShot.MaxRetries = 3
	cfg.Fallback.SingleShot.RetryBaseSeconds = 10.0 // would cause timeout if retried

	callCount := 0
	chain := []string{"provider_a:model1", "provider_b:model2"}

	fn := func(entry string) (*Response, error) {
		callCount++
		if entry == "provider_a:model1" {
			return nil, errors.New("connection refused")
		}
		return &Response{
			Text:           "ok",
			ConversationID: uuid.New(),
			TurnID:         uuid.New(),
		}, nil
	}

	resp, err := ExecuteWithFallback(context.Background(), "test", chain, fn, cfg, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Text != "ok" {
		t.Errorf("Text = %q, want 'ok'", resp.Text)
	}
	// provider_a: 1 call (not_available → break), provider_b: 1 call (success)
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2 (not_available should not be retried)", callCount)
	}
}

func TestExecuteWithFallbackUsesConfigMaxRetries(t *testing.T) {
	// Verify that MaxRetries=1 means exactly 1 attempt per provider
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")
	cfg.Fallback.SingleShot.MaxRetries = 1
	cfg.Fallback.SingleShot.RetryBaseSeconds = 0.001

	callCount := 0
	chain := []string{"provider_a:model1", "provider_b:model2"}

	fn := func(entry string) (*Response, error) {
		callCount++
		return nil, errors.New("rate limit exceeded")
	}

	_, err := ExecuteWithFallback(context.Background(), "test", chain, fn, cfg, false)
	if err == nil {
		t.Fatal("expected error when all providers fail")
	}
	// MaxRetries=1 means exactly 1 attempt per provider
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2 (MaxRetries=1, 2 providers)", callCount)
	}
}

func TestExecuteWithFallbackAttemptsTracking(t *testing.T) {
	cfg, _ := LoadConfig("/nonexistent/path/config.yaml")

	chain := []string{"p1:m1", "p2:m2"}

	fn := func(entry string) (*Response, error) {
		if entry == "p1:m1" {
			return nil, errors.New("rate limit exceeded")
		}
		return &Response{
			Text:           "ok",
			ConversationID: uuid.New(),
			TurnID:         uuid.New(),
		}, nil
	}

	resp, err := ExecuteWithFallback(context.Background(), "test", chain, fn, cfg, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Text != "ok" {
		t.Errorf("Text = %q, want ok", resp.Text)
	}
}
