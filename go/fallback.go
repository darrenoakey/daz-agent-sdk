package dazagentsdk

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"
)

// Error message fragments used to classify provider exceptions.
// Ordered from most specific to least specific within each category.

var rateLimitFragments = []string{
	"rate limit",
	"rate_limit",
	"ratelimit",
	"429",
	"too many requests",
	"capacity",
	"overloaded",
	"quota",
}

var authFragments = []string{
	"401",
	"403",
	"unauthorized",
	"forbidden",
	"authentication",
	"api key",
	"api_key",
	"invalid key",
	"invalid_api_key",
	"permission denied",
}

var timeoutFragments = []string{
	"timeout",
	"timed out",
	"deadline exceeded",
	"read timeout",
	"connect timeout",
}

var invalidRequestFragments = []string{
	"400",
	"invalid request",
	"invalid_request",
	"bad request",
	"validation error",
	"schema",
	"malformed",
}

var notAvailableFragments = []string{
	"connection refused",
	"not available",
	"not_available",
	"service unavailable",
	"503",
	"offline",
	"unreachable",
	"name or service not known",
	"cannot connect",
	"no route to host",
}

// ClassifyError maps any error to an ErrorKind for fallback decision making.
// Checks the error message against known fragments.
// Returns ErrorInternal for anything that does not match a known category.
func ClassifyError(err error) ErrorKind {
	message := strings.ToLower(err.Error())

	// Check for context deadline exceeded (Go equivalent of TimeoutError)
	if err == context.DeadlineExceeded || strings.Contains(message, "context deadline exceeded") {
		return ErrorTimeout
	}

	// Check not_available first (matches Python ordering)
	for _, frag := range notAvailableFragments {
		if strings.Contains(message, frag) {
			return ErrorNotAvailable
		}
	}

	for _, frag := range rateLimitFragments {
		if strings.Contains(message, frag) {
			return ErrorRateLimit
		}
	}

	for _, frag := range authFragments {
		if strings.Contains(message, frag) {
			return ErrorAuth
		}
	}

	for _, frag := range timeoutFragments {
		if strings.Contains(message, frag) {
			return ErrorTimeout
		}
	}

	for _, frag := range invalidRequestFragments {
		if strings.Contains(message, frag) {
			return ErrorInvalidRequest
		}
	}

	return ErrorInternal
}

// ExecuteFn is the function signature for the operation to attempt on each
// provider entry. The string argument is the "provider:model" entry from
// the tier chain.
type ExecuteFn func(providerEntry string) (*Response, error)

// ExecuteWithFallback runs executeFn against each provider in the chain.
//
// Single-shot mode (isConversation=false):
//
//	Retries each provider up to cfg.Fallback.SingleShot.MaxRetries times
//	with exponential backoff starting at cfg.Fallback.SingleShot.RetryBaseSeconds,
//	then cascades to the next provider on exhaustion.
//	Raises immediately on auth or invalid_request.
//
// Conversation mode (isConversation=true):
//
//	Applies exponential backoff (1s, 2s, 4s... up to maxBackoff)
//	before cascading to the next provider.
//
// Returns the first successful result, or an AgentError with all
// attempt records if every provider in the chain fails.
func ExecuteWithFallback(
	ctx context.Context,
	tier string,
	chain []string,
	executeFn ExecuteFn,
	cfg *Config,
	isConversation bool,
) (*Response, error) {
	if cfg == nil {
		var err error
		cfg, err = LoadConfig("")
		if err != nil {
			cfg = &Config{}
			cfg.applyDefaults()
		}
	}

	maxBackoff := float64(cfg.Fallback.Conversation.MaxBackoffSeconds)
	if maxBackoff <= 0 {
		maxBackoff = 60
	}

	maxRetries := cfg.Fallback.SingleShot.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 3
	}
	retryBase := cfg.Fallback.SingleShot.RetryBaseSeconds
	if retryBase <= 0 {
		retryBase = 1.0
	}

	var attempts []map[string]any

	for index, providerEntry := range chain {
		// Conversation mode: exponential backoff before each cascade after first
		if isConversation && index > 0 {
			delay := math.Min(math.Pow(2, float64(index-1)), maxBackoff)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(delay * float64(time.Second))):
			}
		}

		// Determine retry count: single-shot retries per provider, conversation gets 1
		providerRetries := 1
		if !isConversation {
			providerRetries = maxRetries
		}

		for retry := 0; retry < providerRetries; retry++ {
			// Single-shot per-provider backoff: delay before each retry after the first
			if !isConversation && retry > 0 {
				delay := math.Min(retryBase*math.Pow(2, float64(retry-1)), 30.0)
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(time.Duration(delay * float64(time.Second))):
				}
			}

			attempt := map[string]any{
				"provider": providerEntry,
				"tier":     tier,
				"retry":    retry,
			}

			result, err := executeFn(providerEntry)
			if err == nil {
				attempt["success"] = true
				attempts = append(attempts, attempt)
				return result, nil
			}

			kind := ClassifyError(err)
			attempt["error"] = err.Error()
			attempt["kind"] = string(kind)
			attempt["success"] = false
			attempts = append(attempts, attempt)

			// AUTH and INVALID_REQUEST are caller bugs -- raise immediately
			if kind == ErrorAuth || kind == ErrorInvalidRequest {
				return nil, NewAgentError(
					fmt.Sprintf("Non-retryable error from %s: %v", providerEntry, err),
					kind,
					attempts,
				)
			}

			// TIMEOUT and NOT_AVAILABLE: no point retrying same provider
			if kind == ErrorTimeout || kind == ErrorNotAvailable {
				break
			}

			// RATE_LIMIT and INTERNAL: worth retrying with backoff (loop continues)
		}
	}

	// All providers exhausted
	return nil, NewAgentError(
		fmt.Sprintf("All providers in chain failed for tier '%s' after %d attempt(s)", tier, len(attempts)),
		ErrorNotAvailable,
		attempts,
	)
}
