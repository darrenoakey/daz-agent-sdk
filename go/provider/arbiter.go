package provider

import (
	"bytes"
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// knownArbiterTiers maps served-model-name strings to tiers for the
// LLM workers the arbiter exposes. Used when /v1/models does not carry
// enough metadata to infer a tier. Everything not listed falls back to
// FreeThinking since all arbiter LLMs are 20B+.
var knownArbiterTiers = map[string]sdk.Tier{
	"qwen3.6-27b": sdk.TierSummaries,
	"qwen3.6-35b": sdk.TierFreeThinking,
	"gpt-oss-20b": sdk.TierFreeFast,
	"gemma4-31b":  sdk.TierFreeThinking,
	"gemma4-26b":  sdk.TierFreeThinking,
}

// ArbiterProvider speaks the OpenAI-compatible /v1/chat/completions
// endpoint exposed by the spark arbiter (the GPU job server). Any
// vLLM-served model registered with the arbiter is reachable by its
// served-model-name. No authentication required.
type ArbiterProvider struct {
	baseURL string
	client  *http.Client
}

// NewArbiterProvider creates a new arbiter provider. If baseURL is
// empty it defaults to http://10.0.0.254:8400 — the hardwired spark
// endpoint. Trailing slashes are stripped so callers may pass either form.
func NewArbiterProvider(baseURL string) *ArbiterProvider {
	if baseURL == "" {
		baseURL = "http://10.0.0.254:8400"
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &ArbiterProvider{
		baseURL: baseURL,
		client:  &http.Client{},
	}
}

// Name returns "arbiter".
func (a *ArbiterProvider) Name() string {
	return "arbiter"
}

// Available probes /v1/models. A 200 response from the arbiter means
// it is up and routable. Returns (false, nil) rather than an error so
// fallback chains can treat an offline arbiter as a silent miss.
func (a *ArbiterProvider) Available(ctx context.Context) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.baseURL+"/v1/models", nil)
	if err != nil {
		return false, nil
	}
	resp, err := a.client.Do(req)
	if err != nil {
		return false, nil
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == 200, nil
}

// arbiterModelEntry is a single record in the arbiter's /v1/models JSON
// response. The arbiter returns its native schema here (not the OpenAI
// "data" envelope), so non-LLM workers appear alongside LLM workers —
// we filter by presence of LlmName.
type arbiterModelEntry struct {
	ModelID string `json:"model_id"`
	LlmName string `json:"llm_name"`
}

// ListModels fetches all LLM workers from the arbiter's /v1/models
// endpoint and returns them as ModelInfo. Vision, image, tts and video
// workers are filtered out by requiring a non-empty LlmName. Tier is
// taken from knownArbiterTiers when available, else FreeThinking.
func (a *ArbiterProvider) ListModels(ctx context.Context) ([]sdk.ModelInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.baseURL+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("arbiter /v1/models returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading /v1/models response: %w", err)
	}

	// Try raw array first (arbiter-native form), then OpenAI envelope {"data": [...]}.
	var entries []arbiterModelEntry
	if err := json.Unmarshal(body, &entries); err != nil {
		var envelope struct {
			Data []arbiterModelEntry `json:"data"`
		}
		if err2 := json.Unmarshal(body, &envelope); err2 != nil {
			return nil, fmt.Errorf("decoding /v1/models: %w", err)
		}
		entries = envelope.Data
	}

	var models []sdk.ModelInfo
	for _, entry := range entries {
		if entry.LlmName == "" {
			continue
		}
		tier, ok := knownArbiterTiers[entry.LlmName]
		if !ok {
			tier = sdk.TierFreeThinking
		}
		display := titleCase(strings.ReplaceAll(strings.ReplaceAll(entry.LlmName, "-", " "), "_", " "))
		models = append(models, sdk.ModelInfo{
			Provider:             "arbiter",
			ModelID:              entry.LlmName,
			DisplayName:          display,
			Capabilities:         []sdk.Capability{sdk.CapabilityText, sdk.CapabilityStructured},
			Tier:                 tier,
			SupportsStreaming:    true,
			SupportsStructured:   true,
			SupportsConversation: true,
			SupportsTools:        false,
		})
	}
	return models, nil
}

// openAIMessage is the role/content pair every OpenAI-compatible
// endpoint accepts.
type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// openAIRequest is the JSON payload for /v1/chat/completions.
type openAIRequest struct {
	Model          string          `json:"model"`
	Messages       []openAIMessage `json:"messages"`
	Stream         bool            `json:"stream"`
	ResponseFormat any             `json:"response_format,omitempty"`
}

// openAIResponse decodes the non-streaming /v1/chat/completions body.
// Content may be empty when the model is in reasoning mode — callers
// should fall through to Reasoning in that case so text is never blank.
type openAIResponse struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning"`
		} `json:"message"`
	} `json:"choices"`
	Usage map[string]any `json:"usage"`
}

// buildArbiterMessages converts sdk.Message entries into OpenAI payload form.
func buildArbiterMessages(messages []sdk.Message) []openAIMessage {
	out := make([]openAIMessage, len(messages))
	for i, m := range messages {
		out[i] = openAIMessage{Role: m.Role, Content: m.Content}
	}
	return out
}

// applySchemaInstruction mutates msgs to append the JSON schema as a
// system-message instruction — the arbiter's vLLM worker does not
// always honour the OpenAI response_format field, and schema-in-prompt
// is proven to work with every model we serve.
func applySchemaInstruction(msgs []openAIMessage, schema any) []openAIMessage {
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return msgs
	}
	instruction := fmt.Sprintf(
		"\n\nRespond ONLY with valid JSON that matches this schema:\n%s\nDo not include any explanation or markdown. Output raw JSON only.",
		string(schemaJSON),
	)
	lastSys := -1
	for i, m := range msgs {
		if m.Role == "system" {
			lastSys = i
		}
	}
	if lastSys >= 0 {
		msgs[lastSys].Content = msgs[lastSys].Content + instruction
		return msgs
	}
	sys := openAIMessage{Role: "system", Content: strings.TrimSpace(instruction)}
	return append([]openAIMessage{sys}, msgs...)
}

// Complete sends messages to the arbiter and returns a full response.
// When opts.Schema is non-nil, the JSON schema is appended to the
// system message AND sent as an OpenAI response_format hint. The
// provider does not parse the schema into a typed struct — the SDK's
// structured-output machinery in Conversation handles validation from
// the returned JSON text.
func (a *ArbiterProvider) Complete(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.CompleteOpts) (*sdk.Response, error) {
	// Only impose a wall-clock deadline when the caller asked for one. The
	// arbiter is queue-backed: a job can sit hundreds deep behind unrelated
	// traffic on spark, and a GPU cold-load is ~10 minutes. We do not want
	// to spuriously fail a perfectly valid queued request — the HTTP call
	// will return the moment the server produces a response.
	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(opts.Timeout*float64(time.Second)))
		defer cancel()
	}

	msgs := buildArbiterMessages(messages)
	var responseFormat any
	if opts.Schema != nil {
		msgs = applySchemaInstruction(msgs, opts.Schema)
		responseFormat = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response",
				"schema": opts.Schema,
				"strict": true,
			},
		}
	}

	payload := openAIRequest{
		Model:          model.ModelID,
		Messages:       msgs,
		Stream:         false,
		ResponseFormat: responseFormat,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to marshal request: %v", err), sdk.ErrorInternal, nil)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to create request: %v", err), sdk.ErrorInternal, nil)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := a.client.Do(req)
	if err != nil {
		kind := classifyHTTPError(err)
		return nil, sdk.NewAgentError(err.Error(), kind, nil)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 429 {
		return nil, sdk.NewAgentError(fmt.Sprintf("Arbiter rate limit: %d", resp.StatusCode), sdk.ErrorRateLimit, nil)
	}
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, sdk.NewAgentError(fmt.Sprintf("Arbiter error %d: %s", resp.StatusCode, string(respBody)), sdk.ErrorInternal, nil)
	}

	var chatResp openAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to decode response: %v", err), sdk.ErrorInternal, nil)
	}

	text := ""
	if len(chatResp.Choices) > 0 {
		// reasoning models (qwen3 via vLLM --reasoning-parser qwen3) put
		// the chain-of-thought in Reasoning and the final answer in
		// Content. prefer the answer; fall through to reasoning only
		// when Content is empty (e.g. the model truncated before
		// emitting its final answer).
		text = chatResp.Choices[0].Message.Content
		if text == "" {
			text = chatResp.Choices[0].Message.Reasoning
		}
	}

	usage := chatResp.Usage
	if usage == nil {
		usage = map[string]any{}
	}

	return &sdk.Response{
		Text:           text,
		ModelUsed:      model,
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
		Usage:          usage,
	}, nil
}

// Stream sends messages to the arbiter and yields text chunks as they
// arrive. The arbiter emits OpenAI-style Server-Sent Events — each
// line prefixed `data: ` with the terminator `data: [DONE]`.
// choices[0].delta.content (or reasoning for reasoning models) is
// emitted as each non-empty chunk.
func (a *ArbiterProvider) Stream(ctx context.Context, messages []sdk.Message, model sdk.ModelInfo, opts sdk.StreamOpts) (<-chan sdk.StreamChunk, error) {
	// Only impose a wall-clock deadline when the caller asked for one — the
	// arbiter queue can hold a job indefinitely and that is not a failure
	// condition. See the note in Complete() above.
	var cancel context.CancelFunc = func() {}
	if opts.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(opts.Timeout*float64(time.Second)))
	}

	payload := openAIRequest{
		Model:    model.ModelID,
		Messages: buildArbiterMessages(messages),
		Stream:   true,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to marshal request: %v", err), sdk.ErrorInternal, nil)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("failed to create request: %v", err), sdk.ErrorInternal, nil)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := a.client.Do(req)
	if err != nil {
		cancel()
		kind := classifyHTTPError(err)
		return nil, sdk.NewAgentError(err.Error(), kind, nil)
	}

	if resp.StatusCode == 429 {
		resp.Body.Close()
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("Arbiter rate limit: %d", resp.StatusCode), sdk.ErrorRateLimit, nil)
	}
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		cancel()
		return nil, sdk.NewAgentError(fmt.Sprintf("Arbiter error %d: %s", resp.StatusCode, string(respBody)), sdk.ErrorInternal, nil)
	}

	ch := make(chan sdk.StreamChunk)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		defer cancel()

		scanner := bufio.NewScanner(resp.Body)
		// arbiter SSE frames can be large when the model emits big JSON
		// blocks — bump the scanner buffer well past the default 64K.
		scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			payloadStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payloadStr == "[DONE]" {
				return
			}
			var obj struct {
				Choices []struct {
					Delta struct {
						Content   string `json:"content"`
						Reasoning string `json:"reasoning"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(payloadStr), &obj); err != nil {
				continue
			}
			if len(obj.Choices) == 0 {
				continue
			}
			chunk := obj.Choices[0].Delta.Content
			if chunk == "" {
				chunk = obj.Choices[0].Delta.Reasoning
			}
			if chunk != "" {
				ch <- sdk.StreamChunk{Text: chunk}
			}
		}
		if err := scanner.Err(); err != nil {
			ch <- sdk.StreamChunk{Err: fmt.Errorf("arbiter stream scan error: %w", err)}
		}
	}()

	return ch, nil
}

// EmbedOpts holds optional parameters for ArbiterProvider.Embed.
// Defaults: Task="search_document", BatchSize=16, Timeout=600.
type EmbedOpts struct {
	Task      string
	BatchSize int
	Timeout   float64
}

// embedSubmitResponse decodes the POST /v1/jobs response.
type embedSubmitResponse struct {
	JobID string `json:"job_id"`
}

// embedJobStatus decodes the GET /v1/jobs/<id> response.
// result is raw JSON because embeddings nest under it.
type embedJobStatus struct {
	Status string          `json:"status"`
	Error  string          `json:"error"`
	Result json.RawMessage `json:"result"`
}

// embedJobResult is the shape of the result field when status=completed.
// The arbiter's embed-text adapter returns one float64 vector per input.
type embedJobResult struct {
	Embeddings      [][]float64    `json:"embeddings"`
	Dimension       int            `json:"dimension"`
	ModelRepository string         `json:"model_repository"`
	ElapsedMs       *float64       `json:"elapsed_ms"`
	Count           *int           `json:"count"`
	Extra           map[string]any `json:"-"`
}

// Embed submits an embed-text job to the arbiter, polls /v1/jobs/<id>
// every 1s until a terminal state, and returns the parsed result.
// The arbiter job API is async-by-design — submit returns a job_id,
// then poll /v1/jobs/<id> until the job finishes. nomic-embed-text-v1.5
// is ~50ms/text warm, ~5s cold.
func (a *ArbiterProvider) Embed(ctx context.Context, texts []string, opts EmbedOpts) (*sdk.EmbeddingResult, error) {
	if len(texts) == 0 {
		return nil, sdk.NewAgentError(
			"Embed() requires at least one text",
			sdk.ErrorInvalidRequest, nil,
		)
	}

	task := opts.Task
	if task == "" {
		task = "search_document"
	}
	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = 16
	}
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 600.0
	}

	payload := map[string]any{
		"type": "embed-text",
		"params": map[string]any{
			"texts":      texts,
			"task":       task,
			"batch_size": batchSize,
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, sdk.NewAgentError(
			fmt.Sprintf("failed to marshal embed request: %v", err),
			sdk.ErrorInternal, nil,
		)
	}

	overallCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	// Submit phase — arbiter can be slow to acknowledge on cold models,
	// so allow up to 120s for the POST to return a job_id.
	submitCtx, submitCancel := context.WithTimeout(overallCtx, 120*time.Second)
	req, err := http.NewRequestWithContext(submitCtx, http.MethodPost, a.baseURL+"/v1/jobs", bytes.NewReader(body))
	if err != nil {
		submitCancel()
		return nil, sdk.NewAgentError(
			fmt.Sprintf("failed to create embed submit request: %v", err),
			sdk.ErrorInternal, nil,
		)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := a.client.Do(req)
	if err != nil {
		submitCancel()
		kind := classifyHTTPError(err)
		return nil, sdk.NewAgentError(err.Error(), kind, nil)
	}
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		submitCancel()
		return nil, sdk.NewAgentError(
			fmt.Sprintf("Arbiter embed submit error %d: %s", resp.StatusCode, string(respBody)),
			sdk.ErrorInternal, nil,
		)
	}
	var submit embedSubmitResponse
	if err := json.NewDecoder(resp.Body).Decode(&submit); err != nil {
		resp.Body.Close()
		submitCancel()
		return nil, sdk.NewAgentError(
			fmt.Sprintf("failed to decode embed submit response: %v", err),
			sdk.ErrorInternal, nil,
		)
	}
	resp.Body.Close()
	submitCancel()
	if submit.JobID == "" {
		return nil, sdk.NewAgentError(
			"Arbiter returned no job_id for embed submit",
			sdk.ErrorInternal, nil,
		)
	}

	// Poll phase — 1s interval, 10s per-request timeout, up to overall timeout.
	pollURL := fmt.Sprintf("%s/v1/jobs/%s", a.baseURL, submit.JobID)
	for {
		select {
		case <-overallCtx.Done():
			return nil, sdk.NewAgentError(
				fmt.Sprintf("Arbiter embed job %s timed out after %.0fs", submit.JobID, timeout),
				sdk.ErrorTimeout, nil,
			)
		default:
		}

		pollCtx, pollCancel := context.WithTimeout(overallCtx, 10*time.Second)
		pollReq, err := http.NewRequestWithContext(pollCtx, http.MethodGet, pollURL, nil)
		if err != nil {
			pollCancel()
			return nil, sdk.NewAgentError(
				fmt.Sprintf("failed to create embed poll request: %v", err),
				sdk.ErrorInternal, nil,
			)
		}
		pollResp, err := a.client.Do(pollReq)
		if err != nil {
			pollCancel()
			kind := classifyHTTPError(err)
			return nil, sdk.NewAgentError(err.Error(), kind, nil)
		}
		if pollResp.StatusCode >= 400 {
			respBody, _ := io.ReadAll(pollResp.Body)
			pollResp.Body.Close()
			pollCancel()
			return nil, sdk.NewAgentError(
				fmt.Sprintf("Arbiter embed poll error %d: %s", pollResp.StatusCode, string(respBody)),
				sdk.ErrorInternal, nil,
			)
		}
		var status embedJobStatus
		if err := json.NewDecoder(pollResp.Body).Decode(&status); err != nil {
			pollResp.Body.Close()
			pollCancel()
			return nil, sdk.NewAgentError(
				fmt.Sprintf("failed to decode embed poll response: %v", err),
				sdk.ErrorInternal, nil,
			)
		}
		pollResp.Body.Close()
		pollCancel()

		switch status.Status {
		case "completed":
			var result embedJobResult
			if len(status.Result) > 0 {
				if err := json.Unmarshal(status.Result, &result); err != nil {
					return nil, sdk.NewAgentError(
						fmt.Sprintf("failed to decode embed result: %v", err),
						sdk.ErrorInternal, nil,
					)
				}
			}
			dimension := result.Dimension
			if dimension == 0 && len(result.Embeddings) > 0 {
				dimension = len(result.Embeddings[0])
			}
			modelID := result.ModelRepository
			if modelID == "" {
				modelID = "embed-text"
			}
			usage := map[string]any{}
			if result.ElapsedMs != nil {
				usage["elapsed_ms"] = *result.ElapsedMs
			}
			if result.Count != nil {
				usage["count"] = *result.Count
			}
			modelInfo := sdk.ModelInfo{
				Provider:             "arbiter",
				ModelID:              "embed-text",
				DisplayName:          modelID,
				Capabilities:         []sdk.Capability{sdk.CapabilityEmbedding},
				Tier:                 sdk.TierFreeFast,
				SupportsStreaming:    false,
				SupportsStructured:   false,
				SupportsConversation: false,
				SupportsTools:        false,
			}
			return &sdk.EmbeddingResult{
				Embeddings: result.Embeddings,
				ModelUsed:  modelInfo,
				Dimension:  dimension,
				Task:       task,
				Usage:      usage,
			}, nil
		case "failed", "cancelled":
			return nil, sdk.NewAgentError(
				fmt.Sprintf("Arbiter embed job %s %s: %s", submit.JobID, status.Status, status.Error),
				sdk.ErrorInternal, nil,
			)
		}

		// Not terminal — wait 1s and retry.
		select {
		case <-overallCtx.Done():
			return nil, sdk.NewAgentError(
				fmt.Sprintf("Arbiter embed job %s timed out after %.0fs", submit.JobID, timeout),
				sdk.ErrorTimeout, nil,
			)
		case <-time.After(1 * time.Second):
		}
	}
}

// Compile-time check that ArbiterProvider satisfies Provider.
var _ sdk.Provider = (*ArbiterProvider)(nil)
