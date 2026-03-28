package dazagentsdk

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/google/uuid"
)

// ConversationOption configures a Conversation via the functional options
// pattern.
type ConversationOption func(*conversationOpts)

type conversationOpts struct {
	tier     Tier
	system   string
	provider string
	model    string
	config   *Config
}

// WithTier sets the default model tier for the conversation.
func WithTier(tier Tier) ConversationOption {
	return func(o *conversationOpts) { o.tier = tier }
}

// WithSystem sets the system prompt prepended to the conversation history.
func WithSystem(system string) ConversationOption {
	return func(o *conversationOpts) { o.system = system }
}

// WithProvider pins the conversation to a specific provider name.
func WithProvider(provider string) ConversationOption {
	return func(o *conversationOpts) { o.provider = provider }
}

// WithModel pins the conversation to a specific model ID.
func WithModel(model string) ConversationOption {
	return func(o *conversationOpts) { o.model = model }
}

// WithConfig overrides the default config for this conversation.
func WithConfig(cfg *Config) ConversationOption {
	return func(o *conversationOpts) { o.config = cfg }
}

// Conversation manages a multi-turn exchange with an AI provider. It
// maintains message history, uses fallback across providers, and logs
// all events to a ConversationLogger.
type Conversation struct {
	name           string
	tier           Tier
	system         string
	providerName   string
	modelID        string
	config         *Config
	history        []Message
	logger         *ConversationLogger
	conversationID uuid.UUID
	mu             sync.Mutex
}

// NewConversation creates a new conversation with the given name and options.
// The conversation logger is created immediately and assigned a unique ID.
func NewConversation(name string, opts ...ConversationOption) *Conversation {
	o := &conversationOpts{
		tier: TierHigh,
	}
	for _, fn := range opts {
		fn(o)
	}

	cfg := o.config
	if cfg == nil {
		var err error
		cfg, err = LoadConfig()
		if err != nil {
			cfg = &Config{}
			cfg.applyDefaults()
		}
	}

	convID := uuid.New()
	logger := NewConversationLogger(name, cfg, convID)

	c := &Conversation{
		name:           name,
		tier:           o.tier,
		system:         o.system,
		providerName:   o.provider,
		modelID:        o.model,
		config:         cfg,
		history:        nil,
		logger:         logger,
		conversationID: convID,
	}

	// Pre-populate system message if provided
	if o.system != "" {
		c.history = append(c.history, Message{Role: "system", Content: o.system})
	}

	return c
}

// SayOption configures a single Say call via the functional options pattern.
type SayOption func(*sayOpts)

type sayOpts struct {
	schema any
	tier   *Tier
}

// WithSaySchema sets the JSON schema for structured output on this Say call.
func WithSaySchema(schema any) SayOption {
	return func(o *sayOpts) { o.schema = schema }
}

// WithSayTier overrides the model tier for this specific Say call.
func WithSayTier(tier Tier) SayOption {
	return func(o *sayOpts) { o.tier = &tier }
}

// Say adds a user message to the history, sends all messages to the provider
// via the fallback engine, appends the assistant response to history, and
// returns the Response. Optional SayOption values allow per-call schema and
// tier overrides.
func (c *Conversation) Say(ctx context.Context, prompt string, opts ...SayOption) (*Response, error) {
	o := &sayOpts{}
	for _, fn := range opts {
		fn(o)
	}

	c.mu.Lock()
	c.history = append(c.history, Message{Role: "user", Content: prompt})
	messages := make([]Message, len(c.history))
	copy(messages, c.history)
	tier := c.tier
	if o.tier != nil {
		tier = *o.tier
	}
	providerName := c.providerName
	modelID := c.modelID
	cfg := c.config
	c.mu.Unlock()

	chain := c.buildChain(tier, providerName, modelID)

	completeOpts := CompleteOpts{
		Schema: o.schema,
	}

	executeFn := func(providerEntry string) (*Response, error) {
		pname, mid := splitEntry(providerEntry)
		prov := GetProvider(pname, cfg)
		if prov == nil {
			return nil, fmt.Errorf("provider '%s' is not available", pname)
		}
		minfo := ResolveModel(pname, mid, &tier, cfg)
		if minfo == nil {
			return nil, fmt.Errorf("model '%s:%s' could not be resolved", pname, mid)
		}
		return prov.Complete(ctx, messages, *minfo, completeOpts)
	}

	result, err := ExecuteWithFallback(ctx, string(tier), chain, executeFn, cfg, true)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	c.history = append(c.history, Message{Role: "assistant", Content: result.Text})
	c.mu.Unlock()

	c.logger.LogEvent("turn_complete", map[string]any{
		"model":   result.ModelUsed.QualifiedName(),
		"turn_id": result.TurnID.String(),
	})

	return result, nil
}

// Stream adds a user message to the history, opens a streaming connection to
// the provider, and returns a channel of StreamChunks. The full response text
// is appended to history once the channel is drained.
func (c *Conversation) Stream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	c.mu.Lock()
	c.history = append(c.history, Message{Role: "user", Content: prompt})
	messages := make([]Message, len(c.history))
	copy(messages, c.history)
	tier := c.tier
	providerName := c.providerName
	modelID := c.modelID
	cfg := c.config
	c.mu.Unlock()

	chain := c.buildChain(tier, providerName, modelID)

	// Resolve the first provider in the chain for streaming
	if len(chain) == 0 {
		return nil, NewAgentError("no providers in chain", ErrorNotAvailable, nil)
	}
	firstEntry := chain[0]
	pname, mid := splitEntry(firstEntry)
	prov := GetProvider(pname, cfg)
	if prov == nil {
		return nil, NewAgentError(
			fmt.Sprintf("provider '%s' is not available", pname),
			ErrorNotAvailable, nil,
		)
	}
	minfo := ResolveModel(pname, mid, &tier, cfg)
	if minfo == nil {
		return nil, NewAgentError(
			fmt.Sprintf("model '%s:%s' could not be resolved", pname, mid),
			ErrorNotAvailable, nil,
		)
	}

	srcCh, err := prov.Stream(ctx, messages, *minfo, StreamOpts{})
	if err != nil {
		return nil, err
	}

	outCh := make(chan StreamChunk)
	go func() {
		defer close(outCh)
		var fullText strings.Builder
		for chunk := range srcCh {
			if chunk.Err == nil {
				fullText.WriteString(chunk.Text)
			}
			outCh <- chunk
		}
		c.mu.Lock()
		c.history = append(c.history, Message{Role: "assistant", Content: fullText.String()})
		c.mu.Unlock()
	}()

	return outCh, nil
}

// Fork creates a new Conversation with a deep copy of the current history.
// The fork is independent: changes to one do not affect the other.
func (c *Conversation) Fork(name string) *Conversation {
	c.mu.Lock()
	historyCopy := make([]Message, len(c.history))
	copy(historyCopy, c.history)
	tier := c.tier
	providerName := c.providerName
	modelID := c.modelID
	cfg := c.config
	c.mu.Unlock()

	convID := uuid.New()
	logger := NewConversationLogger(name, cfg, convID)

	return &Conversation{
		name:           name,
		tier:           tier,
		providerName:   providerName,
		modelID:        modelID,
		config:         cfg,
		history:        historyCopy,
		logger:         logger,
		conversationID: convID,
	}
}

// History returns a copy of the current conversation history.
func (c *Conversation) History() []Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	result := make([]Message, len(c.history))
	copy(result, c.history)
	return result
}

// Name returns the conversation name.
func (c *Conversation) Name() string {
	return c.name
}

// ConversationID returns the unique ID for this conversation.
func (c *Conversation) ConversationID() uuid.UUID {
	return c.conversationID
}

// Close writes a final conversation_end event and closes the logger.
func (c *Conversation) Close() {
	if c.logger != nil {
		c.logger.Close()
	}
}

// buildChain produces the ordered list of "provider:model" entries for
// fallback. If provider and model are both set, returns a single-element
// chain. Otherwise uses the tier config chain.
func (c *Conversation) buildChain(tier Tier, providerName, modelID string) []string {
	if providerName != "" && modelID != "" {
		return []string{providerName + ":" + modelID}
	}
	return GetTierChain(tier, c.config)
}

// splitEntry splits a "provider:model_id" string into its two parts.
func splitEntry(entry string) (string, string) {
	idx := strings.Index(entry, ":")
	if idx < 0 {
		return "", ""
	}
	return entry[:idx], entry[idx+1:]
}
