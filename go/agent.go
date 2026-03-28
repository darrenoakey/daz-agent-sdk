package dazagentsdk

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// Agent is the top-level API surface for all agent-sdk operations:
// Ask, Conversation, Image, Speak, Transcribe, and Models.
type Agent struct {
	config *Config

	// ImageFn is the function called by Image(). Set by capability package
	// registration or by the caller directly.
	ImageFn func(ctx context.Context, prompt string, opts ImageCallOpts) (*ImageResult, error)

	// SpeakFn is the function called by Speak(). Set by capability package
	// registration or by the caller directly.
	SpeakFn func(ctx context.Context, text string, opts SpeakCallOpts) (*AudioResult, error)

	// TranscribeFn is the function called by Transcribe(). Set by capability
	// package registration or by the caller directly.
	TranscribeFn func(ctx context.Context, audioPath string, opts TranscribeCallOpts) (string, error)

	// RemoveBackgroundFn is the function called by RemoveBackground().
	// Set by capability package registration or by the caller directly.
	RemoveBackgroundFn func(ctx context.Context, imagePath string, opts RemoveBackgroundCallOpts) (string, error)
}

// ImageCallOpts holds parameters for the image generation function.
type ImageCallOpts struct {
	Width          int
	Height         int
	Output         string
	Provider       string
	Model          string
	Image          string // input image path for i2i
	Steps          int    // inference steps override (0 = derive from tier)
	Tier           Tier
	Transparent    bool
	Timeout        time.Duration
	Config         *Config
	Logger         *ConversationLogger
}

// RemoveBackgroundCallOpts holds parameters for background removal.
type RemoveBackgroundCallOpts struct {
	Timeout time.Duration
	Config  *Config
}

// SpeakCallOpts holds parameters for the speech synthesis function.
type SpeakCallOpts struct {
	Voice   string
	Output  string
	Speed   float64
	Timeout time.Duration
}

// TranscribeCallOpts holds parameters for the transcription function.
type TranscribeCallOpts struct {
	ModelSize string
	Language  string
	Timeout   time.Duration
}

// NewAgent creates a new Agent. If cfg is nil, LoadConfig() defaults are used.
func NewAgent(cfg *Config) *Agent {
	if cfg == nil {
		var err error
		cfg, err = LoadConfig()
		if err != nil {
			cfg = &Config{}
			cfg.applyDefaults()
		}
	}
	return &Agent{config: cfg}
}

// Config returns the agent's configuration.
func (a *Agent) Config() *Config {
	return a.config
}

// Default is the package-level default Agent, lazily initialised on first use.
var Default *Agent

func init() {
	Default = NewAgent(nil)
}

// AskOption configures an Ask call via the functional options pattern.
type AskOption func(*askOpts)

type askOpts struct {
	tier     Tier
	schema   any
	system   string
	provider string
	model    string
	timeout  float64
	tools    []string
	cwd      string
	maxTurns int
}

// WithAskTier sets the model tier for the ask call.
func WithAskTier(tier Tier) AskOption {
	return func(o *askOpts) { o.tier = tier }
}

// WithAskSchema sets the JSON schema for structured output.
func WithAskSchema(schema any) AskOption {
	return func(o *askOpts) { o.schema = schema }
}

// WithAskSystem sets the system prompt for the ask call.
func WithAskSystem(system string) AskOption {
	return func(o *askOpts) { o.system = system }
}

// WithAskProvider pins the ask call to a specific provider name.
func WithAskProvider(provider string) AskOption {
	return func(o *askOpts) { o.provider = provider }
}

// WithAskModel pins the ask call to a specific model ID.
func WithAskModel(model string) AskOption {
	return func(o *askOpts) { o.model = model }
}

// WithAskTimeout sets the timeout in seconds for the ask call.
func WithAskTimeout(timeout float64) AskOption {
	return func(o *askOpts) { o.timeout = timeout }
}

// WithAskTools sets the list of tool names for agentic providers.
func WithAskTools(tools []string) AskOption {
	return func(o *askOpts) { o.tools = tools }
}

// WithAskCwd sets the working directory for agentic providers.
func WithAskCwd(cwd string) AskOption {
	return func(o *askOpts) { o.cwd = cwd }
}

// WithAskMaxTurns sets the maximum number of agentic turns.
func WithAskMaxTurns(n int) AskOption {
	return func(o *askOpts) { o.maxTurns = n }
}

// Ask sends a single-turn prompt and returns a Response. If provider and
// model are both specified via options, uses that pair directly. Otherwise
// resolves the tier chain and uses fallback.
func (a *Agent) Ask(ctx context.Context, prompt string, opts ...AskOption) (*Response, error) {
	o := &askOpts{
		tier:    TierHigh,
		timeout: 300.0,
	}
	for _, fn := range opts {
		fn(o)
	}

	var messages []Message
	if o.system != "" {
		messages = append(messages, Message{Role: "system", Content: o.system})
	}
	messages = append(messages, Message{Role: "user", Content: prompt})

	completeOpts := CompleteOpts{
		Schema:   o.schema,
		Timeout:  o.timeout,
		Tools:    o.tools,
		Cwd:      o.cwd,
		MaxTurns: o.maxTurns,
	}

	// Direct provider+model override
	if o.provider != "" && o.model != "" {
		prov := GetProvider(o.provider, a.config)
		if prov == nil {
			return nil, NewAgentError(
				fmt.Sprintf("Provider '%s' is not available", o.provider),
				ErrorNotAvailable, nil,
			)
		}
		minfo := ResolveModel(o.provider, o.model, &o.tier, a.config)
		if minfo == nil {
			return nil, NewAgentError(
				fmt.Sprintf("Model '%s:%s' could not be resolved", o.provider, o.model),
				ErrorNotAvailable, nil,
			)
		}
		return prov.Complete(ctx, messages, *minfo, completeOpts)
	}

	// Use tier chain with fallback
	chain := GetTierChain(o.tier, a.config)

	executeFn := func(providerEntry string) (*Response, error) {
		if !strings.Contains(providerEntry, ":") {
			return nil, NewAgentError(
				fmt.Sprintf("Invalid provider entry in chain: '%s'", providerEntry),
				ErrorInternal, nil,
			)
		}
		pname, mid := splitEntry(providerEntry)
		prov := GetProvider(pname, a.config)
		if prov == nil {
			return nil, NewAgentError(
				fmt.Sprintf("Provider '%s' is not available", pname),
				ErrorNotAvailable, nil,
			)
		}
		minfo := ResolveModel(pname, mid, &o.tier, a.config)
		if minfo == nil {
			return nil, NewAgentError(
				fmt.Sprintf("Model '%s:%s' could not be resolved", pname, mid),
				ErrorInternal, nil,
			)
		}
		return prov.Complete(ctx, messages, *minfo, completeOpts)
	}

	return ExecuteWithFallback(ctx, string(o.tier), chain, executeFn, a.config, false)
}

// Conversation creates a new multi-turn Conversation with the given name
// and options.
func (a *Agent) Conversation(name string, opts ...ConversationOption) *Conversation {
	// Inject the agent's config as a default
	opts = append([]ConversationOption{WithConfig(a.config)}, opts...)
	return NewConversation(name, opts...)
}

// ImageOpts holds optional parameters for Agent.Image.
type ImageOpts struct {
	Width       int
	Height      int
	Output      string
	Provider    string // "spark", "ollama", "nano-banana-2"
	Model       string // "z-image-turbo", "flux-schnell", etc.
	Image       string // input image path for image-to-image
	Steps       int    // inference steps override (0 = derive from tier)
	Tier        Tier
	Transparent bool
	Timeout     time.Duration
}

// Image generates an image from a text prompt. The ImageFn must be set
// (typically by importing the capability package and calling its
// registration function, or by the CLI).
func (a *Agent) Image(ctx context.Context, prompt string, opts ImageOpts) (*ImageResult, error) {
	if a.ImageFn == nil {
		return nil, NewAgentError(
			"image capability not registered; import capability package or set Agent.ImageFn",
			ErrorNotAvailable, nil,
		)
	}
	return a.ImageFn(ctx, prompt, ImageCallOpts{
		Width:       opts.Width,
		Height:      opts.Height,
		Output:      opts.Output,
		Provider:    opts.Provider,
		Model:       opts.Model,
		Image:       opts.Image,
		Steps:       opts.Steps,
		Tier:        opts.Tier,
		Transparent: opts.Transparent,
		Timeout:     opts.Timeout,
		Config:      a.config,
	})
}

// RemoveBackgroundOpts holds optional parameters for Agent.RemoveBackground.
type RemoveBackgroundOpts struct {
	Timeout time.Duration
}

// RemoveBackground removes the background from an image using the arbiter
// GPU service. Overwrites the image in-place with a PNG with alpha channel.
func (a *Agent) RemoveBackground(ctx context.Context, imagePath string, opts RemoveBackgroundOpts) (string, error) {
	if a.RemoveBackgroundFn == nil {
		return "", NewAgentError(
			"remove_background capability not registered; import capability package or set Agent.RemoveBackgroundFn",
			ErrorNotAvailable, nil,
		)
	}
	return a.RemoveBackgroundFn(ctx, imagePath, RemoveBackgroundCallOpts{
		Timeout: opts.Timeout,
		Config:  a.config,
	})
}

// SpeakOpts holds optional parameters for Agent.Speak.
type SpeakOpts struct {
	Voice   string
	Output  string
	Speed   float64
	Timeout time.Duration
}

// Speak converts text to speech audio. The SpeakFn must be set.
func (a *Agent) Speak(ctx context.Context, text string, opts SpeakOpts) (*AudioResult, error) {
	if a.SpeakFn == nil {
		return nil, NewAgentError(
			"speak capability not registered; import capability package or set Agent.SpeakFn",
			ErrorNotAvailable, nil,
		)
	}
	return a.SpeakFn(ctx, text, SpeakCallOpts{
		Voice:   opts.Voice,
		Output:  opts.Output,
		Speed:   opts.Speed,
		Timeout: opts.Timeout,
	})
}

// TranscribeOpts holds optional parameters for Agent.Transcribe.
type TranscribeOpts struct {
	ModelSize string
	Language  string
	Timeout   time.Duration
}

// Transcribe converts audio to text. The TranscribeFn must be set.
func (a *Agent) Transcribe(ctx context.Context, audioPath string, opts TranscribeOpts) (string, error) {
	if a.TranscribeFn == nil {
		return "", NewAgentError(
			"transcribe capability not registered; import capability package or set Agent.TranscribeFn",
			ErrorNotAvailable, nil,
		)
	}
	return a.TranscribeFn(ctx, audioPath, TranscribeCallOpts{
		ModelSize: opts.ModelSize,
		Language:  opts.Language,
		Timeout:   opts.Timeout,
	})
}

// ModelsOption configures a Models call via the functional options pattern.
type ModelsOption func(*modelsOpts)

type modelsOpts struct {
	tier       *Tier
	capability *Capability
}

// WithModelsTier filters models by tier.
func WithModelsTier(tier Tier) ModelsOption {
	return func(o *modelsOpts) { o.tier = &tier }
}

// WithModelsCapability filters models by capability.
func WithModelsCapability(cap Capability) ModelsOption {
	return func(o *modelsOpts) { o.capability = &cap }
}

// Models returns all known ModelInfo objects, optionally filtered by tier
// and/or capability.
func (a *Agent) Models(ctx context.Context, opts ...ModelsOption) ([]ModelInfo, error) {
	o := &modelsOpts{}
	for _, fn := range opts {
		fn(o)
	}

	if o.tier != nil {
		return GetModelsForTier(*o.tier, o.capability, a.config), nil
	}

	// Collect across all tiers, deduplicating by qualified name
	seen := map[string]bool{}
	var results []ModelInfo
	allTiers := []Tier{TierVeryHigh, TierHigh, TierMedium, TierLow, TierFreeFast, TierFreeThinking}
	for _, t := range allTiers {
		for _, m := range GetModelsForTier(t, o.capability, a.config) {
			qn := m.QualifiedName()
			if !seen[qn] {
				seen[qn] = true
				results = append(results, m)
			}
		}
	}
	return results, nil
}
