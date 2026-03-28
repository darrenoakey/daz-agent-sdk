package dazagentsdk

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// defaultConfigPath returns ~/.daz-agent-sdk/config.yaml.
func defaultConfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".daz-agent-sdk", "config.yaml")
}

// TierConfig holds an ordered list of provider:model strings for a tier.
type TierConfig struct {
	Chain []string `yaml:"chain" json:"chain"`
}

// ImageTierConfig holds the step count for one image quality tier.
type ImageTierConfig struct {
	Steps int `yaml:"steps" json:"steps"`
}

// ImageConfig holds model selection and per-tier step counts for image
// generation.
type ImageConfig struct {
	Model                  string                     `yaml:"model" json:"model"`
	Tiers                  map[string]ImageTierConfig  `yaml:"tiers" json:"tiers"`
	Fallback               []string                   `yaml:"fallback" json:"fallback"`
	TransparentPostProcess string                     `yaml:"-" json:"transparent_post_process"`
}

// TtsVoiceConfig holds per-voice provider and id settings.
type TtsVoiceConfig struct {
	Provider    string `yaml:"provider" json:"provider"`
	VoiceID     string `yaml:"voice_id" json:"voice_id"`
	Description string `yaml:"description" json:"description"`
}

// TtsConfig holds the default provider chain and named voice definitions.
type TtsConfig struct {
	Default []string                  `yaml:"default" json:"default"`
	Voices  map[string]TtsVoiceConfig `yaml:"voices" json:"voices"`
}

// LoggingConfig holds directory, level and retention for conversation logs.
type LoggingConfig struct {
	Directory     string `yaml:"directory" json:"directory"`
	Level         string `yaml:"level" json:"level"`
	RetentionDays int    `yaml:"retention_days" json:"retention_days"`
}

// FallbackSingleShotConfig holds strategy for one-shot requests.
type FallbackSingleShotConfig struct {
	Strategy         string  `yaml:"strategy" json:"strategy"`
	MaxRetries       int     `yaml:"max_retries" json:"max_retries"`
	RetryBaseSeconds float64 `yaml:"retry_base_seconds" json:"retry_base_seconds"`
}

// FallbackConversationConfig holds strategy for multi-turn conversations.
type FallbackConversationConfig struct {
	Strategy          string `yaml:"strategy" json:"strategy"`
	MaxBackoffSeconds int    `yaml:"max_backoff_seconds" json:"max_backoff_seconds"`
	SummariseWith     string `yaml:"summarise_with" json:"summarise_with"`
}

// FallbackConfig holds top-level fallback settings.
type FallbackConfig struct {
	SingleShot   FallbackSingleShotConfig   `yaml:"single_shot" json:"single_shot"`
	Conversation FallbackConversationConfig `yaml:"conversation" json:"conversation"`
}

// Config is the top-level configuration loaded from
// ~/.daz-agent-sdk/config.yaml. All fields have sensible defaults so no
// config file is required.
type Config struct {
	Tiers     map[string]TierConfig      `yaml:"tiers" json:"tiers"`
	Providers map[string]map[string]any   `yaml:"providers" json:"providers"`
	Image     ImageConfig                `yaml:"image" json:"image"`
	TTS       TtsConfig                  `yaml:"tts" json:"tts"`
	Logging   LoggingConfig              `yaml:"logging" json:"logging"`
	Fallback  FallbackConfig             `yaml:"fallback" json:"fallback"`
}

// defaultTierChains returns the built-in tier chain mappings.
func defaultTierChains() map[string]TierConfig {
	return map[string]TierConfig{
		"very_high": {Chain: []string{
			"claude:claude-opus-4-6",
			"codex:gpt-5.3-codex",
			"gemini:gemini-2.5-pro",
		}},
		"high": {Chain: []string{
			"claude:claude-opus-4-6",
			"codex:gpt-5.3-codex",
			"gemini:gemini-2.5-pro",
		}},
		"medium": {Chain: []string{
			"claude:claude-sonnet-4-6",
			"codex:gpt-4.1",
			"gemini:gemini-2.5-flash",
		}},
		"low": {Chain: []string{
			"claude:claude-haiku-4-5-20251001",
			"gemini:gemini-2.5-flash-lite",
			"ollama:qwen3-8b",
		}},
		"free_fast": {Chain: []string{
			"ollama:qwen3-8b",
		}},
		"free_thinking": {Chain: []string{
			"ollama:qwen3-30b-32k",
			"ollama:deepseek-r1:14b",
		}},
	}
}

// defaultProviders returns the built-in provider configurations.
func defaultProviders() map[string]map[string]any {
	return map[string]map[string]any{
		"claude": {"permission_mode": "bypassPermissions"},
		"codex":  {},
		"gemini": {},
		"ollama": {"base_url": "http://localhost:11434"},
	}
}

// defaultImageTiers returns the built-in image step counts per tier.
func defaultImageTiers() map[string]ImageTierConfig {
	return map[string]ImageTierConfig{
		"very_high": {Steps: 8},
		"high":      {Steps: 3},
		"medium":    {Steps: 3},
		"low":       {Steps: 2},
	}
}

// applyDefaults fills in any missing fields with sensible defaults.
func (c *Config) applyDefaults() {
	// Tier chains
	if c.Tiers == nil {
		c.Tiers = make(map[string]TierConfig)
	}
	for k, v := range defaultTierChains() {
		if _, exists := c.Tiers[k]; !exists {
			c.Tiers[k] = v
		}
	}

	// Providers
	if c.Providers == nil {
		c.Providers = make(map[string]map[string]any)
	}
	for k, v := range defaultProviders() {
		if _, exists := c.Providers[k]; !exists {
			c.Providers[k] = v
		}
	}

	// Image defaults
	if c.Image.Model == "" {
		c.Image.Model = "z-image-turbo"
	}
	if c.Image.Tiers == nil {
		c.Image.Tiers = make(map[string]ImageTierConfig)
	}
	for k, v := range defaultImageTiers() {
		if _, exists := c.Image.Tiers[k]; !exists {
			c.Image.Tiers[k] = v
		}
	}
	if c.Image.TransparentPostProcess == "" {
		c.Image.TransparentPostProcess = "birefnet"
	}

	// TTS defaults
	if len(c.TTS.Default) == 0 {
		c.TTS.Default = []string{"local:qwen3-tts"}
	}
	if len(c.TTS.Voices) == 0 {
		c.TTS.Voices = map[string]TtsVoiceConfig{
			"gary":  {Provider: "local", VoiceID: "gary", Description: "British newsreader"},
			"aiden": {Provider: "local", VoiceID: "aiden"},
		}
	}

	// Logging defaults
	if c.Logging.Directory == "" {
		c.Logging.Directory = "~/.daz-agent-sdk/logs"
	}
	if c.Logging.Level == "" {
		c.Logging.Level = "info"
	}
	if c.Logging.RetentionDays == 0 {
		c.Logging.RetentionDays = 30
	}

	// Fallback defaults
	if c.Fallback.SingleShot.Strategy == "" {
		c.Fallback.SingleShot.Strategy = "immediate_cascade"
	}
	if c.Fallback.SingleShot.MaxRetries == 0 {
		c.Fallback.SingleShot.MaxRetries = 3
	}
	if c.Fallback.SingleShot.RetryBaseSeconds == 0 {
		c.Fallback.SingleShot.RetryBaseSeconds = 1.0
	}
	if c.Fallback.Conversation.Strategy == "" {
		c.Fallback.Conversation.Strategy = "backoff_then_cascade"
	}
	if c.Fallback.Conversation.MaxBackoffSeconds == 0 {
		c.Fallback.Conversation.MaxBackoffSeconds = 60
	}
	if c.Fallback.Conversation.SummariseWith == "" {
		c.Fallback.Conversation.SummariseWith = "free_thinking"
	}
}

// rawConfig mirrors the YAML structure for unmarshaling, handling the
// transparent section nesting that differs from the flat struct.
// In the YAML file, tiers are bare lists (not structs with a chain key):
//
//	tiers:
//	  high:
//	    - claude:claude-opus-4-6
type rawConfig struct {
	Tiers     map[string][]string       `yaml:"tiers"`
	Providers map[string]map[string]any `yaml:"providers"`
	Image     struct {
		Model       string                     `yaml:"model"`
		Tiers       map[string]ImageTierConfig  `yaml:"tiers"`
		Fallback    []string                   `yaml:"fallback"`
		Transparent struct {
			PostProcess string `yaml:"post_process"`
		} `yaml:"transparent"`
	} `yaml:"image"`
	TTS      TtsConfig     `yaml:"tts"`
	Logging  LoggingConfig `yaml:"logging"`
	Fallback FallbackConfig `yaml:"fallback"`
}

// LoadConfig reads the YAML config file and returns a Config with defaults
// applied for any missing values. If no path is provided, it reads from
// ~/.daz-agent-sdk/config.yaml. If the file does not exist, a Config with
// all defaults is returned.
func LoadConfig(path ...string) (*Config, error) {
	configPath := defaultConfigPath()
	if len(path) > 0 && path[0] != "" {
		configPath = path[0]
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			cfg := &Config{}
			cfg.applyDefaults()
			return cfg, nil
		}
		return nil, err
	}

	var raw rawConfig
	if err := yaml.Unmarshal(data, &raw); err != nil {
		// Bad YAML: return defaults
		cfg := &Config{}
		cfg.applyDefaults()
		return cfg, nil
	}

	// Convert raw tier lists to TierConfig structs
	tiers := make(map[string]TierConfig, len(raw.Tiers))
	for k, chain := range raw.Tiers {
		tiers[k] = TierConfig{Chain: chain}
	}

	cfg := &Config{
		Tiers:     tiers,
		Providers: raw.Providers,
		Image: ImageConfig{
			Model:                  raw.Image.Model,
			Tiers:                  raw.Image.Tiers,
			Fallback:               raw.Image.Fallback,
			TransparentPostProcess: raw.Image.Transparent.PostProcess,
		},
		TTS:      raw.TTS,
		Logging:  raw.Logging,
		Fallback: raw.Fallback,
	}
	cfg.applyDefaults()
	return cfg, nil
}

// GetTierChain returns the ordered list of "provider:model" strings for a
// given tier. Returns nil if the tier has no chain defined.
func GetTierChain(tier Tier, cfg *Config) []string {
	tc, ok := cfg.Tiers[string(tier)]
	if !ok {
		return nil
	}
	// Return a copy to prevent mutation
	chain := make([]string, len(tc.Chain))
	copy(chain, tc.Chain)
	return chain
}

// GetImageSteps returns the inference step count for a given image generation
// tier. Falls back to the medium tier's steps if the requested tier is not
// configured.
func GetImageSteps(tier Tier, cfg *Config) int {
	tc, ok := cfg.Image.Tiers[string(tier)]
	if !ok {
		// Fall back to medium
		if med, ok := cfg.Image.Tiers["medium"]; ok {
			return med.Steps
		}
		return 4
	}
	return tc.Steps
}
