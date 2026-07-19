// Command agent-sdk provides a CLI for the daz-agent-sdk library.
//
// Usage:
//
//	agent-sdk ask "prompt" [--tier high]
//	agent-sdk image --prompt "..." --width 512 --height 512 [--output path]
//	agent-sdk models [--tier high]
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	sdk "github.com/darrenoakey/daz-agent-sdk/go"
	"github.com/darrenoakey/daz-agent-sdk/go/capability"
	"github.com/darrenoakey/daz-agent-sdk/go/provider"
	"github.com/google/uuid"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	// Register all provider factories
	sdk.RegisterProviderFactory("ollama", func(cfg *sdk.Config) sdk.Provider {
		baseURL := "http://localhost:11434"
		if cfg != nil {
			if p, ok := cfg.Providers["ollama"]; ok {
				if u, ok := p["base_url"].(string); ok && u != "" {
					baseURL = u
				}
			}
		}
		return provider.NewOllamaProvider(baseURL)
	})
	sdk.RegisterProviderFactory("claude", func(cfg *sdk.Config) sdk.Provider {
		return provider.NewClaudeProvider()
	})
	sdk.RegisterProviderFactory("openai", func(cfg *sdk.Config) sdk.Provider {
		return provider.NewOpenAIProvider()
	})
	sdk.RegisterProviderFactory("gemini", func(cfg *sdk.Config) sdk.Provider {
		return provider.NewGeminiProvider()
	})
	sdk.RegisterProviderFactory("arbiter", func(cfg *sdk.Config) sdk.Provider {
		baseURL := "http://10.0.0.254:8400"
		if cfg != nil {
			if p, ok := cfg.Providers["arbiter"]; ok {
				if u, ok := p["base_url"].(string); ok && u != "" {
					baseURL = u
				}
			}
		}
		return provider.NewArbiterProvider(baseURL)
	})

	command := os.Args[1]
	switch command {
	case "ask":
		os.Exit(runAsk(os.Args[2:]))
	case "image":
		os.Exit(runImage(os.Args[2:]))
	case "models":
		os.Exit(runModels(os.Args[2:]))
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("agent-sdk CLI -- one library, every AI capability")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  agent-sdk ask \"prompt\" [--tier high]")
	fmt.Println("  agent-sdk image --prompt \"...\" --width 512 --height 512 [--output path]")
	fmt.Println("  agent-sdk models [--tier high]")
}

func newAgent() *sdk.Agent {
	agent := sdk.NewAgent(nil)

	// Wire up capability functions to avoid circular imports in root package
	agent.ImageFn = func(ctx context.Context, prompt string, opts sdk.ImageCallOpts) (*sdk.ImageResult, error) {
		return capability.GenerateImage(ctx, prompt, capability.ImageOpts{
			Width:          opts.Width,
			Height:         opts.Height,
			Output:         opts.Output,
			Provider:       opts.Provider,
			Model:          opts.Model,
			Image:          opts.Image,
			Images:         opts.Images,
			Steps:          opts.Steps,
			Tier:           opts.Tier,
			Transparent:    opts.Transparent,
			Timeout:        opts.Timeout,
			Config:         opts.Config,
			Logger:         opts.Logger,
			ConversationID: uuid.New(),
			IdempotencyKey: opts.IdempotencyKey,
		})
	}

	agent.SpeakFn = func(ctx context.Context, text string, opts sdk.SpeakCallOpts) (*sdk.AudioResult, error) {
		return capability.SynthesizeSpeech(ctx, text, capability.SpeakOpts{
			Voice:          opts.Voice,
			Output:         opts.Output,
			Speed:          opts.Speed,
			Timeout:        opts.Timeout,
			ConversationID: uuid.New(),
		})
	}

	agent.TranscribeFn = func(ctx context.Context, audioPath string, opts sdk.TranscribeCallOpts) (string, error) {
		return capability.Transcribe(ctx, audioPath, capability.TranscribeOpts{
			ModelSize:      opts.ModelSize,
			Language:       opts.Language,
			Timeout:        opts.Timeout,
			ConversationID: uuid.New(),
		})
	}

	// Wire Embed to the arbiter provider — the only provider that
	// implements embedding. Look up the provider via the registry so
	// caller-pinned base_urls in config propagate.
	agent.EmbedFn = func(ctx context.Context, texts []string, opts sdk.EmbedCallOpts) (*sdk.EmbeddingResult, error) {
		prov := sdk.GetProvider("arbiter", opts.Config)
		if prov == nil {
			return nil, sdk.NewAgentError(
				"Arbiter provider is not registered for embeddings",
				sdk.ErrorNotAvailable, nil,
			)
		}
		arb, ok := prov.(*provider.ArbiterProvider)
		if !ok {
			return nil, sdk.NewAgentError(
				"Embed requires the arbiter provider but the registered provider is a different type",
				sdk.ErrorInternal, nil,
			)
		}
		return arb.Embed(ctx, texts, provider.EmbedOpts{
			Task:      opts.Task,
			BatchSize: opts.BatchSize,
			Timeout:   opts.Timeout,
		})
	}

	return agent
}

func runAsk(args []string) int {
	fs := flag.NewFlagSet("ask", flag.ExitOnError)
	tier := fs.String("tier", "high", "Model tier (very_high, high, medium, low, free_fast, free_thinking)")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}
	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Error: prompt is required")
		fmt.Fprintln(os.Stderr, "Usage: agent-sdk ask \"prompt\" [--tier high]")
		return 1
	}
	prompt := fs.Arg(0)

	agent := newAgent()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	result, err := agent.Ask(ctx, prompt, sdk.WithAskTier(sdk.Tier(*tier)))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}

	fmt.Println(result.Text)
	return 0
}

// stringSliceFlag is a flag.Value implementation that collects repeated string
// flags into a slice. Used for --image which may be repeated to attach
// multiple input images to a codex edit request.
type stringSliceFlag []string

func (s *stringSliceFlag) String() string {
	return strings.Join(*s, ",")
}
func (s *stringSliceFlag) Set(value string) error {
	*s = append(*s, value)
	return nil
}

func runImage(args []string) int {
	fs := flag.NewFlagSet("image", flag.ExitOnError)
	prompt := fs.String("prompt", "", "The image prompt (required)")
	width := fs.Int("width", 512, "Image width in pixels")
	height := fs.Int("height", 512, "Image height in pixels")
	output := fs.String("output", "", "Output file path (default: temp file)")
	provider := fs.String("provider", "", "Image provider (only codex is enabled)")
	transparent := fs.Bool("transparent", false, "Remove background after generation")
	statePath := fs.String("state", "", "Crash-safe image submission state file")
	idempotencyKey := fs.String("idempotency-key", "", "Durable image submission key")
	recoverPath := fs.String("recover", "", "Recover a crash-safe image state file")
	var images stringSliceFlag
	fs.Var(&images, "image", "Input image file for editing/reference. Repeat to attach multiple images (codex only).")
	fs.Var(&images, "i", "Alias for --image.")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}
	if normalized := strings.ToLower(strings.TrimSpace(*provider)); normalized != "" && normalized != "codex" {
		fmt.Fprintf(os.Stderr, "Error: image provider %q is actively disabled; use the Mac mini Codex image service\n", *provider)
		return 1
	}
	configuration, err := sdk.LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}
	if err := configuration.ValidateImageConfig(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}

	if *prompt == "" && *recoverPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --prompt is required")
		fmt.Fprintln(os.Stderr, "Usage: agent-sdk image --prompt \"...\" --width 512 --height 512 [--output path] [--image FILE ...] [--provider codex]")
		return 1
	}
	ctx := context.Background()
	if *recoverPath != "" {
		if *output != "" {
			fmt.Fprintln(os.Stderr, "Error: --output cannot mutate recovered image operation metadata")
			return 1
		}
		result, err := capability.ResumeImageOperation(ctx, *recoverPath, configuration)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			return 1
		}
		fmt.Println(result.Path)
		return 0
	}

	opts := capability.ImageOpts{
		Width:          *width,
		Height:         *height,
		Output:         *output,
		Provider:       *provider,
		Transparent:    *transparent,
		StatePath:      *statePath,
		IdempotencyKey: *idempotencyKey,
		Config:         configuration,
	}
	if len(images) == 1 {
		opts.Image = images[0]
	} else if len(images) > 1 {
		opts.Images = images
	}

	result, err := capability.GenerateImage(ctx, *prompt, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}

	fmt.Println(result.Path)
	return 0
}

func runModels(args []string) int {
	fs := flag.NewFlagSet("models", flag.ExitOnError)
	tier := fs.String("tier", "", "Filter by tier (optional)")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}

	agent := newAgent()
	ctx := context.Background()

	var modelOpts []sdk.ModelsOption
	if *tier != "" {
		modelOpts = append(modelOpts, sdk.WithModelsTier(sdk.Tier(*tier)))
	}

	models, err := agent.Models(ctx, modelOpts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}

	for _, m := range models {
		fmt.Printf("%s  %s  [%s]\n", m.QualifiedName(), m.DisplayName, m.Tier)
	}
	if len(models) == 0 {
		fmt.Println("No models available.")
	}
	return 0
}
