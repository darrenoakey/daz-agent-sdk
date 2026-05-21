package capability

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

// codexImageDir is where codex writes generated PNGs, one folder per session.
func codexImageDir() string {
	return filepath.Join(os.Getenv("HOME"), ".codex", "generated_images")
}

// codexModelInfo describes the codex native image_gen.imagegen tool for result metadata.
var codexModelInfo = agentsdk.ModelInfo{
	Provider:     "codex",
	ModelID:      "codex-image-generation",
	DisplayName:  "Codex native image_gen.imagegen",
	Capabilities: []agentsdk.Capability{agentsdk.CapabilityImage},
	Tier:         agentsdk.TierHigh,
}

// codexImagePrompt returns the prompt that reliably gets the codex CLI to
// invoke its built-in image_gen.imagegen tool — never the skill, the arbiter,
// or an external shell script.
func codexImagePrompt(prompt string, width, height int, hasInputs bool, numInputs int) string {
	aspectHint := " (square)"
	if width > height {
		aspectHint = " (landscape orientation)"
	} else if height > width {
		aspectHint = " (portrait orientation)"
	}

	if hasInputs {
		action := "EDIT the attached image"
		if numInputs > 1 {
			action = fmt.Sprintf("COMBINE/EDIT the %d attached images into a single new image", numInputs)
		}
		return fmt.Sprintf(
			"You MUST call your built-in `image_gen.imagegen` tool exactly once to %s. "+
				"It is a NATIVE tool that you DO have access to in this environment — there is no need "+
				"to apologise or claim you cannot generate images. Do NOT run any shell command. "+
				"Do NOT call `~/bin/generate_image`, the `generate_image` skill, the `arbiter` tool, "+
				"or any external script. The ONLY acceptable action is invoking `image_gen.imagegen` "+
				"with a single `prompt` argument that fully describes the desired output image "+
				"(approx %dx%d%s). Use the attached image(s) as visual reference. "+
				"\n\nEdit description: %s\n\n"+
				"After the tool returns, reply with one short sentence confirming success.",
			action, width, height, aspectHint, prompt,
		)
	}
	return fmt.Sprintf(
		"You MUST call your built-in `image_gen.imagegen` tool exactly once. "+
			"It is a NATIVE tool that you DO have access to in this environment — there is no "+
			"need to apologise or claim you cannot generate images. Do NOT run any shell command. "+
			"Do NOT call `~/bin/generate_image`, the `generate_image` skill, the `arbiter` tool, "+
			"or any external script. The ONLY acceptable action is invoking `image_gen.imagegen` "+
			"with a single `prompt` argument that fully describes the desired image "+
			"(approx %dx%d%s). "+
			"\n\nDesired image: %s\n\n"+
			"After the tool returns, reply with one short sentence confirming success.",
		width, height, aspectHint, prompt,
	)
}

// generateCodex shells out to `codex exec` and asks the codex CLI to invoke
// its built-in `image_gen.imagegen` tool. The generated PNG is written into
// ~/.codex/generated_images/<thread_id>/ig_*.png; we recover it and copy to
// the requested output path. Multiple input images are passed via -i flags.
func generateCodex(ctx context.Context, prompt string, opts ImageOpts, timeout time.Duration) (*agentsdk.ImageResult, error) {
	if _, err := exec.LookPath("codex"); err != nil {
		return nil, agentsdk.NewAgentError(
			"codex CLI not found on PATH",
			agentsdk.ErrorNotAvailable, nil,
		)
	}

	inputs := opts.inputImages()
	for _, p := range inputs {
		if _, err := os.Stat(p); err != nil {
			return nil, agentsdk.NewAgentError(
				fmt.Sprintf("input image not found: %s", p),
				agentsdk.ErrorInvalidRequest, nil,
			)
		}
	}

	outputPath, err := resolveOutputPath(opts.Output)
	if err != nil {
		return nil, err
	}

	instruction := codexImagePrompt(prompt, opts.Width, opts.Height, len(inputs) > 0, len(inputs))

	args := []string{
		"exec",
		"--dangerously-bypass-approvals-and-sandbox",
		"--skip-git-repo-check",
		"--ephemeral",
		"--json",
		"-m", "gpt-5.3-codex",
	}
	for _, p := range inputs {
		args = append(args, "-i", p)
	}
	args = append(args, "--", instruction)

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(reqCtx, "codex", args...)
	// Strip CLAUDECODE — propagates from Claude Code sessions and segfaults codex.
	env := os.Environ()
	out := env[:0]
	for _, e := range env {
		if strings.HasPrefix(e, "CLAUDECODE=") {
			continue
		}
		out = append(out, e)
	}
	cmd.Env = out
	cmd.Stdin = nil

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("codex stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("codex stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("codex start failed: %v", err),
			agentsdk.ErrorInternal, nil,
		)
	}

	var stdoutBuf strings.Builder
	threadID := ""
	go func() {
		// Stream stderr to discard but consume to avoid blocking
		_, _ = io.Copy(io.Discard, stderrPipe)
	}()

	scanner := bufio.NewScanner(stdoutPipe)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		stdoutBuf.WriteString(line)
		stdoutBuf.WriteString("\n")
		if threadID == "" {
			var event struct {
				Type     string `json:"type"`
				ThreadID string `json:"thread_id"`
			}
			if err := json.Unmarshal([]byte(line), &event); err == nil {
				if event.Type == "thread.started" && event.ThreadID != "" {
					threadID = event.ThreadID
				}
			}
		}
	}
	if err := cmd.Wait(); err != nil {
		if reqCtx.Err() == context.DeadlineExceeded {
			return nil, agentsdk.NewAgentError(
				fmt.Sprintf("codex image generation timed out after %s", timeout),
				agentsdk.ErrorTimeout, nil,
			)
		}
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("codex exited with error: %v: %s", err, tail(stdoutBuf.String(), 500)),
			agentsdk.ErrorInternal, nil,
		)
	}

	// Recover the generated file.
	candidates := findCodexImages(threadID)
	if len(candidates) == 0 {
		return nil, agentsdk.NewAgentError(
			fmt.Sprintf("codex did not produce a generated image. stdout tail: %s", tail(stdoutBuf.String(), 500)),
			agentsdk.ErrorInternal, nil,
		)
	}

	newest := candidates[0]
	if err := copyFile(newest, outputPath); err != nil {
		return nil, fmt.Errorf("copying codex output to %s: %w", outputPath, err)
	}

	return &agentsdk.ImageResult{
		Path:      outputPath,
		ModelUsed: codexModelInfo,
		Prompt:    prompt,
		Width:     opts.Width,
		Height:    opts.Height,
	}, nil
}

// findCodexImages returns generated codex images sorted newest-first. When
// threadID is set, it scans that session's folder; otherwise it falls back to
// every session folder.
func findCodexImages(threadID string) []string {
	root := codexImageDir()
	dirs := []string{}
	if threadID != "" {
		dirs = append(dirs, filepath.Join(root, threadID))
	} else {
		entries, err := os.ReadDir(root)
		if err != nil {
			return nil
		}
		for _, e := range entries {
			if e.IsDir() {
				dirs = append(dirs, filepath.Join(root, e.Name()))
			}
		}
	}
	type fileInfo struct {
		path    string
		modTime time.Time
	}
	var files []fileInfo
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for _, e := range entries {
			name := e.Name()
			if !strings.HasPrefix(name, "ig_") || !strings.HasSuffix(name, ".png") {
				continue
			}
			info, err := e.Info()
			if err != nil {
				continue
			}
			files = append(files, fileInfo{path: filepath.Join(d, name), modTime: info.ModTime()})
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].modTime.After(files[j].modTime) })
	out := make([]string, 0, len(files))
	for _, f := range files {
		out = append(out, f.path)
	}
	return out
}

func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
