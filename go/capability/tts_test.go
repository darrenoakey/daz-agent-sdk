package capability

import (
	"os/exec"
	"testing"
)

func TestBuildTTSCommand(t *testing.T) {
	cmd := buildTTSCommand("hello world", "gary", "/tmp/out.wav", 1.5)
	expected := []string{
		"tts", "tts",
		"--text", "hello world",
		"--voice", "gary",
		"--output", "/tmp/out.wav",
		"--speed", "1.5",
	}

	if len(cmd) != len(expected) {
		t.Fatalf("expected %d args, got %d: %v", len(expected), len(cmd), cmd)
	}
	for i, arg := range expected {
		if cmd[i] != arg {
			t.Errorf("arg[%d]: expected %q, got %q", i, arg, cmd[i])
		}
	}
}

func TestBuildTTSCommand_DefaultSpeed(t *testing.T) {
	cmd := buildTTSCommand("test text", "aiden", "/tmp/test.wav", 1.0)
	// Speed should be "1" (no trailing zeros)
	speedArg := cmd[len(cmd)-1]
	if speedArg != "1" {
		t.Errorf("expected speed '1', got %q", speedArg)
	}
}

func TestBuildTTSCommand_FractionalSpeed(t *testing.T) {
	cmd := buildTTSCommand("test", "aiden", "/tmp/test.wav", 0.75)
	speedArg := cmd[len(cmd)-1]
	if speedArg != "0.75" {
		t.Errorf("expected speed '0.75', got %q", speedArg)
	}
}

func TestSynthesizeSpeech_TTSNotInstalled(t *testing.T) {
	// Skip if tts is actually installed
	if _, err := exec.LookPath("tts"); err == nil {
		t.Skip("tts is installed, skipping not-installed test")
	}

	_, err := SynthesizeSpeech(t.Context(), "hello", SpeakOpts{})
	if err == nil {
		t.Fatal("expected error when tts is not installed")
	}
}
