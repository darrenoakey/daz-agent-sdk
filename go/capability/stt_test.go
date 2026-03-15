package capability

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestBuildSTTCommand_Basic(t *testing.T) {
	cmd := buildSTTCommand("/tmp/audio.wav", "small", "")
	expected := []string{
		"whisper",
		"/tmp/audio.wav",
		"--model", "small",
		"--output_format", "txt",
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

func TestBuildSTTCommand_WithLanguage(t *testing.T) {
	cmd := buildSTTCommand("/tmp/audio.wav", "large-v3-turbo", "en")
	expected := []string{
		"whisper",
		"/tmp/audio.wav",
		"--model", "large-v3-turbo",
		"--output_format", "txt",
		"--language", "en",
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

func TestTranscribe_FileNotFound(t *testing.T) {
	_, err := Transcribe(t.Context(), "/nonexistent/audio.wav", TranscribeOpts{})
	if err == nil {
		t.Fatal("expected error for nonexistent audio file")
	}
}

func TestTranscribe_WhisperNotInstalled(t *testing.T) {
	// Skip if whisper is actually installed
	if _, err := exec.LookPath("whisper"); err == nil {
		t.Skip("whisper is installed, skipping not-installed test")
	}

	// Create a temp file to pass the existence check
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "test.wav")
	if err := os.WriteFile(tmpFile, []byte("fake audio data"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := Transcribe(t.Context(), tmpFile, TranscribeOpts{})
	if err == nil {
		t.Fatal("expected error when whisper is not installed")
	}
}
