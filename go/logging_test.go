package dazagentsdk

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/google/uuid"
)

func TestMetaJSONCreation(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}
	id := uuid.New()

	cl := NewConversationLogger("test-conv", cfg, id)

	metaPath := filepath.Join(cl.LogDir(), "meta.json")
	data, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatalf("meta.json not created: %v", err)
	}

	var meta map[string]any
	if err := json.Unmarshal(data, &meta); err != nil {
		t.Fatalf("invalid JSON in meta.json: %v", err)
	}

	if meta["conversation_id"] != id.String() {
		t.Errorf("conversation_id = %v, want %v", meta["conversation_id"], id.String())
	}
	if meta["name"] != "test-conv" {
		t.Errorf("name = %v, want %q", meta["name"], "test-conv")
	}
	if _, ok := meta["created_at"]; !ok {
		t.Error("missing created_at in meta.json")
	}
}

func TestLogEventWritesJSONL(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}

	cl := NewConversationLogger("test", cfg)
	cl.LogEvent("user_message", map[string]any{"role": "user", "content": "hello"})

	eventsPath := filepath.Join(cl.LogDir(), "events.jsonl")
	data, err := os.ReadFile(eventsPath)
	if err != nil {
		t.Fatalf("events.jsonl not created: %v", err)
	}

	var event map[string]any
	if err := json.Unmarshal(data, &event); err != nil {
		t.Fatalf("invalid JSONL: %v", err)
	}

	if event["event"] != "user_message" {
		t.Errorf("event = %v, want %q", event["event"], "user_message")
	}
	if event["role"] != "user" {
		t.Errorf("role = %v, want %q", event["role"], "user")
	}
	if event["content"] != "hello" {
		t.Errorf("content = %v, want %q", event["content"], "hello")
	}
	if _, ok := event["ts"]; !ok {
		t.Error("missing ts in event")
	}
}

func TestMultipleEvents(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}

	cl := NewConversationLogger("multi", cfg)
	cl.LogEvent("event_a")
	cl.LogEvent("event_b", map[string]any{"key": "value"})
	cl.LogEvent("event_c")

	eventsPath := filepath.Join(cl.LogDir(), "events.jsonl")
	data, err := os.ReadFile(eventsPath)
	if err != nil {
		t.Fatalf("events.jsonl not created: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d", len(lines))
	}

	// Verify each line is valid JSON with the correct event type
	expectedEvents := []string{"event_a", "event_b", "event_c"}
	for i, line := range lines {
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("line %d invalid JSON: %v", i, err)
		}
		if event["event"] != expectedEvents[i] {
			t.Errorf("line %d event = %v, want %q", i, event["event"], expectedEvents[i])
		}
	}

	// Verify the extra key on event_b
	var eventB map[string]any
	_ = json.Unmarshal([]byte(lines[1]), &eventB)
	if eventB["key"] != "value" {
		t.Errorf("event_b key = %v, want %q", eventB["key"], "value")
	}
}

func TestCloseWritesConversationEnd(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}

	cl := NewConversationLogger("close-test", cfg)
	cl.LogEvent("some_event")
	cl.Close()

	eventsPath := filepath.Join(cl.LogDir(), "events.jsonl")
	data, err := os.ReadFile(eventsPath)
	if err != nil {
		t.Fatalf("events.jsonl not created: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d", len(lines))
	}

	var lastEvent map[string]any
	if err := json.Unmarshal([]byte(lines[1]), &lastEvent); err != nil {
		t.Fatalf("last line invalid JSON: %v", err)
	}
	if lastEvent["event"] != "conversation_end" {
		t.Errorf("last event = %v, want %q", lastEvent["event"], "conversation_end")
	}
}

func TestCloseIsIdempotent(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}

	cl := NewConversationLogger("idempotent", cfg)
	cl.Close()
	cl.Close()
	cl.Close()

	eventsPath := filepath.Join(cl.LogDir(), "events.jsonl")
	data, err := os.ReadFile(eventsPath)
	if err != nil {
		t.Fatalf("events.jsonl not created: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected 1 line (single conversation_end), got %d", len(lines))
	}
}

func TestThreadSafety(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}

	cl := NewConversationLogger("concurrent", cfg)

	const goroutines = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(n int) {
			defer wg.Done()
			cl.LogEvent("concurrent_event", map[string]any{"n": n})
		}(i)
	}
	wg.Wait()

	eventsPath := filepath.Join(cl.LogDir(), "events.jsonl")
	data, err := os.ReadFile(eventsPath)
	if err != nil {
		t.Fatalf("events.jsonl not created: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != goroutines {
		t.Fatalf("expected %d lines, got %d", goroutines, len(lines))
	}

	// Verify every line is valid JSON
	for i, line := range lines {
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("line %d invalid JSON: %v", i, err)
		}
		if event["event"] != "concurrent_event" {
			t.Errorf("line %d event = %v, want %q", i, event["event"], "concurrent_event")
		}
	}
}

func TestUUIDSerializationInExtra(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}

	cl := NewConversationLogger("uuid-test", cfg)
	turnID := uuid.New()
	cl.LogEvent("turn_start", map[string]any{"turn_id": turnID.String()})

	eventsPath := filepath.Join(cl.LogDir(), "events.jsonl")
	data, err := os.ReadFile(eventsPath)
	if err != nil {
		t.Fatalf("events.jsonl not created: %v", err)
	}

	var event map[string]any
	if err := json.Unmarshal(data, &event); err != nil {
		t.Fatalf("invalid JSONL: %v", err)
	}

	if event["turn_id"] != turnID.String() {
		t.Errorf("turn_id = %v, want %v", event["turn_id"], turnID.String())
	}
}

func TestLogDirPath(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}
	id := uuid.New()

	cl := NewConversationLogger("dir-test", cfg, id)

	expected := filepath.Join(tmpDir, id.String())
	if cl.LogDir() != expected {
		t.Errorf("LogDir() = %q, want %q", cl.LogDir(), expected)
	}
}

func TestConversationIDAccessor(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}
	id := uuid.New()

	cl := NewConversationLogger("id-test", cfg, id)

	if cl.ConversationID() != id {
		t.Errorf("ConversationID() = %v, want %v", cl.ConversationID(), id)
	}
}

func TestAutoGeneratedUUID(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &Config{Logging: LoggingConfig{Directory: tmpDir}}

	cl := NewConversationLogger("auto-uuid", cfg)

	if cl.ConversationID() == uuid.Nil {
		t.Error("auto-generated UUID should not be nil")
	}
}
