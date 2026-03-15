package dazagentsdk

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// defaultLogBase returns ~/.daz-agent-sdk/logs as the fallback log directory.
func defaultLogBase() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".daz-agent-sdk", "logs")
}

// ConversationLogger writes structured JSONL logs for a single conversation
// into its own directory. All file writes are protected by a mutex.
// IO errors are swallowed — logging failures must never crash the caller.
type ConversationLogger struct {
	conversationID uuid.UUID
	name           string
	logDir         string
	mu             sync.Mutex
	closed         bool
}

// NewConversationLogger creates a new logger, creates the log directory, and
// writes the initial meta.json. If no conversationID is provided, a new UUID
// is generated. The log directory is determined from cfg.Logging.Directory,
// falling back to ~/.daz-agent-sdk/logs.
func NewConversationLogger(name string, cfg *Config, conversationID ...uuid.UUID) *ConversationLogger {
	var id uuid.UUID
	if len(conversationID) > 0 {
		id = conversationID[0]
	} else {
		id = uuid.New()
	}

	base := defaultLogBase()
	if cfg != nil && cfg.Logging.Directory != "" {
		dir := cfg.Logging.Directory
		// Expand ~ prefix
		if strings.HasPrefix(dir, "~/") {
			if home, err := os.UserHomeDir(); err == nil {
				dir = filepath.Join(home, dir[2:])
			}
		}
		base = dir
	}

	logDir := filepath.Join(base, id.String())

	cl := &ConversationLogger{
		conversationID: id,
		name:           name,
		logDir:         logDir,
	}

	// Create directory and write meta.json — swallow errors
	if err := os.MkdirAll(logDir, 0o755); err == nil {
		meta := map[string]any{
			"conversation_id": id.String(),
			"name":            name,
			"created_at":      nowISO(),
		}
		if data, err := json.MarshalIndent(meta, "", "  "); err == nil {
			_ = os.WriteFile(filepath.Join(logDir, "meta.json"), data, 0o644)
		}
	}

	return cl
}

// LogDir returns the path to this conversation's log directory.
func (cl *ConversationLogger) LogDir() string {
	return cl.logDir
}

// ConversationID returns the UUID for this conversation.
func (cl *ConversationLogger) ConversationID() uuid.UUID {
	return cl.conversationID
}

// LogEvent appends a single JSON line to events.jsonl.
// The timestamp is added automatically. Extra key-value pairs from the
// optional map are merged into the event.
func (cl *ConversationLogger) LogEvent(eventType string, extra ...map[string]any) {
	entry := map[string]any{
		"ts":    nowISO(),
		"event": eventType,
	}
	for _, m := range extra {
		for k, v := range m {
			entry[k] = v
		}
	}
	cl.writeJSONL(entry)
}

// Close writes a final conversation_end event and marks the logger closed.
// Subsequent calls to Close are no-ops.
func (cl *ConversationLogger) Close(extra ...map[string]any) {
	cl.mu.Lock()
	if cl.closed {
		cl.mu.Unlock()
		return
	}
	cl.closed = true
	cl.mu.Unlock()

	cl.LogEvent("conversation_end", extra...)
}

// writeJSONL appends a dict as one JSON line to events.jsonl.
// Acquires the lock and silently ignores all IO errors.
func (cl *ConversationLogger) writeJSONL(entry map[string]any) {
	data, err := json.Marshal(entry)
	if err != nil {
		return
	}
	line := append(data, '\n')

	cl.mu.Lock()
	defer cl.mu.Unlock()

	f, err := os.OpenFile(filepath.Join(cl.logDir, "events.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(line)
}

// nowISO returns the current UTC time as an ISO 8601 string.
func nowISO() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}
