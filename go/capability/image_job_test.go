package capability

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
)

func TestGetImageJobRejectsInvalidIdBeforeNetwork(t *testing.T) {
	for _, jobId := range []string{"", " ", "../alternate", "nested/job"} {
		status, err := GetImageJob(context.Background(), jobId)
		if status != nil || err == nil || !strings.Contains(err.Error(), "valid image job id") {
			t.Errorf("job id %q returned status=%+v error=%v", jobId, status, err)
		}
	}
}

func TestPendingImageResultDoesNotTrustPreexistingOutput(t *testing.T) {
	output := filepath.Join(t.TempDir(), "existing.png")
	if err := os.WriteFile(output, []byte("not a completed job artifact"), 0o600); err != nil {
		t.Fatal(err)
	}
	result := pendingImageResult(output, &agentsdk.ImageJobStatus{
		JobID: "durable-job", Status: "running", Provider: "codex",
	})
	if result.Ready || result.Status != "running" || result.Path != output {
		t.Fatalf("pending result trusted preexisting output: %+v", result)
	}
}

func TestTransientImageErrorsAreRecoverableWithoutDefaultDeadline(t *testing.T) {
	transient := agentsdk.NewAgentError("temporary status failure", agentsdk.ErrorNotAvailable, nil)
	terminal := agentsdk.NewAgentError("terminal status failure", agentsdk.ErrorInternal, nil)
	if !isTransientImageError(transient) || isTransientImageError(terminal) {
		t.Fatal("transient image error classification is incorrect")
	}
	if errors.Is(transient, context.DeadlineExceeded) {
		t.Fatal("default recovery introduced a finite deadline")
	}
}

func TestTerminalImageStatusPreservesIdentityAndNeverRecovers(t *testing.T) {
	terminal, err := classifyImageStatus("terminal-job", imageServiceStatus{Status: "failed", Error: "service failure"})
	if !terminal || err == nil {
		t.Fatalf("terminal status result: terminal=%v error=%v", terminal, err)
	}
	var agentError *agentsdk.AgentError
	if !errors.As(err, &agentError) || len(agentError.Attempts) != 1 {
		t.Fatalf("terminal status identity metadata missing: %v", err)
	}
	if agentError.Attempts[0]["job_id"] != "terminal-job" || agentError.Attempts[0]["recoverable"] != false {
		t.Fatalf("terminal status metadata = %+v", agentError.Attempts)
	}
}
