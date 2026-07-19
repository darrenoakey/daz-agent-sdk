package capability

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

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

func TestResumeImageJobRejectsTimeoutBeforeNetwork(t *testing.T) {
	result, err := ResumeImageJob(context.Background(), "durable-job", ImageJobOpts{Timeout: time.Second})
	var agentError *agentsdk.AgentError
	if result != nil || !errors.As(err, &agentError) || agentError.Kind != agentsdk.ErrorInvalidRequest {
		t.Fatalf("finite timeout result=%+v error=%v", result, err)
	}
	if !strings.Contains(err.Error(), "deadlines are actively disabled") {
		t.Fatalf("finite timeout guidance = %v", err)
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
