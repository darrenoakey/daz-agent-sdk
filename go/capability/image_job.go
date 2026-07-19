package capability

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"strings"
	"time"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
	"github.com/google/uuid"
)

// ImageJobOpts controls recovery of an existing durable image job.
type ImageJobOpts struct {
	Output      string
	Transparent bool
	Timeout     time.Duration
	Config      *agentsdk.Config
}

// GetImageJob returns durable IGS state without creating or changing a job.
func GetImageJob(parent context.Context, jobId string, configurations ...*agentsdk.Config) (*agentsdk.ImageJobStatus, error) {
	if err := validateOptionalImageConfig(configurations); err != nil {
		return nil, err
	}
	if strings.TrimSpace(jobId) == "" || strings.Contains(jobId, "/") {
		return nil, agentsdk.NewAgentError("a valid image job id is required", agentsdk.ErrorInvalidRequest, nil)
	}
	status, err := fetchImageServiceStatus(parent, jobId)
	if err != nil {
		return nil, err
	}
	if status.Id != "" && status.Id != jobId {
		return nil, agentsdk.NewAgentError("image service returned a mismatched job id", agentsdk.ErrorInternal, nil)
	}
	if _, err := classifyImageStatus(jobId, status); err != nil && status.Status != "failed" && status.Status != "cancelled" && status.Status != "canceled" {
		return nil, err
	}
	return publicImageJobStatus(jobId, status), nil
}

func publicImageJobStatus(jobId string, status imageServiceStatus) *agentsdk.ImageJobStatus {
	provider := status.Provider
	if provider == "" {
		provider = "codex"
	}
	provenance := map[string]any{
		"id": jobId, "status": status.Status, "attempts": status.Attempts,
		"error": status.Error, "provider": provider, "prompt_version": status.PromptVersion,
		"attempt_history": status.AttemptHistory, "created_at": status.CreatedAt, "updated_at": status.UpdatedAt,
	}
	return &agentsdk.ImageJobStatus{
		JobID: jobId, Status: status.Status, Ready: status.Status == "done", ModelUsed: codexModelInfo,
		Provider: provider, Attempts: status.Attempts, Error: status.Error, PromptVersion: status.PromptVersion,
		AttemptHistory: status.AttemptHistory, CreatedAt: status.CreatedAt, UpdatedAt: status.UpdatedAt, Provenance: provenance,
	}
}

// DownloadImageJob downloads and validates one completed IGS artifact.
func DownloadImageJob(parent context.Context, jobId, output string, transparent bool, configurations ...*agentsdk.Config) (*agentsdk.ImageResult, error) {
	if err := validateOptionalImageConfig(configurations); err != nil {
		return nil, err
	}
	status, err := GetImageJob(parent, jobId)
	if err != nil {
		return nil, err
	}
	if status.Status != "done" {
		if status.Status == "failed" || status.Status == "cancelled" || status.Status == "canceled" {
			attempts := []map[string]any{{"job_id": jobId, "status": status.Status, "recoverable": false}}
			return nil, agentsdk.NewAgentError(fmt.Sprintf("image service job %s ended with status %s: %s", jobId, status.Status, status.Error), agentsdk.ErrorInternal, attempts)
		}
		return nil, agentsdk.NewAgentError(fmt.Sprintf("image service job %s is %s, not done", jobId, status.Status), agentsdk.ErrorInvalidRequest, nil)
	}
	resolved, err := resolveOutputPath(output)
	if err != nil {
		return nil, err
	}
	data, err := fetchImageServiceImage(parent, jobId)
	if err != nil {
		return nil, err
	}
	if err := writeServiceImage(data, resolved, transparent); err != nil {
		return nil, err
	}
	configuration, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("reading validated image dimensions: %w", err)
	}
	return &agentsdk.ImageResult{
		Path: resolved, ModelUsed: codexModelInfo, ConversationID: uuid.New(), Width: configuration.Width,
		Height: configuration.Height, JobID: jobId, Status: "done", Ready: true, Provider: status.Provider,
		Provenance: status.Provenance,
	}, nil
}

// WaitImageJob waits through transient service failures and downloads only a terminal artifact.
func WaitImageJob(parent context.Context, jobId string, options ImageJobOpts) (*agentsdk.ImageResult, error) {
	return ResumeImageJob(parent, jobId, options)
}

// ResumeImageJob waits for an existing IGS id and never submits a replacement.
func ResumeImageJob(parent context.Context, jobId string, options ImageJobOpts) (*agentsdk.ImageResult, error) {
	if err := options.Config.ValidateImageConfig(); err != nil {
		return nil, err
	}
	var deadline <-chan time.Time
	if options.Timeout > 0 {
		timer := time.NewTimer(options.Timeout)
		defer timer.Stop()
		deadline = timer.C
	}
	last := &agentsdk.ImageJobStatus{JobID: jobId, Status: "unknown", Provider: "codex"}
	for {
		status, err := GetImageJob(parent, jobId)
		if err != nil {
			if !isTransientImageError(err) {
				return nil, err
			}
			select {
			case <-deadline:
				return pendingImageResult(options.Output, last), nil
			case <-parent.Done():
				return pendingImageResult(options.Output, last), nil
			case <-time.After(imagePollInterval):
				continue
			}
		}
		last = status
		if status.Status == "done" {
			result, downloadError := DownloadImageJob(parent, jobId, options.Output, options.Transparent)
			if downloadError == nil {
				return result, nil
			}
			if !isTransientImageError(downloadError) {
				return nil, downloadError
			}
		}
		if status.Status == "failed" || status.Status == "cancelled" || status.Status == "canceled" {
			attempts := []map[string]any{{"job_id": jobId, "status": status.Status, "recoverable": false}}
			return nil, agentsdk.NewAgentError(fmt.Sprintf("image service job %s ended with status %s: %s", jobId, status.Status, status.Error), agentsdk.ErrorInternal, attempts)
		}
		select {
		case <-deadline:
			return pendingImageResult(options.Output, status), nil
		case <-parent.Done():
			return pendingImageResult(options.Output, status), nil
		case <-time.After(imagePollInterval):
		}
	}
}

func validateOptionalImageConfig(configurations []*agentsdk.Config) error {
	for _, configuration := range configurations {
		if err := configuration.ValidateImageConfig(); err != nil {
			return err
		}
	}
	return nil
}

func isTransientImageError(err error) bool {
	var agentError *agentsdk.AgentError
	return errors.As(err, &agentError) && agentError.Kind == agentsdk.ErrorNotAvailable
}

func pendingImageResult(output string, status *agentsdk.ImageJobStatus) *agentsdk.ImageResult {
	return &agentsdk.ImageResult{
		Path: output, ModelUsed: codexModelInfo, ConversationID: uuid.New(), JobID: status.JobID,
		Status: status.Status, Provider: status.Provider, Provenance: status.Provenance,
	}
}
