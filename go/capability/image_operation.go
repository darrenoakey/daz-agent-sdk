package capability

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	agentsdk "github.com/darrenoakey/daz-agent-sdk/go"
	"github.com/google/uuid"
	"golang.org/x/sys/unix"
)

const imageOperationStateVersion = 2
const imageOperationNamespace = "daz-agent-sdk:image-operation:"
const maxImageOperationStateBytes = 1 << 20

type imageOperationState struct {
	Version        int    `json:"version"`
	OperationId    string `json:"operation_id"`
	IdempotencyKey string `json:"idempotency_key"`
	RequestBody    string `json:"request_body"`
	OutputIntent   string `json:"output_intent"`
	OutputPath     string `json:"output_path"`
	OutputFormat   string `json:"output_format"`
	Transparent    bool   `json:"transparent"`
	JobId          string `json:"job_id"`
}

func imageOperationRoot() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolving image operation home: %w", err)
	}
	root := filepath.Join(home, ".daz-agent-sdk", "image-operations")
	directory, err := openImageOperationDirectory(home)
	if err != nil {
		return "", fmt.Errorf("opening image operation home: %w", err)
	}
	for index, component := range []string{".daz-agent-sdk", "image-operations"} {
		if mkdirError := unix.Mkdirat(int(directory.Fd()), component, 0o700); mkdirError != nil && !errors.Is(mkdirError, syscall.EEXIST) {
			_ = directory.Close()
			return "", fmt.Errorf("creating image operation registry: %w", mkdirError)
		}
		descriptor, openError := unix.Openat(int(directory.Fd()), component, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW, 0)
		_ = directory.Close()
		if openError != nil {
			return "", fmt.Errorf("opening image operation registry: %w", openError)
		}
		directory = os.NewFile(uintptr(descriptor), component)
		details, statError := directory.Stat()
		if statError != nil {
			_ = directory.Close()
			return "", fmt.Errorf("inspecting image operation registry: %w", statError)
		}
		statData, ok := details.Sys().(*syscall.Stat_t)
		if !ok || statData.Uid != uint32(os.Geteuid()) || details.Mode().Perm()&0o022 != 0 || (index == 1 && details.Mode().Perm() != 0o700) {
			_ = directory.Close()
			return "", fmt.Errorf("image operation registry is not private: %s", root)
		}
	}
	_ = directory.Close()
	if err := validateImageOperationDirectory(root); err != nil {
		return "", err
	}
	return root, nil
}

func validateImageOperationDirectory(path string) error {
	directory, err := openImageOperationDirectory(path)
	if err != nil {
		return fmt.Errorf("inspecting image operation directory: %w", err)
	}
	defer directory.Close()
	details, err := directory.Stat()
	if err != nil {
		return fmt.Errorf("inspecting image operation directory: %w", err)
	}
	statData, ok := details.Sys().(*syscall.Stat_t)
	if !details.IsDir() || !ok || statData.Uid != uint32(os.Geteuid()) || details.Mode().Perm() != 0o700 {
		return fmt.Errorf("image operation directory must be a current-user owner-only directory: %s", path)
	}
	return nil
}

func openImageOperationDirectory(path string) (*os.File, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolving image operation directory: %w", err)
	}
	if absolute == "/var" || strings.HasPrefix(absolute, "/var/") {
		absolute = "/private" + absolute
	}
	descriptor, err := unix.Open(string(filepath.Separator), unix.O_RDONLY|unix.O_DIRECTORY, 0)
	if err != nil {
		return nil, err
	}
	for _, component := range strings.Split(strings.TrimPrefix(absolute, string(filepath.Separator)), string(filepath.Separator)) {
		if component == "" {
			continue
		}
		next, openError := unix.Openat(descriptor, component, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW, 0)
		_ = unix.Close(descriptor)
		if openError != nil {
			return nil, openError
		}
		descriptor = next
	}
	return os.NewFile(uintptr(descriptor), absolute), nil
}

func openOrCreateImageOperationDirectory(path string) (*os.File, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolving image operation directory: %w", err)
	}
	if absolute == "/var" || strings.HasPrefix(absolute, "/var/") {
		absolute = "/private" + absolute
	}
	descriptor, err := unix.Open(string(filepath.Separator), unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	for _, component := range strings.Split(strings.TrimPrefix(absolute, string(filepath.Separator)), string(filepath.Separator)) {
		if component == "" {
			continue
		}
		next, openError := unix.Openat(descriptor, component, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW, 0)
		if errors.Is(openError, syscall.ENOENT) {
			openError = unix.Mkdirat(descriptor, component, 0o700)
			if openError == nil || errors.Is(openError, syscall.EEXIST) {
				next, openError = unix.Openat(descriptor, component, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW, 0)
			}
		}
		_ = unix.Close(descriptor)
		if openError != nil {
			return nil, openError
		}
		descriptor = next
	}
	return os.NewFile(uintptr(descriptor), absolute), nil
}

func rejectSymlinkComponents(path string) error {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("resolving path: %w", err)
	}
	current := string(filepath.Separator)
	for _, component := range strings.Split(strings.TrimPrefix(absolute, string(filepath.Separator)), string(filepath.Separator)) {
		if component == "" {
			continue
		}
		current = filepath.Join(current, component)
		details, statError := os.Lstat(current)
		if statError != nil {
			return fmt.Errorf("inspecting path component %s: %w", current, statError)
		}
		if details.Mode()&os.ModeSymlink != 0 {
			// /var is the stable macOS system alias for /private/var; rejecting it
			// would make every OS temporary directory unusable. Caller-controlled
			// symlink components remain forbidden.
			if current == "/var" {
				continue
			}
			return fmt.Errorf("symlink path component is forbidden: %s", current)
		}
	}
	return nil
}

func imageOutputIntent(output string) (string, error) {
	if output == "" {
		return "automatic:png", nil
	}
	resolved, err := filepath.Abs(output)
	if err != nil {
		return "", fmt.Errorf("resolving image output intent: %w", err)
	}
	return "path:" + resolved, nil
}

func imageOperationId(body []byte, outputIntent, key string) string {
	identity := append([]byte("request\x00"), body...)
	identity = append(identity, []byte("\x00output\x00"+outputIntent)...)
	if key != "" {
		identity = []byte("idempotency-key\x00" + key)
	}
	digest := sha256.Sum256(identity)
	return hex.EncodeToString(digest[:])
}

func imageOperationKey(operationId, requested string) string {
	if requested != "" {
		return requested
	}
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte(imageOperationNamespace+operationId)).String()
}

func imageOperationPaths(operationId, output, selectedState string) (string, string, error) {
	root, err := imageOperationRoot()
	if err != nil {
		return "", "", err
	}
	statePath := selectedState
	if statePath == "" {
		statePath = filepath.Join(root, operationId+".json")
	} else if statePath, err = filepath.Abs(statePath); err != nil {
		return "", "", fmt.Errorf("resolving image operation state: %w", err)
	}
	outputPath := output
	if outputPath == "" {
		artifactRoot := filepath.Join(root, "artifacts")
		directory, createError := openOrCreateImageOperationDirectory(artifactRoot)
		if createError != nil {
			return "", "", fmt.Errorf("creating image artifact registry: %w", createError)
		}
		_ = directory.Close()
		if err := validateImageOperationDirectory(artifactRoot); err != nil {
			return "", "", fmt.Errorf("creating image artifact registry: %w", err)
		}
		outputPath = filepath.Join(artifactRoot, operationId+".png")
	} else if outputPath, err = filepath.Abs(outputPath); err != nil {
		return "", "", fmt.Errorf("resolving image output: %w", err)
	}
	return statePath, outputPath, nil
}

func imageOperationOutputFormat(outputPath string) string {
	extension := strings.ToLower(filepath.Ext(outputPath))
	if extension == ".jpg" || extension == ".jpeg" {
		return "jpeg"
	}
	return "png"
}

func prepareImageOperation(body []byte, options ImageOpts) (string, imageOperationState, error) {
	intent, err := imageOutputIntent(options.Output)
	if err != nil {
		return "", imageOperationState{}, err
	}
	operationId := imageOperationId(body, intent, strings.TrimSpace(options.IdempotencyKey))
	statePath, outputPath, err := imageOperationPaths(operationId, options.Output, options.StatePath)
	if err != nil {
		return "", imageOperationState{}, err
	}
	state := imageOperationState{
		Version: imageOperationStateVersion, OperationId: operationId,
		IdempotencyKey: imageOperationKey(operationId, strings.TrimSpace(options.IdempotencyKey)),
		RequestBody:    string(body), OutputIntent: intent, OutputPath: outputPath,
		OutputFormat: imageOperationOutputFormat(outputPath), Transparent: options.Transparent,
	}
	if _, err := os.Lstat(statePath); err == nil {
		persisted, readError := readImageOperation(statePath)
		if readError != nil {
			return "", imageOperationState{}, readError
		}
		if persisted.Version != state.Version || persisted.OperationId != state.OperationId || persisted.RequestBody != state.RequestBody || persisted.OutputIntent != state.OutputIntent {
			return "", imageOperationState{}, agentsdk.NewAgentError("image operation identity conflicts with immutable state "+statePath, agentsdk.ErrorInvalidRequest, nil)
		}
		return statePath, persisted, nil
	} else if !os.IsNotExist(err) {
		return "", imageOperationState{}, fmt.Errorf("inspecting image operation state: %w", err)
	}
	return statePath, state, writeImageOperation(statePath, state)
}

func writeImageOperation(path string, state imageOperationState) error {
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("encoding image operation state: %w", err)
	}
	createdDirectory, err := openOrCreateImageOperationDirectory(filepath.Dir(path))
	if err != nil {
		return fmt.Errorf("creating image operation directory: %w", err)
	}
	_ = createdDirectory.Close()
	if err := validateImageOperationDirectory(filepath.Dir(path)); err != nil {
		return err
	}
	if details, err := os.Lstat(path); err == nil && !details.Mode().IsRegular() {
		return fmt.Errorf("image operation state target is unsafe: %s", path)
	} else if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("inspecting image operation state target: %w", err)
	}
	directory, err := openImageOperationDirectory(filepath.Dir(path))
	if err != nil {
		return fmt.Errorf("opening image operation directory: %w", err)
	}
	defer directory.Close()
	temporaryName := ".image-operation-" + uuid.NewString()
	descriptor, err := unix.Openat(int(directory.Fd()), temporaryName, unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW, 0o600)
	if err != nil {
		return fmt.Errorf("creating image operation state: %w", err)
	}
	temporary := os.NewFile(uintptr(descriptor), temporaryName)
	defer func() { _ = unix.Unlinkat(int(directory.Fd()), temporaryName, 0) }()
	if _, err := temporary.Write(data); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("writing image operation state: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("syncing image operation state: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("closing image operation state: %w", err)
	}
	if err := unix.Renameat(int(directory.Fd()), temporaryName, int(directory.Fd()), filepath.Base(path)); err != nil {
		return fmt.Errorf("replacing image operation state: %w", err)
	}
	return directory.Sync()
}

func readImageOperation(path string) (imageOperationState, error) {
	if err := validateImageOperationDirectory(filepath.Dir(path)); err != nil {
		return imageOperationState{}, err
	}
	directory, err := openImageOperationDirectory(filepath.Dir(path))
	if err != nil {
		return imageOperationState{}, fmt.Errorf("opening image operation directory: %w", err)
	}
	defer directory.Close()
	descriptor, err := unix.Openat(int(directory.Fd()), filepath.Base(path), unix.O_RDONLY|unix.O_NOFOLLOW, 0)
	if err != nil {
		return imageOperationState{}, fmt.Errorf("opening image operation state: %w", err)
	}
	file := os.NewFile(uintptr(descriptor), path)
	defer file.Close()
	details, err := file.Stat()
	if err != nil {
		return imageOperationState{}, fmt.Errorf("inspecting image operation state: %w", err)
	}
	statData, ok := details.Sys().(*syscall.Stat_t)
	if !details.Mode().IsRegular() || !ok || statData.Uid != uint32(os.Geteuid()) || details.Mode().Perm() != 0o600 || details.Size() > maxImageOperationStateBytes {
		return imageOperationState{}, fmt.Errorf("image operation state is not a private current-user regular file: %s", path)
	}
	var state imageOperationState
	decoder := json.NewDecoder(io.LimitReader(file, maxImageOperationStateBytes+1))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&state); err != nil {
		return imageOperationState{}, fmt.Errorf("decoding image operation state: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return imageOperationState{}, err
	}
	if err := validateImageOperationState(state, path); err != nil {
		return imageOperationState{}, err
	}
	return state, nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("image operation state contains trailing data")
	}
	return nil
}

func validateImageOperationState(state imageOperationState, path string) error {
	if state.Version != imageOperationStateVersion || state.OperationId == "" || state.IdempotencyKey == "" || state.RequestBody == "" || state.OutputIntent == "" || state.OutputPath == "" {
		return fmt.Errorf("image operation state has invalid schema: %s", path)
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(state.RequestBody), &body); err != nil || body == nil {
		return fmt.Errorf("image operation request body is invalid: %s", path)
	}
	if !filepath.IsAbs(state.OutputPath) || state.OutputFormat != imageOperationOutputFormat(state.OutputPath) {
		return fmt.Errorf("image operation output metadata is invalid: %s", path)
	}
	if state.OutputIntent != "automatic:png" && state.OutputIntent != "path:"+state.OutputPath {
		return fmt.Errorf("image operation output intent conflicts: %s", path)
	}
	keyed := imageOperationId([]byte(state.RequestBody), state.OutputIntent, state.IdempotencyKey)
	unkeyed := imageOperationId([]byte(state.RequestBody), state.OutputIntent, "")
	if state.OperationId != keyed && state.OperationId != unkeyed {
		return fmt.Errorf("image operation immutable identity is invalid: %s", path)
	}
	return nil
}

func lockImageOperation(path string) (*os.File, error) {
	createdDirectory, err := openOrCreateImageOperationDirectory(filepath.Dir(path))
	if err != nil {
		return nil, fmt.Errorf("creating image operation directory: %w", err)
	}
	_ = createdDirectory.Close()
	if err := validateImageOperationDirectory(filepath.Dir(path)); err != nil {
		return nil, err
	}
	lockPath := path + ".lock"
	directory, err := openImageOperationDirectory(filepath.Dir(lockPath))
	if err != nil {
		return nil, fmt.Errorf("opening image operation directory: %w", err)
	}
	defer directory.Close()
	descriptor, err := unix.Openat(int(directory.Fd()), filepath.Base(lockPath), unix.O_CREAT|unix.O_RDWR|unix.O_NOFOLLOW, 0o600)
	if err != nil {
		return nil, fmt.Errorf("opening image operation lock: %w", err)
	}
	file := os.NewFile(uintptr(descriptor), lockPath)
	details, statError := file.Stat()
	if statError != nil || !details.Mode().IsRegular() || details.Mode().Perm() != 0o600 {
		_ = file.Close()
		return nil, fmt.Errorf("image operation lock is not private: %s", lockPath)
	}
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("locking image operation: %w", err)
	}
	return file, nil
}

func unlockImageOperation(file *os.File) {
	_ = syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
	_ = file.Close()
}

func completeImageOperation(parent context.Context, prompt string, options ImageOpts, body []byte) (*agentsdk.ImageResult, error) {
	statePath, _, err := prepareImageOperation(body, options)
	if err != nil {
		return nil, err
	}
	lock, err := lockImageOperation(statePath)
	if err != nil {
		return nil, err
	}
	defer unlockImageOperation(lock)
	_, state, err := prepareImageOperation(body, options)
	if err != nil {
		return nil, err
	}
	replayed := false
	if state.JobId == "" {
		submission, submitError := WaitImageSubmission(parent, []byte(state.RequestBody), state.IdempotencyKey, options.Config)
		if submitError != nil {
			if options.Timeout > 0 && parent.Err() != nil {
				return pendingImageOperationResult(statePath, state), nil
			}
			return nil, submitError
		}
		state.JobId = submission.JobID
		replayed = submission.Replayed
		if err := writeImageOperation(statePath, state); err != nil {
			return nil, err
		}
	}
	result, err := ResumeImageJob(parent, state.JobId, ImageJobOpts{Output: state.OutputPath, Transparent: state.Transparent})
	if result != nil {
		result.Prompt = prompt
		result.Width = options.Width
		result.Height = options.Height
		result.ConversationID = options.ConversationID
		result.IdempotencyKey = state.IdempotencyKey
		result.Replayed = replayed
	}
	return result, err
}

func pendingImageOperationResult(statePath string, state imageOperationState) *agentsdk.ImageResult {
	status := "submitting"
	if state.JobId != "" {
		status = "accepted"
	}
	return &agentsdk.ImageResult{
		Path: state.OutputPath, ModelUsed: codexModelInfo, ConversationID: uuid.New(), JobID: state.JobId,
		Status: status, Provider: "codex", IdempotencyKey: state.IdempotencyKey,
		Provenance: map[string]any{
			"operation_id": state.OperationId, "operation_state": statePath, "recoverable": true,
		},
	}
}

// ResumeImageOperation resumes the immutable request recorded at statePath.
func ResumeImageOperation(parent context.Context, statePath string, configurations ...*agentsdk.Config) (*agentsdk.ImageResult, error) {
	if err := validateOptionalImageConfig(configurations); err != nil {
		return nil, err
	}
	resolved, err := filepath.Abs(statePath)
	if err != nil {
		return nil, fmt.Errorf("resolving image operation state: %w", err)
	}
	lock, err := lockImageOperation(resolved)
	if err != nil {
		return nil, err
	}
	defer unlockImageOperation(lock)
	state, err := readImageOperation(resolved)
	if err != nil {
		return nil, err
	}
	if state.Version != imageOperationStateVersion {
		return nil, agentsdk.NewAgentError("unsupported image operation state version", agentsdk.ErrorInvalidRequest, nil)
	}
	if state.JobId == "" {
		submission, submitError := WaitImageSubmission(parent, []byte(state.RequestBody), state.IdempotencyKey)
		if submitError != nil {
			return nil, submitError
		}
		state.JobId = submission.JobID
		if err := writeImageOperation(resolved, state); err != nil {
			return nil, err
		}
	}
	result, err := WaitImageJob(parent, state.JobId, ImageJobOpts{Output: state.OutputPath, Transparent: state.Transparent})
	if result != nil {
		result.IdempotencyKey = state.IdempotencyKey
	}
	return result, err
}
