package provider

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

type credentialReference struct {
	service string
	account string
}

func readCredential(ctx context.Context, references ...credentialReference) (string, error) {
	var failures []string
	for _, reference := range references {
		value, err := readCredentialEntry(ctx, reference)
		if err == nil {
			return value, nil
		}
		failures = append(failures, err.Error())
	}
	return "", fmt.Errorf("credential unavailable: %s", strings.Join(failures, "; "))
}

func readCredentialEntry(ctx context.Context, reference credentialReference) (string, error) {
	command := exec.CommandContext(ctx, "/usr/bin/security", "find-generic-password",
		"-s", reference.service, "-a", reference.account, "-w")
	output, err := command.Output()
	if err != nil {
		return "", fmt.Errorf("%s/%s: %w", reference.service, reference.account, err)
	}
	value := strings.TrimSpace(string(output))
	if value == "" {
		return "", fmt.Errorf("%s/%s: empty credential", reference.service, reference.account)
	}
	return value, nil
}
