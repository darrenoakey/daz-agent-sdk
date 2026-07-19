package provider

import (
	"context"
	"testing"
)

func TestReadCredentialMissingReturnsError(t *testing.T) {
	_, err := readCredential(context.Background(), credentialReference{
		service: "daz-agent-sdk-missing-service",
		account: "daz-agent-sdk-missing-account",
	})
	if err == nil {
		t.Fatal("readCredential should fail closed for a missing credential")
	}
}
