package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunImageRejectsLegacyProvidersBeforeEveryDurableBranch(t *testing.T) {
	providers := []string{"flux", "z-image-turbo", "ollama", "gemini", "spark"}
	branches := []string{"default", "state", "recover", "idempotency-key"}
	for _, provider := range providers {
		for _, branch := range branches {
			t.Run(provider+"/"+branch, func(t *testing.T) {
				path := filepath.Join(t.TempDir(), branch+".json")
				arguments := []string{"--provider", provider}
				if branch == "recover" {
					arguments = append(arguments, "--recover", path)
				} else {
					arguments = append(arguments, "--prompt", "route proof", "--width", "64", "--height", "64")
					if branch != "default" {
						value := path
						if branch == "idempotency-key" {
							value = "durable-key"
						}
						arguments = append(arguments, "--"+branch, value)
					}
				}
				if code := runImage(arguments); code == 0 {
					t.Fatal("legacy provider entered durable image branch")
				}
				if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("durable branch touched %s: %v", path, err)
				}
			})
		}
	}
}

func TestRunImageDefaultUsesCapabilityDurability(t *testing.T) {
	source, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	if !strings.Contains(text, "capability.GenerateImage(ctx, *prompt, opts)") {
		t.Fatal("default CLI image path bypasses the durable capability")
	}
	if strings.Contains(text, "if *statePath !=") {
		t.Fatal("CLI durability remains conditional on an optional state flag")
	}
}
