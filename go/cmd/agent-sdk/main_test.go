package main

import (
	"os"
	"strings"
	"testing"
)

func TestRunImageUsesOnlyCanonicalCapabilityDefaults(t *testing.T) {
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
	for _, unsafe := range []string{"ImageFn", "ImageCallOpts", "fs.String(\"provider\"", "Provider:"} {
		if strings.Contains(text, unsafe) {
			t.Fatalf("CLI exposes unsafe image override %q", unsafe)
		}
	}
}
