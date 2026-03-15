package dazagentsdk

import (
	"encoding/json"
	"fmt"
)

// ExtractStructured parses JSON from a Response's text into the target struct.
// It uses ParseJSONFromLLM to handle markdown fences and prose-wrapped JSON.
func ExtractStructured(response *Response, target any) error {
	if response == nil {
		return fmt.Errorf("response is nil")
	}
	parsed, err := ParseJSONFromLLM(response.Text)
	if err != nil {
		return fmt.Errorf("extracting structured output: %w", err)
	}
	// Re-marshal the map and unmarshal into the target struct
	data, err := json.Marshal(parsed)
	if err != nil {
		return fmt.Errorf("re-marshaling parsed JSON: %w", err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("unmarshaling into target: %w", err)
	}
	return nil
}

// SchemaInstructions builds a system message suffix that instructs the AI
// to produce JSON matching the given schema. The schemaJSON parameter should
// be a valid JSON schema string.
func SchemaInstructions(schemaJSON string) string {
	return "\n\n## REQUIRED: Structured Output\n" +
		"You MUST produce valid JSON matching this exact schema:\n" +
		"```json\n" + schemaJSON + "\n```\n" +
		"Your response MUST end with the complete JSON object -- " +
		"after any explanation, include a line containing ONLY the raw JSON. " +
		"This is critical -- the JSON must appear at the end of your response.\n"
}
