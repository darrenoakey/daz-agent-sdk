package dazagentsdk

import (
	"strings"
	"testing"

	"github.com/google/uuid"
)

func TestExtractStructured_PlainJSON(t *testing.T) {
	resp := &Response{
		Text:           `{"name": "Alice", "age": 30}`,
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
	}

	type Person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}

	var person Person
	err := ExtractStructured(resp, &person)
	if err != nil {
		t.Fatalf("ExtractStructured() error: %v", err)
	}
	if person.Name != "Alice" {
		t.Errorf("Name = %q, want %q", person.Name, "Alice")
	}
	if person.Age != 30 {
		t.Errorf("Age = %d, want %d", person.Age, 30)
	}
}

func TestExtractStructured_MarkdownFenced(t *testing.T) {
	resp := &Response{
		Text: "Here is the result:\n```json\n{\"color\": \"blue\", \"count\": 5}\n```",
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
	}

	type Item struct {
		Color string `json:"color"`
		Count int    `json:"count"`
	}

	var item Item
	err := ExtractStructured(resp, &item)
	if err != nil {
		t.Fatalf("ExtractStructured() error: %v", err)
	}
	if item.Color != "blue" {
		t.Errorf("Color = %q, want %q", item.Color, "blue")
	}
	if item.Count != 5 {
		t.Errorf("Count = %d, want %d", item.Count, 5)
	}
}

func TestExtractStructured_ProseWithJSON(t *testing.T) {
	resp := &Response{
		Text: "Sure! Here is the data you requested.\n\n{\"title\": \"Go Programming\", \"pages\": 350}",
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
	}

	type Book struct {
		Title string `json:"title"`
		Pages int    `json:"pages"`
	}

	var book Book
	err := ExtractStructured(resp, &book)
	if err != nil {
		t.Fatalf("ExtractStructured() error: %v", err)
	}
	if book.Title != "Go Programming" {
		t.Errorf("Title = %q, want %q", book.Title, "Go Programming")
	}
	if book.Pages != 350 {
		t.Errorf("Pages = %d, want %d", book.Pages, 350)
	}
}

func TestExtractStructured_NilResponse(t *testing.T) {
	var person struct {
		Name string `json:"name"`
	}
	err := ExtractStructured(nil, &person)
	if err == nil {
		t.Fatal("ExtractStructured(nil) should return error")
	}
}

func TestExtractStructured_NoJSON(t *testing.T) {
	resp := &Response{
		Text:           "This is just plain text with no JSON",
		ConversationID: uuid.New(),
		TurnID:         uuid.New(),
	}

	var target struct {
		Name string `json:"name"`
	}
	err := ExtractStructured(resp, &target)
	if err == nil {
		t.Fatal("ExtractStructured() should return error for non-JSON text")
	}
}

func TestSchemaInstructions_ContainsSchema(t *testing.T) {
	schema := `{"type": "object", "properties": {"name": {"type": "string"}}}`
	result := SchemaInstructions(schema)

	if !strings.Contains(result, schema) {
		t.Error("SchemaInstructions() should contain the schema")
	}
	if !strings.Contains(result, "REQUIRED: Structured Output") {
		t.Error("SchemaInstructions() should contain the header")
	}
	if !strings.Contains(result, "```json") {
		t.Error("SchemaInstructions() should contain json code fence")
	}
}
