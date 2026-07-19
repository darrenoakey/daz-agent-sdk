package dazagentsdk

import (
	"context"

	"github.com/google/uuid"
)

// registryProvider is a concrete in-process provider used by registry tests.
type registryProvider struct {
	name string
}

func (provider *registryProvider) Name() string { return provider.name }

func (provider *registryProvider) Available(_ context.Context) (bool, error) { return true, nil }

func (provider *registryProvider) ListModels(_ context.Context) ([]ModelInfo, error) {
	return []ModelInfo{{Provider: provider.name, ModelID: "registry-model", DisplayName: "Registry Model"}}, nil
}

func (provider *registryProvider) Complete(_ context.Context, messages []Message, model ModelInfo, _ CompleteOpts) (*Response, error) {
	text := ""
	if len(messages) != 0 {
		text = messages[len(messages)-1].Content
	}
	return &Response{Text: text, ModelUsed: model, ConversationID: uuid.New(), TurnID: uuid.New()}, nil
}

func (provider *registryProvider) Stream(context.Context, []Message, ModelInfo, StreamOpts) (<-chan StreamChunk, error) {
	chunks := make(chan StreamChunk, 1)
	chunks <- StreamChunk{Text: "registry response"}
	close(chunks)
	return chunks, nil
}
