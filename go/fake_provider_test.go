package dazagentsdk

import "context"

// fakeProvider is a test-only Provider implementation used by registry tests.
type fakeProvider struct {
	name string
}

func (f *fakeProvider) Name() string { return f.name }

func (f *fakeProvider) Available(_ context.Context) (bool, error) { return true, nil }

func (f *fakeProvider) ListModels(_ context.Context) ([]ModelInfo, error) { return nil, nil }

func (f *fakeProvider) Complete(_ context.Context, _ []Message, _ ModelInfo, _ CompleteOpts) (*Response, error) {
	return nil, nil
}

func (f *fakeProvider) Stream(_ context.Context, _ []Message, _ ModelInfo, _ StreamOpts) (<-chan StreamChunk, error) {
	return nil, nil
}
