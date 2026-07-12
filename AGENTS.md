<claude-mem-context>
# Memory Context

# [daz-agent-sdk] recent context, 2026-05-22 6:22am GMT+10

No previous sessions found.
</claude-mem-context>

## Image generation architecture

- All SDK image generation goes through the single Mac mini `image_generation_service` on port 8830. Do not add provider-specific CLI/model implementations back into the SDK; callers receive one stable capability while the service owns the backing image model and serialization.
- A release is not verified by a successful PyPI upload alone. Exercise `generate_image` against the real service and validate the resulting image dimensions/format before publishing or pinning the version downstream.
