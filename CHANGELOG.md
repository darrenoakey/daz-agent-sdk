# Changelog

## 0.2.19

- Report expected CLI image validation, execution, and recovery failures as one actionable `Error: ...` line without a Python traceback.
- Retain legacy image provider, model, step, finite-timeout, and configuration controls only as fail-closed compatibility options that direct callers to the durable Mac mini Codex image service.
