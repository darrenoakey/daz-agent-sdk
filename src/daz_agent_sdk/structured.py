"""Structured output via file-based schema extraction.

When a caller passes schema= to agent.ask(), the SDK:
1. Generates a unique temp filename
2. Appends instructions to write JSON to that file
3. After the provider completes:
   - If the file exists (agentic providers wrote it): reads from file
   - If the file doesn't exist: extracts JSON from response text and writes it
4. Validates against the pydantic schema
5. Cleans up the temp file
6. Returns StructuredResponse with .parsed
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from daz_agent_sdk.types import parse_json_from_llm

_logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def schema_filename() -> str:
    """Generate a unique filename for structured output."""
    return f"_structured_output_{uuid4().hex[:12]}.json"


def schema_instructions(schema_cls: Type[BaseModel], filename: str) -> str:
    """Build the prompt suffix that tells the AI to write its structured output to a file."""
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    return (
        f"\n\n## REQUIRED: Structured Output\n"
        f"You MUST produce valid JSON matching this exact schema:\n"
        f"```json\n{schema_json}\n```\n"
        f"Write the raw JSON to `{filename}` (no markdown fences, no extra text in the file).\n"
        f"ALSO, your final response text MUST end with the complete JSON object — "
        f"after any explanation, include a line containing ONLY the raw JSON. "
        f"This is critical — the JSON must appear at the end of your response.\n"
    )


def ensure_cwd(cwd: str | Path | None) -> tuple[str, bool]:
    """Ensure we have a working directory. Returns (cwd_path, created_temp).

    If cwd is already set, returns it unchanged.
    If cwd is None, creates a temp directory and returns it.
    """
    if cwd is not None:
        return str(cwd), False
    tmp = tempfile.mkdtemp(prefix="agent-structured-")
    return tmp, True


def extract_result(
    schema_cls: Type[T],
    filename: str,
    cwd: str,
    response_text: str,
) -> T:
    """Extract and validate structured output.

    1. Check if the file was written by the AI
    2. If not, try to parse from response text and write the file
    3. Validate against schema
    4. Clean up the file
    """
    filepath = os.path.join(cwd, filename)
    _logger.info("structured: looking for %s in %s", filename, cwd)
    if os.path.isdir(cwd):
        contents = os.listdir(cwd)
        _logger.info("structured: cwd contents (%d files): %s", len(contents), contents[:20])
    else:
        _logger.warning("structured: cwd does not exist: %s", cwd)

    json_text: str | None = None

    # Try reading from file first (agentic providers write it directly)
    if os.path.exists(filepath):
        _logger.info("structured: reading output from %s", filename)
        with open(filepath) as f:
            json_text = f.read().strip()

    # Fall back to response text (non-agentic providers return it as text)
    if not json_text:
        if not response_text or not response_text.strip():
            raise RuntimeError(
                f"Structured output failed: file {filename} not written and response text is empty"
            )
        _logger.info("structured: extracting from response text (%d chars)", len(response_text))
        json_text = response_text.strip()

    # Parse and validate
    try:
        parsed_json = parse_json_from_llm(json_text)
        result = schema_cls.model_validate(parsed_json)
    except Exception as e:
        # Log first 500 chars for debugging
        preview = json_text[:500] if json_text else "(empty)"
        _logger.error("structured: parse failed — preview: %s", preview)
        raise RuntimeError(f"Structured output parse failed: {e}") from e
    finally:
        # Clean up the file
        if os.path.exists(filepath):
            os.unlink(filepath)

    return result
