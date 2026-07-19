from __future__ import annotations

import argparse
import asyncio
from typing import Any

from daz_agent_sdk.config import load_config
from daz_agent_sdk.core import Agent
from daz_agent_sdk.types import Tier


# ##################################################################
# run image
# generates an image and prints the output path
async def _run_image(args: argparse.Namespace) -> int:
    from daz_agent_sdk.capabilities.image import (
        _validate_legacy_image_config,
        _validate_image_route,
        resume_image_operation,
    )

    config = load_config()
    _validate_legacy_image_config(config)
    _validate_image_route("validation", 1, 1, args.provider, None, None)
    if args.recover:
        result = await resume_image_operation(
            args.recover, output=args.output, config=config
        )
        print(str(result.path))
        return 0
    if not args.prompt:
        raise ValueError("--prompt is required unless --recover is used")
    agent = Agent(config)
    output = args.output if args.output else None

    images: list[str] = list(args.image) if args.image else []
    image_arg: Any
    if not images:
        image_arg = None
    elif len(images) == 1:
        image_arg = images[0]
    else:
        image_arg = images
    result = await agent.image(
        args.prompt,
        width=args.width,
        height=args.height,
        output=output,
        image=image_arg,
        transparent=args.transparent,
        provider=args.provider,
        idempotency_key=args.idempotency_key,
        operation_state=args.state,
    )
    print(str(result.path))
    return 0


# ##################################################################
# run ask
# executes a single-shot ask and prints the result
async def _run_ask(args: argparse.Namespace) -> int:
    agent = Agent(load_config())
    tier = Tier(args.tier)
    result = await agent.ask(args.prompt, tier=tier)
    print(result.text)
    return 0


# ##################################################################
# run models
# lists available models and prints them
async def _run_models(args: argparse.Namespace) -> int:
    agent = Agent(load_config())
    tier = Tier(args.tier) if args.tier else None
    models = await agent.models(tier=tier)
    for m in models:
        print(f"{m.qualified_name}  {m.display_name}  [{m.tier.value}]")
    if not models:
        print("No models available.")
    return 0


# ##################################################################
# build parser
# creates the argparse parser with subcommands
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-sdk",
        description="agent-sdk CLI — one library, every AI capability",
    )
    sub = parser.add_subparsers(dest="command")

    # ask
    ask_p = sub.add_parser("ask", help="Send a single-shot prompt")
    ask_p.add_argument("prompt", help="The prompt to send")
    ask_p.add_argument(
        "--tier",
        default="high",
        choices=[t.value for t in Tier],
        help="Model tier (default: high)",
    )

    # image
    img_p = sub.add_parser("image", help="Generate an image from a text prompt")
    img_p.add_argument("--prompt", default="", help="The image prompt")
    img_p.add_argument("--width", type=int, default=0, help="Image width")
    img_p.add_argument("--height", type=int, default=0, help="Image height")
    img_p.add_argument("--output", default=None, help="Output file path")
    img_p.add_argument(
        "--image",
        "-i",
        action="append",
        default=None,
        help="Input image file for editing/reference. Repeat to attach multiple images (codex only).",
    )
    img_p.add_argument("--transparent", action="store_true", help="Remove background")
    img_p.add_argument(
        "--state", default=None, help="Crash-safe image submission state file"
    )
    img_p.add_argument(
        "--idempotency-key", default=None, help="Durable image submission key"
    )
    img_p.add_argument(
        "--recover", default=None, help="Recover a crash-safe image state file"
    )
    img_p.add_argument(
        "--provider",
        default=None,
        help="Image provider (only 'codex' is supported; routes to the mac mini image_generation_service)",
    )

    # models
    models_p = sub.add_parser("models", help="List available models")
    models_p.add_argument(
        "--tier",
        default=None,
        choices=[t.value for t in Tier],
        help="Filter by tier",
    )

    return parser


# ##################################################################
# main
# entry point called by the run script
def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "ask": _run_ask,
        "image": _run_image,
        "models": _run_models,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return asyncio.run(handler(args))
