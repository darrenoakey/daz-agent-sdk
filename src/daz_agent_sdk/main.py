from __future__ import annotations

import argparse
import asyncio

from daz_agent_sdk.config import load_config
from daz_agent_sdk.core import Agent
from daz_agent_sdk.types import Tier


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
        "models": _run_models,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return asyncio.run(handler(args))
