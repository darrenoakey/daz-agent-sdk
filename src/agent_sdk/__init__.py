__version__ = "0.1.0"

from agent_sdk.conversation import Conversation
from agent_sdk.core import Agent
from agent_sdk.types import (
    AgentError,
    AudioResult,
    Capability,
    ErrorKind,
    ImageResult,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    Tier,
)

# module-level singleton — ready to use on import
agent = Agent()

__all__ = [
    "Agent",
    "AgentError",
    "AudioResult",
    "Capability",
    "Conversation",
    "ErrorKind",
    "ImageResult",
    "Message",
    "ModelInfo",
    "Response",
    "StructuredResponse",
    "Tier",
    "agent",
]
