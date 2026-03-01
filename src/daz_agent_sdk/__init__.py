__version__ = "0.2.2"

from daz_agent_sdk.conversation import Conversation
from daz_agent_sdk.core import Agent
from daz_agent_sdk.types import (
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
