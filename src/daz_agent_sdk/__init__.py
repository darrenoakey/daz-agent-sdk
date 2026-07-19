__version__ = "0.2.18"

from daz_agent_sdk.conversation import Conversation
from daz_agent_sdk.core import Agent
from daz_agent_sdk.capabilities.image import (
    download_image_job,
    get_image_job,
    resume_image_job,
    resume_image_operation,
    submit_image_job,
)
from daz_agent_sdk.types import (
    AgentError,
    AudioResult,
    Capability,
    EmbeddingResult,
    ErrorKind,
    ImageJobStatus,
    ImageResult,
    ImageSubmission,
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
    "EmbeddingResult",
    "ErrorKind",
    "ImageJobStatus",
    "ImageResult",
    "ImageSubmission",
    "Message",
    "ModelInfo",
    "Response",
    "StructuredResponse",
    "Tier",
    "agent",
    "download_image_job",
    "get_image_job",
    "resume_image_job",
    "resume_image_operation",
    "submit_image_job",
]
