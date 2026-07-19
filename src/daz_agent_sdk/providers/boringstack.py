from __future__ import annotations

from daz_agent_sdk.providers.ollama import OllamaProvider


# ##################################################################
# boringstack provider
# the "boringstack" box (Darren-Boringstack, 10.0.0.42) runs a dedicated
# Ollama instance hosting the larger local models — notably the
# qwen3.6:35b-a3b MoE. it speaks the exact same Ollama REST API as a local
# Ollama server, so this is just OllamaProvider pointed at the remote host.
# keeping it a distinct provider name lets tier chains target boringstack
# without disturbing the localhost ollama provider used by other projects.
class BoringstackProvider(OllamaProvider):
    name = "boringstack"

    # ##################################################################
    # init
    # default to the boringstack host. the registry passes base_url from
    # config when present, so this default only applies with no config.
    def __init__(self, base_url: str = "http://10.0.0.42:11434") -> None:
        super().__init__(base_url=base_url)
