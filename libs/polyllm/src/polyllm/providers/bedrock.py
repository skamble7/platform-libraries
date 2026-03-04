from __future__ import annotations

from typing import Any, Dict, Optional

from polyllm.config import ModelProfile
from polyllm.secrets import SecretProvider

from .base import ProviderAdapter


class BedrockAdapter(ProviderAdapter):
    def create_chat_model(
        self,
        profile: ModelProfile,
        *,
        api_key: Optional[str],
        credentials: Dict[str, str],
        secrets: SecretProvider,
    ) -> Any:
        raise RuntimeError(
            "bedrock adapter not enabled yet. "
            "When ready, implement aws_region + secret_refs mapping (or ambient creds) using langchain-aws."
        )