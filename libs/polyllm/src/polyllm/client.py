from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import ModelProfile, PolyllmConfig
from .secrets import SecretProvider, default_secret_provider


@dataclass
class ChatResult:
    text: str
    raw: Dict[str, Any]


def _resolve_api_key(profile: ModelProfile, secrets: SecretProvider) -> Optional[str]:
    """
    Resolve the API key using:
      1) api_key_ref (preferred)  e.g. env:OPENAI_API_KEY or file:./secrets.json#OPENAI_API_KEY
      2) api_key_env (deprecated, backward compatible)
    """
    if profile.api_key_ref:
        return secrets.get(profile.api_key_ref)

    # Backward compatibility: api_key_env="OPENAI_API_KEY"
    if profile.api_key_env:
        # Resolve as env:* to keep one mechanism
        return secrets.get(f"env:{profile.api_key_env}")

    return None


class LLMClient:
    """
    Minimal facade with internal secret resolution.

    - Clients may pass PolyllmConfig, but must not pass raw secrets.
    - Secrets are resolved by polyllm using a SecretProvider:
        - env:* (CI/dev)
        - file:* (local dev)
      Vault can be added later without changing the public API.
    """

    def __init__(self, cfg: PolyllmConfig, *, secrets: Optional[SecretProvider] = None) -> None:
        self.cfg = cfg
        self.secrets: SecretProvider = secrets or default_secret_provider()

    def _resolve_profile(self, profile: Optional[str]) -> ModelProfile:
        name = profile or self.cfg.default_profile
        if name not in self.cfg.profiles:
            raise KeyError(f"Unknown profile '{name}'. Known: {list(self.cfg.profiles.keys())}")
        return self.cfg.profiles[name]

    async def chat(self, messages: List[Dict[str, str]], *, profile: Optional[str] = None) -> ChatResult:
        p = self._resolve_profile(profile)

        # LangChain is optional; require langchain extra for actual calls
        try:
            from langchain.chat_models import init_chat_model
        except Exception as e:
            raise RuntimeError(
                "LangChain backend not installed. Install with: pip install polyllm[langchain]"
            ) from e

        api_key = _resolve_api_key(p, self.secrets)

        llm = init_chat_model(
            model=p.model,
            model_provider=p.provider,
            temperature=p.temperature,
            api_key=api_key,   # direct providers use this; vertex/bedrock often use ambient creds
            base_url=p.base_url,
        )

        resp = await llm.ainvoke(messages)
        text = getattr(resp, "content", None) or str(resp)

        return ChatResult(
            text=text,
            raw={
                "provider": p.provider,
                "model": p.model,
                "transport": p.transport,
            },
        )