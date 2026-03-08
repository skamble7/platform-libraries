from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import ModelProfile, PolyllmConfig
from .providers import get_provider_adapter
from .secrets import SecretProvider, default_secret_provider


@dataclass
class ChatResult:
    text: str
    raw: Dict[str, Any]


def _coerce_content(raw: Any) -> Optional[str]:
    """
    Normalize LangChain AIMessage.content to a plain string.

    langchain-google-genai>=2.0 (and some other providers) return content as a
    list of content blocks, e.g. [{'type': 'text', 'text': '...'}], instead of
    a plain string. We flatten that here so ChatResult.text is always str.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(raw)


def _resolve_api_key(profile: ModelProfile, secrets: SecretProvider) -> Optional[str]:
    """
    Resolve API key using:
      1) api_key_ref (preferred)
      2) api_key_env (deprecated, backward compatible)
    """
    if profile.api_key_ref:
        return secrets.get(profile.api_key_ref)

    if profile.api_key_env:
        return secrets.get(f"env:{profile.api_key_env}")

    return None


def _resolve_credentials_bundle(profile: ModelProfile, secrets: SecretProvider) -> Dict[str, str]:
    """
    Resolve secret_refs (Option A) into credential_name -> secret_value.

    Missing secret value is treated as an error (better fail-fast than partial auth).
    """
    out: Dict[str, str] = {}
    for name, ref in (profile.secret_refs or {}).items():
        ref = (ref or "").strip()
        if not ref:
            continue
        val = secrets.get(ref)
        if val is None:
            raise ValueError(
                f"Missing secret for '{name}' (ref='{ref}') in profile '{profile.provider}:{profile.model}'."
            )
        out[name] = val
    return out


async def _maybe_close_chat_model(llm: Any) -> None:
    """
    Best-effort cleanup for LangChain models / underlying HTTP clients.

    Some LangChain integrations keep an AsyncClient that should be closed.
    We try common patterns without hard dependency on any specific provider.
    """
    if llm is None:
        return

    # aclose() is most common for async cleanup
    aclose = getattr(llm, "aclose", None)
    if callable(aclose):
        try:
            res = aclose()
            if hasattr(res, "__await__"):
                await res
        except Exception:
            return

    # close() sometimes exists (sync)
    close = getattr(llm, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return


class LLMClient:
    """
    Minimal facade with:
      - internal secret resolution (SecretProvider)
      - provider adapter layer (per-provider auth + param mapping)

    Clients may pass PolyllmConfig, but must not pass raw secrets.
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

        api_key = _resolve_api_key(p, self.secrets)
        credentials = _resolve_credentials_bundle(p, self.secrets)

        adapter = get_provider_adapter(p.provider)
        llm = adapter.create_chat_model(
            p,
            api_key=api_key,
            credentials=credentials,
            secrets=self.secrets,
        )

        try:
            resp = await llm.ainvoke(messages)
            text = _coerce_content(getattr(resp, "content", None)) or str(resp)
        finally:
            # Prevent "Event loop is closed" warnings from dangling httpx clients
            await _maybe_close_chat_model(llm)

        return ChatResult(
            text=text,
            raw={
                "provider": p.provider,
                "model": p.model,
                "transport": p.transport,
            },
        )
