from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .config import ModelProfile, PolyllmConfig
from .providers import get_provider_adapter
from .secrets import SecretProvider, default_secret_provider

_CONFIG_FORGE_URL_ENV = "CONFIG_FORGE_URL"


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

    Accepts either an inline PolyllmConfig or a ConfigForge canonical ref string.

    Inline config (local development / no ConfigForge):
        cfg = PolyllmConfig(profiles={"default": {...}})
        client = LLMClient(cfg)
        result = await client.chat(messages)

    ConfigForge ref (recommended for platform services):
        client = LLMClient("prod.llm.openai.astra.primary")
        result = await client.chat(messages)

    When a ref string is passed, CONFIG_FORGE_URL must be set in the environment.
    The config is fetched lazily on the first chat() call and cached.
    """

    def __init__(
        self,
        config: Union[PolyllmConfig, str],
        *,
        secrets: Optional[SecretProvider] = None,
        timeout: float = 5.0,
        config_forge_url: Optional[str] = None,
    ) -> None:
        if isinstance(config, str):
            self._ref: Optional[str] = config
            self.cfg: Optional[PolyllmConfig] = None
        else:
            self._ref = None
            self.cfg = config

        self.secrets: SecretProvider = secrets or default_secret_provider()
        self._timeout = timeout
        self._config_forge_url = config_forge_url  # overrides CONFIG_FORGE_URL env var

    async def _ensure_config(self) -> None:
        """Lazily fetch config from ConfigForge if this client was created with a ref string."""
        if self.cfg is not None:
            return

        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required to use ConfigForge ref strings with LLMClient. "
                "Install with: pip install polyllm[remote]"
            ) from exc

        base_url = (
            self._config_forge_url
            or os.environ.get(_CONFIG_FORGE_URL_ENV, "")
        ).rstrip("/")

        if not base_url:
            raise ValueError(
                f"ConfigForge URL not found. Set the {_CONFIG_FORGE_URL_ENV} environment "
                f"variable or pass config_forge_url= to LLMClient()."
            )

        url = f"{base_url}/config/resolve/{self._ref}"
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            response = await http.get(url)
            response.raise_for_status()

        payload = response.json()
        profile = ModelProfile(**payload["data"])
        self.cfg = PolyllmConfig(profiles={"default": profile})

    def _resolve_profile(self, profile: Optional[str]) -> ModelProfile:
        name = profile or self.cfg.default_profile
        if name not in self.cfg.profiles:
            raise KeyError(f"Unknown profile '{name}'. Known: {list(self.cfg.profiles.keys())}")
        return self.cfg.profiles[name]

    async def chat(self, messages: List[Dict[str, str]], *, profile: Optional[str] = None) -> ChatResult:
        await self._ensure_config()

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
