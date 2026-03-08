from __future__ import annotations

from typing import Optional

from .client import LLMClient
from .config import ModelProfile, PolyllmConfig
from .secrets import SecretProvider, default_secret_provider


class RemoteConfigLoader:
    """
    Fetches a ModelProfile from ConfigForge by canonical ref and returns a ready LLMClient.

    The canonical ref format is: {env}.{kind}[.{provider}][.{platform}].{name}
    Examples:
        prod.llm.openai.default
        prod.llm.anthropic.raina.primary
        dev.llm.openai.orko.agents

    Usage:
        loader = RemoteConfigLoader(base_url="http://config-forge-service:8040")
        client = await loader.load("prod.llm.openai.raina.primary")
        result = await client.chat([{"role": "user", "content": "Hello"}])

    Requires: pip install polyllm[remote]
    """

    def __init__(
        self,
        base_url: str,
        *,
        secrets: Optional[SecretProvider] = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.secrets = secrets or default_secret_provider()
        self.timeout = timeout

    async def load(self, ref: str) -> LLMClient:
        """Fetch config by canonical ref from ConfigForge and return an initialized LLMClient."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "Missing dependency for RemoteConfigLoader. Install with: pip install polyllm[remote]"
            ) from e

        url = f"{self.base_url}/config/resolve/{ref}"
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            response = await http.get(url)
            response.raise_for_status()

        payload = response.json()
        profile = ModelProfile(**payload["data"])
        cfg = PolyllmConfig(profiles={"default": profile})
        return LLMClient(cfg, secrets=self.secrets)
