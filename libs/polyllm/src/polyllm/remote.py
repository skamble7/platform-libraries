from __future__ import annotations

from typing import Optional

from .client import LLMClient
from .secrets import SecretProvider, default_secret_provider


class RemoteConfigLoader:
    """
    Thin wrapper for fetching a config from ConfigForge by ref and returning an LLMClient.

    For most platform code, prefer using LLMClient directly with a ref string:

        client = LLMClient("prod.llm.openai.astra.primary")
        result = await client.chat(messages)

    RemoteConfigLoader is useful when you need to:
    - Override the ConfigForge URL explicitly (e.g. in tests)
    - Configure a custom timeout or SecretProvider once and reuse for multiple refs

        loader = RemoteConfigLoader(base_url="http://localhost:8040", timeout=10.0)
        client_a = await loader.load("prod.llm.openai.astra.primary")
        client_b = await loader.load("prod.llm.bedrock.zeta.modernization")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        secrets: Optional[SecretProvider] = None,
        timeout: float = 5.0,
    ) -> None:
        self._base_url = base_url
        self._secrets = secrets or default_secret_provider()
        self._timeout = timeout

    async def load(self, ref: str) -> LLMClient:
        """Fetch config by canonical ref from ConfigForge and return an initialized LLMClient."""
        client = LLMClient(
            ref,
            secrets=self._secrets,
            timeout=self._timeout,
            config_forge_url=self._base_url,
        )
        await client._ensure_config()
        return client
