from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

Transport = Literal["direct", "vertex", "bedrock", "gateway"]


class ModelProfile(BaseModel):
    provider: str
    model: str
    transport: Transport = "direct"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None

    # headers/query are useful for gateways and custom routing (non-secret only)
    headers: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, str] = Field(default_factory=dict)

    # Secret reference resolved by polyllm (NOT the secret itself)
    # Supported now:
    #   "env:OPENAI_API_KEY"
    #   "file:/abs/path/to/secrets.json#OPENAI_API_KEY"
    #   "file:./secrets.local.json#OPENAI_API_KEY"
    # Future:
    #   "vault:kv/data/polyllm/openai#api_key"
    api_key_ref: Optional[str] = None

    # Backward-compatible (deprecated): environment variable name
    api_key_env: Optional[str] = None


class PolyllmConfig(BaseModel):
    default_profile: str = "default"
    profiles: Dict[str, ModelProfile]
    metadata: Dict[str, Any] = Field(default_factory=dict)