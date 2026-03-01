# libs/polyllm/tests/test_smoke.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from polyllm import LLMClient, PolyllmConfig


def _load_local_secrets(file_path: str) -> dict:
    p = Path(file_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _has_real_local_key(file_path: str, key_name: str) -> bool:
    """
    Returns True if secrets.local.json exists and contains a non-placeholder value for key_name.
    """
    data = _load_local_secrets(file_path)
    v = data.get(key_name)
    if not isinstance(v, str) or not v.strip():
        return False

    # Placeholder detection (matches your examples)
    if v.strip() in {"sk-...", "AIza..."}:
        return False

    # Light sanity check
    if key_name == "OPENAI_API_KEY":
        return v.strip().startswith("sk-")
    if key_name == "GEMINI_API_KEY":
        return v.strip().startswith("AIza")
    return True


def test_config_load_smoke():
    cfg = PolyllmConfig(
        default_profile="default",
        profiles={
            "default": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key_ref": "file:./secrets.local.json#OPENAI_API_KEY",
            }
        },
    )
    client = LLMClient(cfg)
    assert client.cfg.default_profile == "default"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_chat_integration_print_response():
    """
    Integration test: makes a real OpenAI network call.

    Run:
      uv run pytest -m integration -s
    """
    if not _has_real_local_key("./secrets.local.json", "OPENAI_API_KEY"):
        pytest.skip("secrets.local.json missing or OPENAI_API_KEY looks like a placeholder")

    cfg = PolyllmConfig(
        default_profile="openai",
        profiles={
            "openai": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key_ref": "file:./secrets.local.json#OPENAI_API_KEY",
                "temperature": 0.1,
                "max_tokens": 128,
            }
        },
    )

    client = LLMClient(cfg)

    result = await client.chat(
        [
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": (
                    "Reply with 'OK' then exactly 1 sentence: "
                    "'polyllm is a config-driven, provider-agnostic LLM client library that lets apps "
                    "switch providers/models without code changes.'"
                ),
            },
        ]
    )

    print("\n[polyllm integration] provider=openai model=gpt-4o-mini")
    print(result.text)

    assert isinstance(result.text, str)
    assert len(result.text.strip()) > 0
    assert "OK" in result.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_chat_integration_print_response():
    """
    Integration test: makes a real Gemini network call via LangChain's google_genai provider.

    Important:
    - LangChain init_chat_model does NOT accept model_provider="gemini".
    - For Gemini through langchain-google-genai, use model_provider="google_genai".
    - Model names are provider/package-version dependent; "gemini-flash-latest" works with this setup.

    Run:
      uv run pytest -m integration -s
    """
    if not _has_real_local_key("./secrets.local.json", "GEMINI_API_KEY"):
        pytest.skip("secrets.local.json missing or GEMINI_API_KEY looks like a placeholder")

    cfg = PolyllmConfig(
        default_profile="gemini",
        profiles={
            "gemini": {
                "provider": "google_genai",
                "model": "gemini-flash-latest",
                "api_key_ref": "file:./secrets.local.json#GEMINI_API_KEY",
                "temperature": 0.1,
                "max_tokens": 128,
            }
        },
    )

    client = LLMClient(cfg)

    result = await client.chat(
        [
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": (
                    "Reply with 'OK' then exactly 1 sentence: "
                    "'polyllm is a config-driven, provider-agnostic LLM client library that lets apps "
                    "switch providers/models without code changes.'"
                ),
            },
        ]
    )

    print("\n[polyllm integration] provider=google_genai model=gemini-flash-latest")
    print(result.text)

    assert isinstance(result.text, str)
    assert len(result.text.strip()) > 0
    assert "OK" in result.text