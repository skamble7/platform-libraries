# polyllm

**A Config-Driven, Provider-Agnostic LLM Client for LLM-Driven
Applications and Agents**

------------------------------------------------------------------------

# 1. Why polyllm Exists

## The Reality of Modern LLM Applications

If you are building:

-   An AI agent
-   A copilots-style assistant
-   A workflow automation engine
-   A RAG-based system
-   A multi-tenant AI platform

You will eventually face this problem:

> Your application must support multiple LLM providers --- without
> rewriting your code.

Different users, environments, and enterprises prefer different
providers:

  Scenario                Preferred LLM Access
  ----------------------- ----------------------------
  Startup MVP             OpenAI direct API
  Google Cloud org        Gemini / Vertex
  AWS enterprise          Bedrock
  Regulated environment   Private gateway
  On-prem                 Self-hosted model endpoint

Hardcoding provider SDKs in your business logic leads to:

-   Vendor lock-in
-   Secret sprawl
-   Code duplication
-   Difficult environment portability
-   Complex conditional logic everywhere

------------------------------------------------------------------------

# 2. What polyllm Is

`polyllm` is:

> A config-driven abstraction layer that initializes and routes LLM
> calls across providers without requiring code changes in your
> application.

It sits between your application logic and provider SDKs.

Application / Agent\
↓\
polyllm\
↓\
OpenAI / Gemini / Vertex / Bedrock / Gateway

------------------------------------------------------------------------

# 3. What polyllm Does

polyllm provides:

-   A unified `LLMClient`
-   Runtime model/profile selection
-   Secret resolution via references
-   Provider initialization abstraction
-   A consistent chat interface

It allows you to:

-   Switch models via config
-   Switch providers via config
-   Route different requests to different LLMs
-   Centralize secret handling
-   Keep business logic clean

------------------------------------------------------------------------

# 4. What polyllm Does NOT Do

polyllm does not:

-   Design prompts
-   Implement RAG
-   Manage vector databases
-   Perform orchestration
-   Replace LangChain entirely
-   Provide agent frameworks

It is intentionally focused on **LLM initialization and routing**.

------------------------------------------------------------------------

# 5. Core Design Principles

### 1. Configuration Over Conditionals

Provider selection is done via configuration, not `if` statements.

### 2. Secrets Are Never Passed Directly

Clients pass secret references --- not raw API keys.

### 3. Runtime Flexibility

Different profiles can be selected per request.

### 4. Backend Replaceability

Today polyllm uses LangChain as a backend. Tomorrow it can use direct
SDKs or alternative backends without changing your app code.

------------------------------------------------------------------------

# 6. Architecture Overview

Application / Agent\
↓\
PolyllmConfig\
↓\
LLMClient\
↓\
SecretProvider (env / file / vault)\
↓\
Provider Backend\
↓\
External LLM

------------------------------------------------------------------------

# 7. Installation

## Using pip

pip install polyllm\[langchain\]

## Using uv

uv sync

------------------------------------------------------------------------

# 8. Configuration Model

## ModelProfile

-   provider
-   model
-   transport (direct / vertex / bedrock / gateway)
-   temperature
-   max_tokens
-   base_url
-   headers
-   query_params
-   api_key_ref

------------------------------------------------------------------------

# 9. Basic Usage

``` python
from polyllm import LLMClient, PolyllmConfig

cfg = PolyllmConfig(
    default_profile="openai",
    profiles={
        "openai": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_ref": "env:OPENAI_API_KEY",
            "temperature": 0.1,
        }
    },
)

client = LLMClient(cfg)

result = await client.chat([
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain polyllm in one sentence."}
])

print(result.text)
```

------------------------------------------------------------------------

# 10. Summary

polyllm is:

-   A lightweight abstraction boundary for LLM initialization
-   A configuration-driven routing layer
-   A secret-safe integration layer
-   A future-proof foundation for multi-provider AI systems

It enables:

> Switching LLM providers without rewriting application logic.
