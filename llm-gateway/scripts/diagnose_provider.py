"""Diagnostic script for a single provider entry.

This script loads the gateway settings, resolves a provider entry by name,
instantiates the concrete provider (cloud or local) with the exact
configuration from the tier config, sends a minimal ``ChatCompletion`` request,
and prints detailed request/response information. It is useful for debugging
issues such as missing ``base_url`` or API‑key problems and for verifying that
the NVIDIA endpoint is reachable.

Usage::

    python scripts/diagnose_provider.py --provider nvidia-nano

The script prints:
    * Resolved provider type and base URL
    * Request URL and headers (API key is masked)
    * Request payload (JSON)
    * Response status code and URL
    * First 500 characters of the response body

The script runs asynchronously using ``asyncio.run``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from app.api.schemas.requests import ChatCompletionRequest, Message
from app.core.config import Settings
from app.providers.cloud_provider import CloudProvider
from app.providers.local_provider import LocalProvider
from app.router.tier_config import build_tier_configs_from_settings


def _mask_api_key(key: str | None) -> str:
    """Return a masked version of an API key for safe logging.

    The key is truncated to the first 4 and last 4 characters with ``...`` in
    between. If ``None`` is passed, an empty string is returned.
    """
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


async def _run_diagnostic(provider_name: str) -> None:
    # Load settings from the development YAML file (relative to project root).
    settings_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.dev.yaml")
    settings = Settings.from_yaml(settings_path)

    # Build tier configs and locate the requested provider entry.
    tier_configs: list[Any] = build_tier_configs_from_settings(settings)
    entry = None
    for tier in tier_configs:
        for pe in tier.providers:
            if pe.name == provider_name:
                entry = pe
                break
        if entry:
            break
    if entry is None:
        print(f"Provider entry '{provider_name}' not found in configuration.")
        sys.exit(1)

    # Resolve API key if needed.
    api_key = None
    if entry.api_key_env:
        api_key = os.getenv(entry.api_key_env)

    # Instantiate the concrete provider.
    if entry.provider_kind == "cloud":
        provider = CloudProvider(
            settings,
            base_url=entry.base_url or None,
            api_key=api_key,
            timeout_seconds=entry.timeout_seconds,
            provider_name=entry.name,
        )
        endpoint = "/chat/completions"
    else:
        provider = LocalProvider(
            settings,
            base_url=entry.base_url or None,
            timeout_seconds=entry.timeout_seconds,
            provider_name=entry.name,
        )
        endpoint = "/v1/chat/completions"

    # Prepare a minimal request payload.
    request = ChatCompletionRequest(
        model=entry.model,
        messages=[Message(role="user", content="Hello" )],
        temperature=0.0,
    )

    # Print diagnostic information before sending.
    print("=== Provider Diagnostic ===")
    print(f"Provider name   : {entry.name}")
    print(f"Kind            : {entry.provider_kind}")
    print(f"Base URL        : {entry.base_url or '(none)'}")
    print(f"API key (masked): {_mask_api_key(api_key)}")
    print(f"Model           : {entry.model}")
    print(f"Timeout seconds : {entry.timeout_seconds}")
    print(f"Endpoint        : {endpoint}")
    print("--- Request payload (JSON) ---")
    print(json.dumps(request.model_dump(), indent=2))

    # Perform the request.
    try:
        response = await provider.generate(request)
    except Exception as exc:  # noqa: BLE001
        print("\n=== Request failed ===")
        print(str(exc))
        # If the exception is a ProviderError, it may contain extra attributes.
        for attr in ("url", "response_status", "response_body", "request_id"):
            if hasattr(exc, attr):
                print(f"{attr}: {getattr(exc, attr)}")
        sys.exit(1)

    # Successful response – print summary.
    print("\n=== Response summary ===")
    # The provider's client base URL may have been normalized; construct full URL.
    full_url = f"{entry.base_url.rstrip('/')}{endpoint}"
    print(f"Request URL : {full_url}")
    print(f"Status code : {response.id or 'N/A'} (provider internal ID)" )
    # Show first part of the raw response JSON.
    raw_json = json.dumps(response.model_dump(), ensure_ascii=False)
    print("Response body (first 500 chars):")
    print(raw_json[:500])


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a single provider entry.")
    parser.add_argument(
        "--provider",
        required=True,
        help="Name of the provider entry as defined in the tier configuration.",
    )
    args = parser.parse_args()
    asyncio.run(_run_diagnostic(args.provider))


if __name__ == "__main__":
    main()
