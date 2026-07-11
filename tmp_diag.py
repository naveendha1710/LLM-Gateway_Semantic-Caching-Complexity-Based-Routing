import asyncio, json, os
from app.api.schemas.requests import ChatCompletionRequest, Message
from app.core.config import Settings
from app.providers.cloud_provider import CloudProvider
from app.router.tier_config import build_tier_configs_from_settings

settings_path = os.path.join('llm-gateway','config','settings.dev.yaml')
settings = Settings.from_yaml(settings_path)
configs = build_tier_configs_from_settings(settings)
entry = None
for tier in configs:
    for pe in tier.providers:
        if pe.name == 'nvidia-nano':
            entry = pe
            break
    if entry:
        break
if not entry:
    raise SystemExit('Provider not found')
api_key = os.getenv(entry.api_key_env) if entry.api_key_env else None
provider = CloudProvider(settings, base_url=entry.base_url or None, api_key=api_key, timeout_seconds=entry.timeout_seconds, provider_name=entry.name)
request = ChatCompletionRequest(model=entry.model, messages=[Message(role='user', content='Hi')], temperature=0.0)
resp = asyncio.run(provider.generate(request))
print(json.dumps(resp.model_dump(), ensure_ascii=False))
