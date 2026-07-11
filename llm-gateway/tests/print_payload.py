import yaml
from pathlib import Path

# Load dev settings to get provider config
settings_path = Path(__file__).parent / 'llm-gateway' / 'config' / 'settings.dev.yaml'
with open(settings_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Find nvidia-nano provider config
provider_cfg = None
for tier in cfg.get('models', {}).values():
    for prov in tier.get('providers', []):
        if prov.get('name') == 'nvidia-nano':
            provider_cfg = prov
            break
    if provider_cfg:
        break

if not provider_cfg:
    raise RuntimeError('nvidia-nano provider not found')

model = provider_cfg['model']
max_tokens = provider_cfg.get('max_tokens')
temperature = provider_cfg.get('params', {}).get('temperature', 0.7)

payload = {
    'model': model,
    'messages': [{'role': 'user', 'content': 'Hi'}],
    'temperature': temperature,
}
if max_tokens:
    payload['max_tokens'] = max_tokens
# Add NVIDIA reasoning fields
reasoning_budget = max_tokens if max_tokens else 1024
payload['chat_template_kwargs'] = {'enable_thinking': True}
payload['reasoning_budget'] = reasoning_budget

print('Request payload:')
print(payload)
