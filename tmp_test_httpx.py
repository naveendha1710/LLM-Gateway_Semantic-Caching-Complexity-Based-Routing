import os, httpx, json, sys
api_key = os.getenv('NVIDIA_API_KEY')
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
print('Starting request...')
try:
    resp = httpx.get('https://integrate.api.nvidia.com/v1/models', headers=headers, timeout=30.0)
    print('status', resp.status_code)
    print('body', resp.text[:200])
except Exception as e:
    print('error', e)
