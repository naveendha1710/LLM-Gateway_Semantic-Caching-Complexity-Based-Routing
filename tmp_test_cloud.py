import os, json, httpx, asyncio, time

payload = {
    "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": True},
    "reasoning_budget": 1024,
}

async def main():
    api_key = os.getenv('NVIDIA_API_KEY')
    if not api_key:
        print('No key')
        return
    async with httpx.AsyncClient(base_url='https://integrate.api.nvidia.com/v1', timeout=60, headers={
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }) as client:
        try:
            resp = await client.post('/chat/completions', json=payload)
            print('status', resp.status_code)
            print('text', resp.text[:200])
        except Exception as e:
            print('exception', e)

asyncio.run(main())
