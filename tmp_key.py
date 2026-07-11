import hashlib, json, sys
prompt = 'What is the capital of France?'
print('key', hashlib.sha256(prompt.encode()).hexdigest())
