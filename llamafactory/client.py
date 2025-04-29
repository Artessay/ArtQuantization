# api_call_example.py
from openai import OpenAI
client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")
messages = [{"role": "user", "content": "Who are you?"}]
result = client.chat.completions.create(messages=messages, model="Qwen/Qwen2.5-32B-Instruct-Medical")
print(result.choices[0].message.content)