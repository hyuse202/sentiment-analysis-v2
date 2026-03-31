#!/usr/bin/env python3
"""Debug GLM-5 API response"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from zai import ZaiClient

client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))

prompt = """Bạn là chuyên gia phân tích tài chính. Phân loại sentiment:

TIN TỨC: VN-Index tăng mạnh 3%, vượt mốc 1,300 điểm.

Trả lời: POSITIVE, NEGATIVE, hoặc NEUTRAL"""

print("Sending request to GLM-5...")
response = client.chat.completions.create(
    model="glm-5",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500,
    temperature=0.1
)

print("\n=== FULL RESPONSE ===")
print(f"Type: {type(response)}")
print(f"\nChoices: {len(response.choices)}")

choice = response.choices[0]
print(f"\nMessage type: {type(choice.message)}")
print(f"\nContent: '{choice.message.content}'")
print(f"Content type: {type(choice.message.content)}")
print(f"Content is None: {choice.message.content is None}")

# Check all attributes
print(f"\nAll message attributes: {dir(choice.message)}")

# Check reasoning_content
if hasattr(choice.message, 'reasoning_content'):
    print(f"\nReasoning content: {choice.message.reasoning_content[:500] if choice.message.reasoning_content else 'None'}")

# Try to get raw response
print(f"\n=== RAW RESPONSE DICT ===")
if hasattr(response, '__dict__'):
    print(response.__dict__)
elif hasattr(response, 'model_dump'):
    print(response.model_dump())
