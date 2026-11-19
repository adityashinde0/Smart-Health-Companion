import os
import sys
from openai import OpenAI

# Retrieve API key securely from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Warning: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
    print("Please set OPENAI_API_KEY before running this script:", file=sys.stderr)
    print("  $env:OPENAI_API_KEY='your_api_key_here' (PowerShell) or export OPENAI_API_KEY='your_api_key_here' (Bash)", file=sys.stderr)
    sys.exit(1)

# Initialize the OpenAI client configured for CloudRift DeepSeek-V3
client = OpenAI(
    api_key=api_key,
    base_url="https://inference.cloudrift.ai/v1"
)

# Create a streaming chat completion
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[
        {"role": "system", "content": "You are a professional clinical health AI assistant."},
        {"role": "user", "content": "Explain cardiovascular risk assessment and preventative measures in simple terms."}
    ],
    stream=True
)

# Print the streamed response
for chunk in response:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print()
