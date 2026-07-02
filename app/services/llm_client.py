import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Groq is OpenAI-compatible - same SDK, different base_url.
# This single client is the one place we talk to the generation LLM.
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

GEN_MODEL = "llama-3.3-70b-versatile"


def chat(prompt: str, system: str | None = None, temperature: float = 0.2,
         return_usage: bool = False):
    """Single-turn completion. Optionally returns token usage."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        temperature=temperature,
    )
    text = response.choices[0].message.content

    if return_usage:
        usage = response.usage
        return text, {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
    return text