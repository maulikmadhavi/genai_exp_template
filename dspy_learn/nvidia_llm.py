from openai import OpenAI
import os
import re
import dspy

UA = "TopicCollector/1.0"

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ.get("NVIDIA_API_KEY"))

# -----------------------------
# LLM call
# -----------------------------


def call_llm(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=1,
        max_tokens=4096,
        stream=False,
    )
    return completion.choices[0].message.content


# y = call_llm("how DSPy library works? how it help to optimize the prompt?")
# print(y)


lm = dspy.LM(
    "openai/openai/gpt-oss-20b",
    api_base="https://integrate.api.nvidia.com/v1",  # ensure this points to your port
    api_key=os.getenv("NVIDIA_API_KEY"),
    model_type="chat",
)
