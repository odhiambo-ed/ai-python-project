import asyncio
import httpx
from typing import List, Dict, Any
from src.models.schemas import ChatRequest, ChatMessage

async def call_llm(prompt: str, api_key: str, model: str = "gpt-4o") -> str:
    """Make async call to LLM API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

async def batch_inference(prompts: List[str], api_key: str, 
                         max_concurrent: int = 5) -> List[str]:
    """Process multiple prompts concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_call(prompt: str):
        async with semaphore:
            return await call_llm(prompt, api_key)
    
    tasks = [limited_call(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)