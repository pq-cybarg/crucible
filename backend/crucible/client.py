from __future__ import annotations
import httpx


class ChatClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    async def chat(self, messages: list[dict], model: str = "local",
                   temperature: float = 0.7, max_tokens: int = 512) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{self.endpoint}/v1/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                return (await c.get(f"{self.endpoint}/health")).status_code == 200
        except httpx.HTTPError:
            return False
