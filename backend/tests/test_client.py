import httpx
import pytest
import respx

from crucible.client import ChatClient


@pytest.mark.asyncio
@respx.mock
async def test_chat_returns_content():
    route = respx.post("http://127.0.0.1:8081/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hello world"}}]
        })
    )
    client = ChatClient("http://127.0.0.1:8081")
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "hello world"
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_health():
    respx.get("http://127.0.0.1:8081/health").mock(return_value=httpx.Response(200))
    assert await ChatClient("http://127.0.0.1:8081").health() is True
