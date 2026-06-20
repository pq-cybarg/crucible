from crucible.tools.base import ToolRegistry, ToolResult


class Echo:
    name = "echo"
    description = "echo text"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def run(self, text: str) -> ToolResult:
        return ToolResult(ok=True, output=text)


def test_registry_register_get_all():
    reg = ToolRegistry()
    reg.register(Echo())
    assert reg.get("echo").run(text="hi").output == "hi"
    assert [t.name for t in reg.all()] == ["echo"]


def test_schemas_openai_shape():
    reg = ToolRegistry()
    reg.register(Echo())
    s = reg.schemas()[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "echo"
    assert s["function"]["parameters"]["required"] == ["text"]
