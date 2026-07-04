from crucible.task_router import classify_task, infer_tags, route, route_by_task


def test_classify_code_math_creative_chat():
    assert classify_task("Write a python function to refactor this code")["type"] == "code"
    assert classify_task("Calculate the derivative and solve the equation")["type"] == "math"
    assert classify_task("Write a story, a poem, something creative")["type"] == "creative"
    assert classify_task("hey how's it going")["type"] == "chat"
    assert classify_task("")["type"] == "chat"


def test_classify_confidence():
    c = classify_task("solve this equation: derivative of x^2 =")
    assert c["type"] == "math" and c["confidence"] > 0


def test_infer_tags_and_tier():
    tags, tier = infer_tags("qwen2.5-coder", "Q4_K_M")
    assert "code" in tags and tier == 1
    tags2, tier2 = infer_tags("Llama-70B-Instruct", "Q4_K_M")
    assert tier2 == 2
    tags3, tier3 = infer_tags("qwen2.5-0.5b", "Q2_K")
    assert tier3 == 0                       # tiny + heavy quant -> fast tier
    assert infer_tags("mistral-7b")[0] == ["chat"]


def test_route_by_task_prefers_tag_match():
    models = [
        {"id": "coder", "tags": ["code"], "tier": 1},
        {"id": "chatter", "tags": ["chat"], "tier": 1},
    ]
    assert route_by_task("code", models) == "coder"
    assert route_by_task("chat", models) == "chatter"


def test_route_by_task_honors_level():
    models = [
        {"id": "small", "tags": ["chat"], "tier": 0},
        {"id": "big", "tags": ["chat"], "tier": 2},
    ]
    assert route_by_task("chat", models, user_level="fast") == "small"
    assert route_by_task("chat", models, user_level="max") == "big"


def test_route_skips_unavailable():
    models = [{"id": "a", "tags": ["code"], "tier": 1}, {"id": "b", "tags": ["chat"], "tier": 1}]
    up = {"b"}
    assert route_by_task("code", models, is_available=lambda i: i in up) == "b"
    assert route_by_task("code", [], is_available=lambda i: True) is None


def test_route_end_to_end():
    models = [{"id": "coder", "tags": ["code"], "tier": 2}, {"id": "mini", "tags": ["chat"], "tier": 0}]
    r = route("fix the bug in this python function", models, user_level="max")
    assert r["task"] == "code" and r["chosen"] == "coder"
    assert "code task" in r["why"]
