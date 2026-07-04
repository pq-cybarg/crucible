from crucible.media import (comfyui_txt2img, content_modalities, embeddings_request,
                            is_multimodal_content, media_endpoint, openai_image_request)
from crucible.tools import default_registry
from crucible.tools.media import GenerateImage, Transcribe


def test_comfyui_workflow_valid_shape():
    wf = comfyui_txt2img("a cat", steps=30, width=768)["prompt"]
    assert wf["6"]["inputs"]["text"] == "a cat"
    assert wf["3"]["inputs"]["steps"] == 30
    assert wf["5"]["inputs"]["width"] == 768
    assert wf["9"]["class_type"] == "SaveImage"


def test_openai_image_and_embeddings_requests():
    assert openai_image_request("x", "256x256", 2) == {"prompt": "x", "size": "256x256", "n": 2}
    assert embeddings_request("hi")["input"] == ["hi"]
    assert embeddings_request(["a", "b"])["input"] == ["a", "b"]


def test_is_multimodal_and_modalities():
    text = [{"role": "user", "content": "hi"}]
    mm = [{"role": "user", "content": [{"type": "text", "text": "look"},
                                       {"type": "image_url", "image_url": {"url": "x"}}]}]
    assert is_multimodal_content(text) is False
    assert is_multimodal_content(mm) is True
    assert content_modalities(mm) == ["image", "text"]
    assert content_modalities([{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]) == ["audio"]


def test_media_tools_registered(tmp_path):
    names = {t.name for t in default_registry(tmp_path).all()}
    assert {"generate_image", "transcribe_audio"} <= names


def test_media_tools_error_without_backend(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_IMAGE_ENDPOINT", raising=False)
    monkeypatch.delenv("CRUCIBLE_STT_ENDPOINT", raising=False)
    assert GenerateImage().run("a cat").ok is False
    assert Transcribe().run("a.wav").ok is False


def test_media_endpoint_env(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_IMAGE_ENDPOINT", "http://127.0.0.1:8188/")
    assert media_endpoint("image") == "http://127.0.0.1:8188"
