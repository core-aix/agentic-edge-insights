import base64
import io
import os
import sys
from pathlib import Path

import pytest
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark as rgb


def _encode_solid_image_rgb(rgb_tuple=(255, 0, 0), size=(256, 256)) -> str:
    img = Image.new("RGB", size, rgb_tuple)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _has_vision_capability(host: str, model: str) -> bool:
    try:
        resp = requests.post(
            f"{host.rstrip('/')}/api/show",
            json={"model": model},
            timeout=20,
        )
        if resp.status_code != 200:
            return False
        caps = [str(x).lower() for x in (resp.json().get("capabilities") or [])]
        return "vision" in caps
    except Exception:  # noqa: BLE001
        return False


def test_actual_vlm_receives_image_via_ollama():
    host = os.environ.get("BENCHMARK_TEST_OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("BENCHMARK_TEST_VLM_MODEL", "qwen3-vl:8b")

    if not _has_vision_capability(host, model):
        pytest.skip(
            "No reachable vision-capable Ollama model configured for integration test"
        )

    b64 = _encode_solid_image_rgb((255, 0, 0))
    out = rgb.ollama_chat(
        host,
        model,
        [
            {
                "role": "system",
                "content": "You are a vision model. Answer in one word.",
            },
            {
                "role": "user",
                "content": "What is the dominant color in this image? one word only.",
                "images": [b64],
            },
        ],
        timeout=180,
    )
    s = str(out or "").strip().lower()
    assert s
    assert any(k in s for k in ["red", "crimson", "scarlet"])
