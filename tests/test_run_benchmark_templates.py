import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark as r


class _FakeResponse:
    def __init__(self):
        self.status_code = 200

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        yield json.dumps({"message": {"content": "OK"}, "done": True})


def test_ollama_chat_adapts_roles_for_prompt_only_template(monkeypatch):
    sent_payloads = []

    def fake_post(url, json=None, timeout=None, stream=False):
        sent_payloads.append({"url": url, "json": json, "stream": stream})
        return _FakeResponse()

    monkeypatch.setattr(r.requests, "post", fake_post)
    monkeypatch.setattr(r, "get_model_max_context", lambda host, model: 8192)
    monkeypatch.setattr(r, "get_model_max_output_tokens", lambda host, model, ctx: 512)
    monkeypatch.setattr(
        r,
        "model_uses_prompt_only_template",
        lambda host, model: model == "prompt-only-model",
    )

    msgs = [
        {"role": "system", "content": "System rule"},
        {"role": "user", "content": "Question one"},
        {"role": "assistant", "content": "Interim answer"},
        {"role": "user", "content": "Question two", "images": ["BASE64"]},
    ]

    out = r.ollama_chat("http://localhost:11434", "prompt-only-model", msgs, timeout=30)
    assert out == "OK"
    payload = sent_payloads[-1]["json"]
    assert payload["stream"] is True
    assert len(payload["messages"]) == 1
    only = payload["messages"][0]
    assert only["role"] == "user"
    assert "[SYSTEM]" in only["content"]
    assert "[USER]" in only["content"]
    assert "[ASSISTANT]" in only["content"]
    assert only.get("images") == ["BASE64"]


def test_ollama_chat_keeps_roles_for_chat_template(monkeypatch):
    sent_payloads = []

    def fake_post(url, json=None, timeout=None, stream=False):
        sent_payloads.append({"url": url, "json": json, "stream": stream})
        return _FakeResponse()

    monkeypatch.setattr(r.requests, "post", fake_post)
    monkeypatch.setattr(r, "get_model_max_context", lambda host, model: 8192)
    monkeypatch.setattr(r, "get_model_max_output_tokens", lambda host, model, ctx: 512)
    monkeypatch.setattr(r, "model_uses_prompt_only_template", lambda host, model: False)

    msgs = [
        {"role": "system", "content": "System rule"},
        {"role": "user", "content": "Question one"},
    ]

    out = r.ollama_chat(
        "http://localhost:11434", "chat-template-model", msgs, timeout=30
    )
    assert out == "OK"
    payload = sent_payloads[-1]["json"]
    assert payload["stream"] is True
    assert payload["messages"] == msgs


def test_ollama_chat_uses_longer_read_timeout_for_image_messages(monkeypatch):
    sent = []

    def fake_post(url, json=None, timeout=None, stream=False):
        sent.append({"timeout": timeout, "json": json})
        return _FakeResponse()

    monkeypatch.setattr(r.requests, "post", fake_post)
    monkeypatch.setattr(r, "get_model_max_context", lambda host, model: 8192)
    monkeypatch.setattr(r, "get_model_max_output_tokens", lambda host, model, ctx: 512)
    monkeypatch.setattr(r, "model_uses_prompt_only_template", lambda host, model: False)

    r.OLLAMA_STEP_TIMEOUT = 120
    r.OLLAMA_IMAGE_STEP_TIMEOUT = 420

    r.ollama_chat(
        "http://localhost:11434",
        "chat-template-model",
        [{"role": "user", "content": "text only"}],
        timeout=600,
    )
    non_img_timeout = sent[-1]["timeout"][1]

    r.ollama_chat(
        "http://localhost:11434",
        "chat-template-model",
        [{"role": "user", "content": "with image", "images": ["BASE64"]}],
        timeout=600,
    )
    img_timeout = sent[-1]["timeout"][1]

    assert img_timeout > non_img_timeout


def test_solve_question_simple_includes_image_in_both_passes(monkeypatch):
    calls = []

    def fake_chat(host, model, messages, timeout):
        calls.append(messages)
        return "red"

    monkeypatch.setattr(r, "ollama_chat", fake_chat)
    monkeypatch.setattr(
        r, "maybe_attachment_image_b64", lambda host, path, model: "IMG_B64"
    )

    draft, final = r.solve_question_simple(
        host="http://localhost:11434",
        model="qwen3-vl:8b",
        question="q",
        timeout=30,
        attachment_path="/tmp/sample.png",
    )
    assert draft == "red"
    assert final == "red"
    assert len(calls) == 2
    assert calls[0][1].get("images") == ["IMG_B64"]
    assert calls[1][1].get("images") == ["IMG_B64"]


def test_solve_question_simple_prefers_concise_draft_when_final_is_meta(monkeypatch):
    replies = iter(
        [
            "Guava",
            "We are given a set of instructions and a candidate answer.",
        ]
    )

    def fake_chat(host, model, messages, timeout):
        return next(replies)

    monkeypatch.setattr(r, "ollama_chat", fake_chat)
    monkeypatch.setattr(r, "maybe_attachment_image_b64", lambda host, path, model: "")

    draft, final = r.solve_question_simple(
        host="http://localhost:11434",
        model="qwen3-vl:32b",
        question="fruit?",
        timeout=30,
        attachment_path="",
    )
    assert draft == "Guava"
    assert final == "Guava"


def test_execute_job_uses_simple_mode_for_image_when_no_native_tools(monkeypatch):
    monkeypatch.setattr(r, "model_has_native_tool_support", lambda host, model: False)
    monkeypatch.setattr(
        r, "maybe_attachment_image_b64", lambda host, path, model: "IMG_B64"
    )

    called = {"tool": 0, "simple": 0}

    def fake_tool(*args, **kwargs):
        called["tool"] += 1
        return "d", "p", {"steps_used": 1, "max_steps_reached": 0, "tool_errors": 0}

    def fake_simple(*args, **kwargs):
        called["simple"] += 1
        return "draft", "final"

    monkeypatch.setattr(r, "solve_question_tool", fake_tool)
    monkeypatch.setattr(r, "solve_question_simple", fake_simple)

    job = {
        "run": "x",
        "host": "http://localhost:11434",
        "model": "gemma3:27b",
        "question": "q",
        "augmented_question": "",
        "file_path": "/tmp/a.png",
        "task_id": "tid",
        "level": 1,
        "gold": "ans",
        "task_type": "document_or_vision",
        "file_name": "a.png",
        "timeout": 30,
        "agent_mode": "tool",
        "max_steps": 4,
        "scorer": "itbench_root_cause",
        "level_steps": {"2": 12, "3": 20},
    }

    rec = r.execute_job(job)
    assert called["simple"] == 1
    assert called["tool"] == 0
    assert rec["pred"] == "final"
