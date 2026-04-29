import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as a


class _ChatSeq:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, host, model, messages, timeout):
        self.calls.append(messages)
        if self.replies:
            return self.replies.pop(0)
        return "FINAL_ANSWER: fallback"


def test_rejects_multiple_tool_calls_in_one_turn(monkeypatch):
    monkeypatch.setitem(a.TOOLS, "read_file", lambda path: "ok")
    chat = _ChatSeq(
        [
            'TOOL_CALL: {"tool":"read_file","args":{"path":"/tmp/a"}} TOOL_CALL: {"tool":"read_file","args":{"path":"/tmp/b"}}',
            "FINAL_ANSWER: done",
        ]
    )

    draft, pred, meta = a.solve_with_tools(
        ollama_chat_fn=chat,
        host="http://localhost:11434",
        model="m",
        question="q",
        attachment_path="",
        attachment_image_b64="",
        timeout=30,
        max_steps=3,
        tool_prompt_profile="native:itbench_lite_finops",
    )

    assert pred == "done"
    assert meta["tool_errors"] >= 1
    assert "TOOL_RESULT[parse_error]" in draft


def test_malformed_tool_call_json_recovers_then_finishes(monkeypatch):
    monkeypatch.setitem(a.TOOLS, "read_file", lambda path: "ok")
    chat = _ChatSeq(
        [
            'TOOL_CALL: {"tool":"read_file","args":{"path":"/tmp/a"',
            "FINAL_ANSWER: corrected",
        ]
    )

    draft, pred, meta = a.solve_with_tools(
        ollama_chat_fn=chat,
        host="http://localhost:11434",
        model="m",
        question="q",
        attachment_path="",
        attachment_image_b64="",
        timeout=30,
        max_steps=3,
        tool_prompt_profile="native:itbench_lite_finops",
    )

    assert pred == "corrected"
    assert meta["tool_errors"] >= 1
    assert "TOOL_RESULT[parse_error]" in draft


def test_loop_guard_blocks_repeated_identical_calls(monkeypatch):
    monkeypatch.setitem(a.TOOLS, "read_file", lambda path: "file")
    chat = _ChatSeq(
        [
            'TOOL_CALL: {"tool":"read_file","args":{"path":"/tmp/a"}}',
            'TOOL_CALL: {"tool":"read_file","args":{"path":"/tmp/a"}}',
            'TOOL_CALL: {"tool":"read_file","args":{"path":"/tmp/a"}}',
            "FINAL_ANSWER: done",
        ]
    )

    draft, pred, meta = a.solve_with_tools(
        ollama_chat_fn=chat,
        host="http://localhost:11434",
        model="m",
        question="q",
        attachment_path="",
        attachment_image_b64="",
        timeout=30,
        max_steps=6,
        tool_prompt_profile="native:itbench_lite_finops",
    )

    assert pred == "done"
    assert meta["tool_errors"] >= 1
    assert "TOOL_RESULT[loop_guard]" in draft


def test_unknown_tool_is_remapped_for_sre_profile(monkeypatch, tmp_path):
    scenario_file = tmp_path / "k8s_events_raw.tsv"
    scenario_file.write_text("col\n1\n", encoding="utf-8")
    seen = {"query": ""}

    def _fake_nl2kubectl(nl_query: str):
        seen["query"] = nl_query
        return "NL2_OK"

    monkeypatch.setitem(a.TOOLS, "nl2kubectl", _fake_nl2kubectl)
    chat = _ChatSeq(
        [
            'TOOL_CALL: {"tool":"run_python","args":{"code":"print(1)"}}',
            "FINAL_ANSWER: done",
        ]
    )

    draft, pred, _ = a.solve_with_tools(
        ollama_chat_fn=chat,
        host="http://localhost:11434",
        model="m",
        question="q",
        attachment_path=str(scenario_file),
        attachment_image_b64="",
        timeout=30,
        max_steps=4,
        tool_prompt_profile="native:itbench_lite_sre",
    )

    assert pred == "done"
    assert "TOOL_RESULT[nl2kubectl]" in draft
    assert "inspect file" in seen["query"]


def test_execute_tool_with_retries_recovers_from_transient_error():
    a.configure_tool_cache(mode="off", file_path="")
    a.TOOL_CACHE.clear()
    calls = {"n": 0}

    def _flaky(path: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Tool error: not found"
        return "OK"

    obs, tool_errors = a.execute_tool_with_retries(
        tool="read_file",
        args={"path": "/tmp/a"},
        attachment_path="",
        dynamic_tools={"read_file": _flaky},
        max_attempts=2,
    )

    assert obs == "OK"
    assert tool_errors == 1
    assert calls["n"] == 2
