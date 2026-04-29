import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark as r


def _base_job(task_type: str) -> dict:
    return {
        "run": "all",
        "host": "http://localhost:11434",
        "model": "qwen3:8b",
        "question": "q",
        "augmented_question": "",
        "file_path": "",
        "task_id": "tid",
        "level": 1,
        "gold": "gold",
        "task_type": task_type,
        "file_name": "",
        "timeout": 30,
        "agent_mode": "tool",
        "max_steps": 4,
        "scorer": "itbench_root_cause",
        "level_steps": {"2": 12, "3": 20},
        "benchmark": "itbench_lite",
    }


def test_execute_job_routes_finops_profile(monkeypatch):
    seen = {"profile": ""}

    monkeypatch.setattr(r, "model_has_native_tool_support", lambda host, model: False)
    monkeypatch.setattr(r, "maybe_attachment_image_b64", lambda host, path, model: "")
    monkeypatch.setattr(r, "score_prediction", lambda pred, gold, scorer, meta=None: True)
    monkeypatch.setattr(r, "is_vision_model", lambda host, model: False)

    def _fake_tool(host, model, question, attachment_path, timeout, max_steps, profile):
        seen["profile"] = profile
        return "draft", "answer", {"steps_used": 1, "max_steps_reached": 0, "tool_errors": 0}

    monkeypatch.setattr(r, "solve_question_tool", _fake_tool)

    rec = r.execute_job(_base_job("itbench_finops_root_cause"))
    assert seen["profile"].endswith(":itbench_lite_finops")
    assert rec["correct"] == 1


def test_execute_job_routes_sre_profile(monkeypatch):
    seen = {"profile": ""}

    monkeypatch.setattr(r, "model_has_native_tool_support", lambda host, model: False)
    monkeypatch.setattr(r, "maybe_attachment_image_b64", lambda host, path, model: "")
    monkeypatch.setattr(r, "score_prediction", lambda pred, gold, scorer, meta=None: True)
    monkeypatch.setattr(r, "is_vision_model", lambda host, model: False)

    def _fake_tool(host, model, question, attachment_path, timeout, max_steps, profile):
        seen["profile"] = profile
        return "draft", "answer", {"steps_used": 1, "max_steps_reached": 0, "tool_errors": 0}

    monkeypatch.setattr(r, "solve_question_tool", _fake_tool)

    rec = r.execute_job(_base_job("itbench_sre_root_cause"))
    assert seen["profile"].endswith(":itbench_lite_sre")
    assert rec["correct"] == 1


def test_execute_job_unknown_type_uses_generic_itbench_profile(monkeypatch):
    seen = {"profile": ""}

    monkeypatch.setattr(r, "model_has_native_tool_support", lambda host, model: False)
    monkeypatch.setattr(r, "maybe_attachment_image_b64", lambda host, path, model: "")
    monkeypatch.setattr(r, "score_prediction", lambda pred, gold, scorer, meta=None: True)
    monkeypatch.setattr(r, "is_vision_model", lambda host, model: False)

    def _fake_tool(host, model, question, attachment_path, timeout, max_steps, profile):
        seen["profile"] = profile
        return "draft", "answer", {"steps_used": 1, "max_steps_reached": 0, "tool_errors": 0}

    monkeypatch.setattr(r, "solve_question_tool", _fake_tool)

    rec = r.execute_job(_base_job("unknown_type"))
    assert seen["profile"].endswith(":itbench_lite")
    assert rec["correct"] == 1
