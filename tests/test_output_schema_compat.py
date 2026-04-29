import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark as r


def test_execute_job_record_contains_required_paper_fields(monkeypatch):
    monkeypatch.setattr(r, "solve_question_simple", lambda *args, **kwargs: ("d", "p"))
    monkeypatch.setattr(r, "score_prediction", lambda pred, gold, scorer, meta=None: True)
    monkeypatch.setattr(r, "is_vision_model", lambda host, model: False)

    job = {
        "run": "all",
        "repeat_idx": 0,
        "repeat_seed": 42,
        "host": "http://localhost:11434",
        "model": "qwen3:8b",
        "question": "q",
        "augmented_question": "",
        "file_path": "",
        "task_id": "tid",
        "level": 1,
        "gold": "p",
        "task_type": "itbench_finops_root_cause",
        "file_name": "",
        "timeout": 30,
        "agent_mode": "simple",
        "max_steps": 4,
        "scorer": "itbench_root_cause",
        "level_steps": {"2": 12, "3": 20},
        "eval_meta": {},
    }

    rec = r.execute_job(job)
    required = {
        "run",
        "repeat_idx",
        "repeat_seed",
        "model",
        "task_id",
        "level",
        "gold",
        "pred",
        "correct",
        "task_type",
        "latency_sec",
        "steps_used",
        "max_steps_reached",
        "tool_errors",
        "failure_reason",
        "error",
        "eval_meta",
        "family",
        "is_coder",
        "is_vision",
    }
    assert required.issubset(set(rec.keys()))


def test_execute_job_failure_reason_includes_tool_and_step_flags(monkeypatch):
    monkeypatch.setattr(r, "model_has_native_tool_support", lambda host, model: False)
    monkeypatch.setattr(r, "maybe_attachment_image_b64", lambda host, path, model: "")
    monkeypatch.setattr(r, "is_vision_model", lambda host, model: False)
    monkeypatch.setattr(r, "score_prediction", lambda pred, gold, scorer, meta=None: False)

    def _fake_tool(*args, **kwargs):
        return "d", "wrong", {"steps_used": 4, "max_steps_reached": 1, "tool_errors": 2}

    monkeypatch.setattr(r, "solve_question_tool", _fake_tool)

    job = {
        "run": "all",
        "repeat_idx": 0,
        "repeat_seed": 42,
        "host": "http://localhost:11434",
        "model": "qwen3:8b",
        "question": "q",
        "augmented_question": "",
        "file_path": "",
        "task_id": "tid",
        "level": 1,
        "gold": "gold",
        "task_type": "itbench_sre_root_cause",
        "file_name": "",
        "timeout": 30,
        "agent_mode": "tool",
        "max_steps": 4,
        "scorer": "itbench_root_cause",
        "level_steps": {"2": 12, "3": 20},
        "benchmark": "itbench_lite",
        "eval_meta": {},
    }

    rec = r.execute_job(job)
    assert rec["correct"] == 0
    assert rec["failure_reason"] == "tool_error+max_steps_reached"


def test_summarize_writes_expected_summary_columns(tmp_path):
    rows = [
        {
            "run": "all",
            "repeat_idx": 0,
            "repeat_seed": 42,
            "model": "qwen3:8b",
            "task_id": "t1",
            "level": 1,
            "gold": "a",
            "pred": "a",
            "correct": 1,
            "task_type": "itbench_finops_root_cause",
            "file_name": "",
            "file_path": "",
            "draft": "",
            "latency_sec": 0.2,
            "steps_used": 2,
            "max_steps_reached": 0,
            "tool_errors": 0,
            "image_sent": 0,
            "attachment_is_image": 0,
            "attachment_is_audio": 0,
            "attachment_is_video": 0,
            "media_to_text_used": 0,
            "read_attachment_used": 0,
            "image_tool_used": 0,
            "audio_tool_used": 0,
            "video_tool_used": 0,
            "failure_reason": "",
            "error": "",
            "eval_meta": "{}",
            "family": "qwen",
            "is_coder": 0,
            "is_vision": 0,
        },
        {
            "run": "all",
            "repeat_idx": 0,
            "repeat_seed": 42,
            "model": "qwen3:8b",
            "task_id": "t2",
            "level": 1,
            "gold": "b",
            "pred": "x",
            "correct": 0,
            "task_type": "itbench_sre_root_cause",
            "file_name": "",
            "file_path": "",
            "draft": "",
            "latency_sec": 0.4,
            "steps_used": 4,
            "max_steps_reached": 1,
            "tool_errors": 1,
            "image_sent": 0,
            "attachment_is_image": 0,
            "attachment_is_audio": 0,
            "attachment_is_video": 0,
            "media_to_text_used": 0,
            "read_attachment_used": 0,
            "image_tool_used": 0,
            "audio_tool_used": 0,
            "video_tool_used": 0,
            "failure_reason": "tool_error+max_steps_reached",
            "error": "",
            "eval_meta": "{}",
            "family": "qwen",
            "is_coder": 0,
            "is_vision": 0,
        },
    ]
    df = pd.DataFrame(rows)

    r.summarize(df=df, out_dir=tmp_path, tag="all", benchmark="ITBench-Lite")

    model_summary = pd.read_csv(tmp_path / "summary_model_all.csv")
    failure_summary = pd.read_csv(tmp_path / "summary_failure_types_all.csv")

    assert {"model", "accuracy", "latency_sec", "n_repeats"}.issubset(
        model_summary.columns
    )
    assert {"failure_reason", "count", "pct_of_failures"}.issubset(
        failure_summary.columns
    )
