import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark as r
import agent as gta
import tools as brt


def test_resolve_scorer_auto_mapping():
    assert r.resolve_scorer("auto") == "itbench_root_cause"
    assert r.resolve_scorer("enhanced") == "enhanced"


def test_score_prediction_itbench_root_cause_from_meta():
    meta = {"expected_names": ["load-generator-pod-1"]}
    assert r.score_prediction(
        "Root cause is load-generator-pod-1", "", "itbench_root_cause", meta
    )
    assert not r.score_prediction("frontend-proxy", "", "itbench_root_cause", meta)


def test_load_itbench_lite_subset_finops(monkeypatch, tmp_path):
    gt_path = tmp_path / "ground_truth.yaml"
    gt_path.write_text(
        "metadata:\n  version: v1\nresource:\n  - resource:\n      name: ABC123\n      root_cause: true\n",
        encoding="utf-8",
    )
    anomaly_path = tmp_path / "anomaly.json"
    anomaly_path.write_text(
        '{"date":"2025-10-13","account_id":"acc"}', encoding="utf-8"
    )
    data_path = tmp_path / "data.csv"
    data_path.write_text("col\n1\n", encoding="utf-8")

    monkeypatch.setattr(
        r,
        "_collect_itbench_lite_scenarios",
        lambda domains: {
            "finops": ["snapshots/finops/v0.1/Scenario-1/ground_truth.yaml"],
            "sre": [],
        },
    )

    def fake_download(repo_id, filename):
        if filename.endswith("ground_truth.yaml"):
            return str(gt_path)
        if filename.endswith("anomaly.json"):
            return str(anomaly_path)
        if filename.endswith("data.csv"):
            return str(data_path)
        raise AssertionError(filename)

    monkeypatch.setattr(r, "_download_itbench_file", fake_download)

    out = r.load_itbench_lite_subset(
        domains={"finops"},
        per_level=3,
        seed=42,
        all_tasks=True,
    )
    assert len(out) == 1
    assert out[0]["task_type"] == "itbench_finops_root_cause"
    assert out[0]["gold"] == "ABC123"
    assert out[0]["eval_meta"]["expected_names"] == ["ABC123"]


def test_benchmark_toolsets_are_distinct():
    it_tools = set(gta.toolset_for_profile("native:itbench_lite"))
    it_finops_tools = set(gta.toolset_for_profile("native:itbench_lite_finops"))
    it_sre_tools = set(gta.toolset_for_profile("native:itbench_lite_sre"))
    assert "search_web" not in it_tools
    assert "fetch_url" not in it_tools
    assert "nl2kubectl" in it_tools
    assert "query_loki_logs" in it_tools
    assert "analyze_finops_cost_anomaly" in it_finops_tools
    assert "nl2kubectl" not in it_finops_tools
    assert "run_python" not in it_finops_tools
    assert "nl2kubectl" in it_sre_tools
    assert "query_loki_logs" in it_sre_tools
    assert "run_python" not in it_sre_tools
    assert "summarize_csv" not in it_sre_tools


def test_repo_tool_names_are_registered_and_callable():
    required = [
        "nl2kubectl",
        "query_loki_logs",
        "query_jaeger_traces",
        "get_alerts",
        "get_topology_nodes",
        "walk_path",
        "get_node_info_by_name",
        "get_neighbors",
        "check_directly_connected",
        "analyze_finops_cost_anomaly",
    ]
    for name in required:
        assert name in gta.TOOLS


def test_finops_anomaly_tool_normalizes_date_format(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text(
        "date,account_id,instance_family,unblended_cost\n"
        "09/22/2025,ACC1,resA,100\n"
        "9/22/25,ACC1,resB,10\n"
        "2025-09-22,ACC1,resC,5\n",
        encoding="utf-8",
    )
    out = brt.run_tool(
        "analyze_finops_cost_anomaly",
        {
            "path": str(p),
            "anomaly_date": "2025-9-22",
            "account_id": "ACC1",
            "top_k": 2,
        },
    )
    data = json.loads(out)
    assert data.get("date_match_found") is True
    assert "resA" in data.get("top_resources_by_cost", [])
