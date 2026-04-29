import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as a


def _write_sre_fixture(base: Path) -> dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)

    events = pd.DataFrame(
        [
            {
                "Body": json.dumps(
                    {
                        "object": {
                            "metadata": {
                                "uid": "uid-a",
                                "name": "checkout-service",
                                "namespace": "prod",
                            }
                        },
                        "message": "error crashloop in checkout-service",
                    }
                )
            }
        ]
    )
    events_path = base / "k8s_events_raw.tsv"
    events.to_csv(events_path, sep="\t", index=False)

    objects = pd.DataFrame(
        [
            {
                "Kind": "Pod",
                "Name": "checkout-service",
                "Body": json.dumps(
                    {
                        "object": {
                            "metadata": {
                                "uid": "uid-a",
                                "name": "checkout-service",
                                "namespace": "prod",
                            }
                        }
                    }
                ),
            }
        ]
    )
    objects_path = base / "k8s_objects_raw.tsv"
    objects.to_csv(objects_path, sep="\t", index=False)

    topology = {
        "nodes": [
            {"id": "n1", "name": "checkout-service", "type": "service"},
            {"id": "n2", "name": "checkout-pod", "type": "pod"},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
    }
    topo_path = base / "topology.json"
    topo_path.write_text(json.dumps(topology), encoding="utf-8")

    trace_path = base / "app_trace.json"
    trace_path.write_text(
        json.dumps([{"service": "checkout", "operation": "GET /health"}]),
        encoding="utf-8",
    )

    log_path = base / "app.log"
    log_path.write_text(
        "2025-01-01 error checkout-service crashloop\n",
        encoding="utf-8",
    )

    alert_path = base / "service_alerts.json"
    alert_path.write_text(json.dumps({"alerts": ["checkout"]}), encoding="utf-8")

    return {
        "events": events_path,
        "objects": objects_path,
        "topology": topo_path,
        "trace": trace_path,
        "log": log_path,
        "alert": alert_path,
    }


def test_all_remaining_tools_have_coverage(monkeypatch, tmp_path):
    fixture = _write_sre_fixture(tmp_path / "scenario")

    finops_csv = tmp_path / "finops.csv"
    finops_csv.write_text(
        "date,account_id,instance_family,unblended_cost\n"
        "2025-09-22,ACC1,resA,100\n"
        "2025-09-22,ACC1,resB,10\n",
        encoding="utf-8",
    )

    sample_txt = tmp_path / "sample.txt"
    sample_txt.write_text("hello world", encoding="utf-8")

    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text("name,cost\na,1\nb,2\n", encoding="utf-8")

    monkeypatch.setenv("ITBENCH_SCENARIO_DIR", str((tmp_path / "scenario").resolve()))
    monkeypatch.setenv("ITBENCH_TOPOLOGY_PATH", str(fixture["topology"].resolve()))

    expected = {
        "read_file",
        "summarize_csv",
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
        "summarize_sre_candidates",
        "rank_sre_root_cause_candidates",
    }
    assert set(a.TOOLS.keys()) == expected

    assert "hello world" in a.TOOLS["read_file"](path=str(sample_txt))

    csv_summary = json.loads(a.TOOLS["summarize_csv"](path=str(sample_csv), top_k=2))
    assert "columns" in csv_summary

    finops = json.loads(
        a.TOOLS["analyze_finops_cost_anomaly"](
            path=str(finops_csv), anomaly_date="2025-09-22", account_id="ACC1", top_k=2
        )
    )
    assert finops.get("top_resources_by_cost", [""])[0] == "resA"

    assert "checkout-service" in a.TOOLS["nl2kubectl"](
        nl_query="inspect kubernetes pod"
    )

    loki = json.loads(
        a.TOOLS["query_loki_logs"](query="error checkout-service", limit=10)
    )
    assert isinstance(loki, list) and len(loki) >= 1

    traces = json.loads(
        a.TOOLS["query_jaeger_traces"](
            service="checkout", operation="GET /health", limit=5
        )
    )
    assert isinstance(traces, list) and len(traces) >= 1

    alerts = json.loads(a.TOOLS["get_alerts"]())
    assert alerts.get("alerts_files")

    topo_nodes = json.loads(a.TOOLS["get_topology_nodes"]())
    assert "n1" in topo_nodes.get("nodes", [])

    path_walk = json.loads(
        a.TOOLS["walk_path"](
            topology=str(fixture["topology"]),
            start_id="n1",
            start_node_type="service",
            target_node_type="pod",
        )
    )
    assert "n2" in path_walk.get("targets", [])

    node_info = json.loads(
        a.TOOLS["get_node_info_by_name"](
            topology=str(fixture["topology"]), node_name="checkout-service"
        )
    )
    assert node_info.get("id") == "n1"

    neighbors = json.loads(
        a.TOOLS["get_neighbors"](topology=str(fixture["topology"]), node_id="n1")
    )
    assert "n2" in neighbors.get("neighbors", [])

    connected = json.loads(
        a.TOOLS["check_directly_connected"](
            topology=str(fixture["topology"]), node_id1="n1", node_id2="n2"
        )
    )
    assert connected.get("connected") is True

    sre_summary = json.loads(a.TOOLS["summarize_sre_candidates"](scenario_dir=str(tmp_path / "scenario")))
    assert "candidate_entity_names" in sre_summary

    sre_rank = json.loads(
        a.TOOLS["rank_sre_root_cause_candidates"](scenario_dir=str(tmp_path / "scenario"))
    )
    assert "ranked_candidates" in sre_rank
