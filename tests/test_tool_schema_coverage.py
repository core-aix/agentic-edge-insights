import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as gta


def test_tool_registry_is_itbench_only():
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
    assert set(gta.TOOLS.keys()) == expected
    assert set(gta.TOOL_SCHEMAS.keys()) == expected


def test_normalize_tool_name_keeps_itbench_aliases():
    assert gta.normalize_tool_name("nl2_kubectl") == "nl2kubectl"
    assert gta.normalize_tool_name("loki_query") == "query_loki_logs"
    assert gta.normalize_tool_name("jaeger_query") == "query_jaeger_traces"
    assert (
        gta.normalize_tool_name("finops_anomaly") == "analyze_finops_cost_anomaly"
    )


def test_toolset_profiles_match_itbench_domains():
    finops = set(gta.toolset_for_profile("native:itbench_lite_finops"))
    sre = set(gta.toolset_for_profile("native:itbench_lite_sre"))
    both = set(gta.toolset_for_profile("native:itbench_lite"))

    assert finops == {"analyze_finops_cost_anomaly", "read_file", "summarize_csv"}
    assert "nl2kubectl" in sre
    assert "analyze_finops_cost_anomaly" not in sre
    assert "analyze_finops_cost_anomaly" not in both
    assert "nl2kubectl" in both
    assert "summarize_sre_candidates" in both


def test_build_tool_prompt_excludes_non_itbench_tools():
    prompt = gta.build_tool_prompt("native:itbench_lite")
    assert "search_web" not in prompt
    assert "fetch_url" not in prompt
    assert "calculator" not in prompt
    assert "analyze_finops_cost_anomaly" not in prompt
    assert "query_loki_logs" in prompt


def test_external_backend_hook_removed():
    assert not hasattr(gta, "_invoke_external_tool_backend")
