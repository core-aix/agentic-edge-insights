# Official Code for Paper "Agentic Performance at the Edge: Insights from Benchmarking" in the AutoEdge Workshop 2026 (co-located with MobiSys)

This folder contains the runnable ITBench-Lite code for evaluating agentic task accuracy across model families and sizes on an Ollama endpoint.

This is a lightweight local reimplementation of the ITBench-Lite experiment path used in the paper. It is intentionally minimal and focused on reproducible FinOps/SRE root-cause experiments instead of official ITBench infrastructure deployment.

Benchmark-specific toolsets:
- ITBench-Lite FinOps: `analyze_finops_cost_anomaly`, `read_file`, `summarize_csv`
- ITBench-Lite SRE: `nl2kubectl`, `query_loki_logs`, `query_jaeger_traces`, `get_alerts`, `get_topology_nodes`, `walk_path`, `get_node_info_by_name`, `get_neighbors`, `check_directly_connected`, `summarize_sre_candidates`, `rank_sre_root_cause_candidates`

These tools are implemented locally in `tools.py` and executed directly by this harness.
Important scope note: these tools are approximations of ITBench tool intent, not strict drop-in replicas of the official backend services. In particular, this harness uses local snapshot/file-based implementations (for example, local log/trace/topology parsing and heuristic ranking) instead of live Loki/Jaeger/Kubernetes service integrations.

## Environment (uv)

```bash
uv sync
```

## Run experiments


Quick run (FinOps):

```bash
uv run python run_benchmark.py \
  --host http://localhost:11434 \
  --benchmark itbench_lite \
  --itbench-domains finops \
  --agent-mode tool \
  --models "qwen2.5-coder:7b,qwen2.5-coder:14b" \
  --out-dir outputs_itbench_lite
```

Run full local benchmark sweep (FinOps + SRE):

```bash
uv run python run_benchmark.py \
  --host http://localhost:11434 \
  --benchmark itbench_lite \
  --itbench-domains finops,sre \
  --all-tasks \
  --agent-mode tool \
  --models "qwen2.5-coder:7b,qwen2.5-coder:14b" \
  --out-dir outputs_itbench_full
```

Concrete paper figure reproduction:

```bash
uv run python run_benchmark.py \
  --host http://localhost:11434 \
  --benchmark itbench_lite \
  --itbench-domains finops,sre \
  --all-tasks \
  --agent-mode tool \
  --models "qwen2.5-coder:7b,qwen2.5-coder:14b,qwen2.5-coder:32b,qwen3-coder:30b" \
  --out-dir outputs_itbench_qwen_family_all_repeat3_temp02_lowconcurrency_nocap_v1
```


## What the script does

- Supports ITBench-Lite execution via `--benchmark itbench_lite` (offline subset from `ibm-research/ITBench-Lite`).
- Runs a tool-enabled ITBench-Lite agent loop (FinOps + SRE repository-aligned tools) or a simple two-pass baseline.
- Computes ITBench-aware accuracy and latency (root-cause matching).
- Produces CSV summaries and figures, including difficulty- and task-type analyses.

