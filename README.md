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

## Dataset setup

The benchmark now supports loading a frozen local dataset copy via `--dataset-path`.
Use this for paper-reproducible runs.

As we have noted that the official ITBench data has been changing over time, we provide a snapshot of the data that we used for the paper's experiments, which is downloadable at https://www.dropbox.com/scl/fi/dtgg8u4nsqh44vjqxspnm/ITBench-Lite-paper-snapshot-2026.tar.gz?rlkey=cianfxvhrmfk66er3kmlbueuw&st=tilc2q8k&dl=0

The extracted dataset path can be provided to the `--dataset-path` option when running the code.

Expected extracted folder name:
- `ITBench-Lite-paper-snapshot-2026`

Expected structure:
- `ITBench-Lite-paper-snapshot-2026/snapshots/finops/...`
- `ITBench-Lite-paper-snapshot-2026/snapshots/sre/...`

Optional integrity check:

```bash
cd /path/to/ITBench-Lite-paper-snapshot-2026
shasum -a 256 -c CHECKSUMS.sha256
```

License note:
- Upstream dataset license is Apache-2.0 (`ibm-research/ITBench-Lite`).
- Keep attribution and license notice files when redistributing this frozen copy.


## Run experiments


Quick run (FinOps):

```bash
uv run python run_benchmark.py \
  --host http://localhost:11434 \
  --benchmark itbench_lite \
  --itbench-domains finops \
  --agent-mode tool \
  --dataset-path /path/to/ITBench-Lite-paper-snapshot-2026 \
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
  --dataset-path /path/to/ITBench-Lite-paper-snapshot-2026 \
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
  --dataset-path /path/to/ITBench-Lite-paper-snapshot-2026 \
  --models "qwen2.5-coder:7b,qwen2.5-coder:14b,qwen2.5-coder:32b,qwen3-coder:30b" \
  --out-dir outputs_itbench_qwen_family_all_repeat3_temp02_lowconcurrency_nocap_v1
```


## What the script does

- Supports ITBench-Lite execution via `--benchmark itbench_lite` (offline subset from `ibm-research/ITBench-Lite`).
- Supports local frozen data loading via `--dataset-path` (otherwise loads from Hugging Face Hub).
- Runs a tool-enabled ITBench-Lite agent loop (FinOps + SRE repository-aligned tools) or a simple two-pass baseline.
- Computes ITBench-aware accuracy and latency (root-cause matching).
- Produces CSV summaries and figures, including difficulty- and task-type analyses.

## Known Issues
- There are 4 scenarios (out of 35) in the SRE data that have a different schema and cannot be properly loaded by our current code. Those 4 scenarios are currently ignored.