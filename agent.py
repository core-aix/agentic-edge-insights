#!/usr/bin/env python3
import json
import os
import re
import time
from pathlib import Path

import requests
from tools import (
    TOOLS,
    TOOL_SCHEMAS,
    build_tool_prompt,
    configure_docling_quiet,
    normalize_tool_name,
    read_file,
    tool_schema_hint,
    toolset_for_profile,
    _parse_tool_prompt_profile,
    _truncate,
)


TOOL_CACHE: dict[str, str] = {}
TOOL_CACHE_LOADED = False
TOOL_CACHE_FILE_OVERRIDE: str | None = None
TOOL_CACHE_MODE_OVERRIDE: str | None = None
_MODEL_TEMPLATE_CACHE: dict[str, str] = {}


def _get_model_template(host: str, model: str) -> str:
    key = f"{host}|{model}"
    if key in _MODEL_TEMPLATE_CACHE:
        return _MODEL_TEMPLATE_CACHE[key]
    tmpl = ""
    try:
        resp = requests.post(
            f"{str(host).rstrip('/')}/api/show",
            json={"model": model},
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            tmpl = str(data.get("template", "") or "")
    except Exception:  # noqa: BLE001
        tmpl = ""
    _MODEL_TEMPLATE_CACHE[key] = tmpl
    return tmpl


def _preferred_tool_result_role(host: str, model: str) -> str:
    t = _get_model_template(host, model).lower()
    # If template explicitly supports a tool role, use it. Otherwise,
    # use user role for broad compatibility with chat templates.
    tool_patterns = [
        'role "tool"',
        "role 'tool'",
        "<|im_start|>tool",
        "<start_of_turn>tool",
        "<tool>",
    ]
    if any(p in t for p in tool_patterns):
        return "tool"
    return "user"


def configure_tool_cache(mode: str | None = None, file_path: str | None = None) -> None:
    global TOOL_CACHE_FILE_OVERRIDE
    global TOOL_CACHE_MODE_OVERRIDE
    global TOOL_CACHE_LOADED
    TOOL_CACHE_MODE_OVERRIDE = None if mode is None else str(mode).strip().lower()
    TOOL_CACHE_FILE_OVERRIDE = None if file_path is None else str(file_path)
    TOOL_CACHE_LOADED = False


def _tool_cache_file() -> str:
    if TOOL_CACHE_FILE_OVERRIDE is not None:
        return TOOL_CACHE_FILE_OVERRIDE
    return os.environ.get("BENCHMARK_TOOL_CACHE_FILE", "")


def _tool_cache_mode() -> str:
    # auto: read cache if exists; write new entries.
    # replay: read cache only, never fetch network.
    # off: disable cache.
    if TOOL_CACHE_MODE_OVERRIDE is not None:
        return TOOL_CACHE_MODE_OVERRIDE
    return os.environ.get("BENCHMARK_TOOL_CACHE_MODE", "auto").strip().lower()


def _cache_key(tool: str, args: dict) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


def _load_tool_cache_once() -> None:
    global TOOL_CACHE_LOADED
    if TOOL_CACHE_LOADED:
        return
    TOOL_CACHE_LOADED = True
    mode = _tool_cache_mode()
    if mode == "off":
        return
    path = _tool_cache_file()
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        TOOL_CACHE.update(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass


def _save_tool_cache() -> None:
    mode = _tool_cache_mode()
    if mode == "off":
        return
    path = _tool_cache_file()
    if not path:
        return
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(TOOL_CACHE, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def cache_get(tool: str, args: dict) -> str | None:
    _load_tool_cache_once()
    return TOOL_CACHE.get(_cache_key(tool, args))


def cache_set(tool: str, args: dict, value: str) -> None:
    _load_tool_cache_once()
    TOOL_CACHE[_cache_key(tool, args)] = str(value)
    _save_tool_cache()


def extract_declared_tool_name(text: str) -> str:
    s = str(text or "")
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', s, flags=re.IGNORECASE)
    if m:
        return normalize_tool_name(m.group(1))
    return ""


def adapt_tool_args(tool: str, args: dict, attachment_path: str) -> dict:
    args = dict(args or {})

    def pick(*names: str, default: object = "") -> object:
        for n in names:
            if n in args and args[n] not in (None, ""):
                return args[n]
        return default

    def to_int(value: object, fallback: int) -> int:
        try:
            return int(str(value))
        except Exception:  # noqa: BLE001
            return fallback

    def first_nonempty_value(default: object = "") -> object:
        for v in args.values():
            if v not in (None, ""):
                return v
        return default

    def normalize_path_token(value: object) -> object:
        s = str(value or "").strip().lower()
        if s in {"image", "attachment", "file", "media", "audio", "video"}:
            return attachment_path
        return value

    if tool in {"read_attachment"}:
        return {
            "path": normalize_path_token(
                pick("path", "file_path", default=attachment_path)
            )
        }
    if tool == "read_file":
        return {
            "path": normalize_path_token(
                pick(
                    "path",
                    "file_path",
                    "filename",
                    "file",
                    "file_id",
                    "url",
                    default=attachment_path,
                )
            )
        }
    if tool == "summarize_csv":
        return {
            "path": normalize_path_token(
                pick("path", "file_path", "filename", "file", default=attachment_path)
            ),
            "group_by": pick("group_by", "group", "key", default=""),
            "value_column": pick(
                "value_column", "value", "metric", "column", default=""
            ),
            "top_k": to_int(pick("top_k", "k", default="10"), 10),
        }
    if tool == "analyze_finops_cost_anomaly":
        return {
            "path": normalize_path_token(
                pick("path", "file_path", "filename", "file", default=attachment_path)
            ),
            "anomaly_date": pick(
                "anomaly_date", "date", "spike_date", "target_date", default=""
            ),
            "account_id": pick("account_id", "account", "acct", default=""),
            "top_k": to_int(pick("top_k", "k", default="5"), 5),
        }
    if tool == "nl2kubectl":
        return {
            "nl_query": pick(
                "nl_query", "query", "command", default=first_nonempty_value("")
            )
        }
    if tool == "query_loki_logs":
        return {
            "query": pick("query", "q", default=first_nonempty_value("")),
            "limit": to_int(pick("limit", default="100"), 100),
            "start": pick("start", default=""),
            "end": pick("end", default=""),
            "since": pick("since", default=""),
            "step": pick("step", default=""),
            "interval": pick("interval", default=""),
            "direction": pick("direction", default="backward"),
        }
    if tool == "query_jaeger_traces":
        return {
            "service": pick("service", "service_name", default=""),
            "operation": pick("operation", default=""),
            "start_time": to_int(pick("start_time", default="0"), 0),
            "end_time": to_int(pick("end_time", default="0"), 0),
            "limit": to_int(pick("limit", default="20"), 20),
            "error_traces_only": bool(pick("error_traces_only", default=False)),
        }
    if tool in {
        "walk_path",
        "get_node_info_by_name",
        "get_neighbors",
        "check_directly_connected",
    }:
        return dict(args)
    if tool in {
        "get_alerts",
        "get_topology_nodes",
    }:
        return {}
    if tool in {
        "summarize_sre_candidates",
        "rank_sre_root_cause_candidates",
    }:
        scenario_dir = ""
        if attachment_path:
            try:
                scenario_dir = str(Path(attachment_path).parent)
            except Exception:
                scenario_dir = ""
        return {"scenario_dir": scenario_dir}
    return args


def parse_tool_or_answer(text: str) -> tuple[str, dict | str | None]:
    def normalize_payload(data: dict) -> dict | None:
        if not isinstance(data, dict):
            return None
        if "tool" in data:
            return {
                "tool": normalize_tool_name(str(data.get("tool", ""))),
                "args": data.get("args", {}) or {},
            }
        if "name" in data and ("arguments" in data or "args" in data):
            args = data.get("arguments", data.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            return {
                "tool": normalize_tool_name(str(data.get("name", ""))),
                "args": args or {},
            }
        fn = data.get("function")
        if isinstance(fn, dict) and "name" in fn:
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            return {
                "tool": normalize_tool_name(str(fn.get("name", ""))),
                "args": args or {},
            }
        calls = data.get("tool_calls")
        if isinstance(calls, list) and calls:
            c0 = calls[0]
            if isinstance(c0, dict):
                fn = c0.get("function", c0)
                if isinstance(fn, dict) and "name" in fn:
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:  # noqa: BLE001
                            args = {}
                    return {
                        "tool": normalize_tool_name(str(fn.get("name", ""))),
                        "args": args or {},
                    }
        return None

    text = text.strip()
    if text.startswith("FINAL_ANSWER:"):
        return "final", text.split("FINAL_ANSWER:", 1)[1].strip()
    match = re.search(
        r"final answer\s*[:\-]\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return "final", match.group(1).strip()
    tag_match = re.search(
        r"(?is)<(?:tool_call|function_call)>(.*?)</(?:tool_call|function_call)>", text
    )
    if tag_match:
        blob = tag_match.group(1).strip()
        try:
            data = json.loads(blob)
            n = normalize_payload(data)
            if n:
                return "tool", n
        except Exception:  # noqa: BLE001
            pass

    marker = re.search(r"(?i)\btool_call\s*:", text)
    if marker and marker.start() == 0:
        payload = text[marker.end() :].strip()
        payload = payload.strip("`")
        try:
            data = json.loads(payload)
            n = normalize_payload(data)
            if n:
                return "tool", n
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(payload[start : end + 1])
                    n = normalize_payload(data)
                    if n:
                        return "tool", n
                except json.JSONDecodeError:
                    pass
            return "invalid", None
    # TOOL_CALL can appear after preamble text; extract first JSON object after marker.
    if marker or re.search(r"(?i)\btool_call\s*:", text):
        try:
            m2 = re.search(r"(?i)\btool_call\s*:", text)
            if not m2:
                raise ValueError("tool_call marker not found")
            payload = text[m2.end() :]
            start = payload.find("{")
            if start != -1:
                brace = 0
                end = -1
                for i, ch in enumerate(payload[start:], start=start):
                    if ch == "{":
                        brace += 1
                    elif ch == "}":
                        brace -= 1
                        if brace == 0:
                            end = i
                            break
                if end != -1:
                    obj_txt = payload[start : end + 1]
                    data = json.loads(obj_txt)
                    n = normalize_payload(data)
                    if n:
                        return "tool", n
        except Exception:  # noqa: BLE001
            pass
    # Accept raw JSON tool call without prefix.
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            n = normalize_payload(data)
            if n:
                return "tool", n
        except json.JSONDecodeError:
            pass
    # Some models ignore strict format; treat non-empty text as final answer.
    if text and ("TOOL_CALL" not in text.upper()):
        return "final", text
    return "invalid", None


def is_tool_error_output(obs: str) -> bool:
    s = str(obs or "").strip().lower()
    if not s:
        return True
    markers = [
        "tool error",
        "tool_exception",
        "unknown tool",
        "fetch_error",
        "media-to-text error",
        "audio-to-text error",
        "video-to-text error",
        "audio probe error",
        "video probe error",
        "youtube probe error",
        "yt-dlp error",
        "read file error",
        "conversion failed",
        "http 4",
        "http 5",
        "not found",
        "forbidden",
        "unauthorized",
        "invalid media path",
        "unsupported",
        "cannot open",
    ]
    return any(m in s for m in markers)


def is_rate_limited_output(obs: str) -> bool:
    s = str(obs or "").strip().lower()
    markers = [
        "too many requests",
        "rate limit",
        "rate-limit",
        "http 429",
        "status code 429",
    ]
    return any(m in s for m in markers)


def normalize_tool_observation(tool: str, obs: str) -> str:
    return str(obs or "")


def _retry_args(
    tool: str, adapted_args: dict, attachment_path: str, attempt: int
) -> dict:
    args = dict(adapted_args or {})
    if tool == "read_file":
        p = str(args.get("path", "") or "").strip()
        if attachment_path:
            ap = Path(attachment_path)
            should_replace = (not p) or (not Path(p).exists())
            if should_replace and ap.exists() and not ap.is_dir():
                args["path"] = attachment_path
    return args


def execute_tool_with_retries(
    tool: str,
    args: dict,
    attachment_path: str,
    dynamic_tools: dict,
    max_attempts: int = 2,
) -> tuple[str, int]:
    adapted_args = adapt_tool_args(tool, args, attachment_path)
    tool_errors = 0
    obs = ""

    mode = _tool_cache_mode()
    cached = cache_get(tool, adapted_args)
    if cached is not None and not is_tool_error_output(cached):
        return cached, 0
    if mode == "replay":
        # No cache hit in replay-only mode.
        return "TOOL_ERROR: cache miss in replay mode", 1

    effective_attempts = max(max_attempts, 2) + 3
    for attempt in range(effective_attempts):
        try:
            obs = dynamic_tools[tool](**adapted_args)
        except Exception as exc:  # noqa: BLE001
            obs = f"TOOL_EXCEPTION: {exc}"

        obs = normalize_tool_observation(tool, obs)

        if not is_tool_error_output(obs):
            out = str(obs)
            cache_set(tool, adapted_args, out)
            return out, tool_errors

        if is_rate_limited_output(obs):
            wait_sec = min(20, 2 * (attempt + 1))
            time.sleep(wait_sec)
            continue

        tool_errors += 1
        adapted_args = _retry_args(tool, adapted_args, attachment_path, attempt)

    return str(obs), tool_errors


def force_final_answer(
    ollama_chat_fn, host: str, model: str, raw_text: str, timeout: int
) -> str:
    prompt = (
        "Convert the following assistant output into ONLY the final short answer. "
        "Do not include reasoning or explanation.\n\n"
        f"Assistant output:\n{raw_text}\n\n"
        "Return: FINAL_ANSWER: <answer>"
    )
    out = ollama_chat_fn(
        host,
        model,
        [
            {"role": "system", "content": "You format final answers only."},
            {"role": "user", "content": prompt},
        ],
        timeout,
    )
    kind, payload = parse_tool_or_answer(out)
    if kind == "final":
        return str(payload)
    return out.strip()


def _best_effort_final_answer(
    ollama_chat_fn,
    host: str,
    model: str,
    question: str,
    trace: list[str],
    timeout: int,
) -> str:
    transcript = "\n---\n".join([t for t in trace if str(t).strip()][-12:])
    prompt = (
        "Produce the final short answer for this ITBench-Lite question. "
        "Do not call tools and do not include reasoning.\n\n"
        f"Question:\n{question}\n\n"
        f"Recent transcript:\n{_truncate(transcript, 12000)}\n\n"
        "Return exactly: FINAL_ANSWER: <answer>"
    )
    out = ollama_chat_fn(
        host,
        model,
        [
            {"role": "system", "content": "No tools. Output final answer only."},
            {"role": "user", "content": prompt},
        ],
        timeout,
    )
    kind, payload = parse_tool_or_answer(out)
    if kind == "final":
        ans = str(payload).strip()
        if ans and "TOOL_CALL" not in ans.upper():
            return ans

    for block in reversed(trace):
        txt = str(block or "").strip()
        if not txt or txt.startswith("TOOL_RESULT["):
            continue
        if txt.upper().startswith("TOOL_CALL:"):
            continue
        cleaned = re.sub(
            r"```tool_(?:call|code)[\s\S]*?```", "", txt, flags=re.IGNORECASE
        )
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines() if "TOOL_CALL" not in ln.upper()
        ).strip()
        if not cleaned:
            continue
        k2, p2 = parse_tool_or_answer(cleaned)
        if k2 == "final":
            out2 = str(p2).strip()
        else:
            out2 = cleaned
        out2 = " ".join(out2.split())
        if out2:
            return _truncate(out2, 500)

    return "UNKNOWN"


def solve_with_tools(
    ollama_chat_fn,
    host: str,
    model: str,
    question: str,
    attachment_path: str,
    attachment_image_b64: str,
    timeout: int,
    max_steps: int = 8,
    tool_prompt_profile: str = "native",
):
    allowed_tools = set(toolset_for_profile(tool_prompt_profile))
    dynamic_tools = {k: v for k, v in TOOLS.items() if k in allowed_tools}
    media_to_text_used = 0
    read_attachment_used = 0
    image_tool_used = 0
    audio_tool_used = 0
    video_tool_used = 0

    # Provide per-task defaults for ITBench repository tools.
    _base_profile, _bench = _parse_tool_prompt_profile(tool_prompt_profile)
    if _bench.startswith("itbench_lite") and attachment_path:
        ap = Path(attachment_path)
        scenario_dir = str(ap.parent) if ap.exists() else ""
        if scenario_dir:
            os.environ["ITBENCH_SCENARIO_DIR"] = scenario_dir
            topo = None
            for cand in Path(scenario_dir).glob("*.json"):
                if "topology" in cand.name.lower():
                    topo = str(cand)
                    break
            if topo:
                os.environ["ITBENCH_TOPOLOGY_PATH"] = topo

    def _remap_unknown_tool_for_benchmark(
        name: str, raw_args: dict
    ) -> tuple[str, dict] | None:
        n = str(name or "").strip().lower()
        a = dict(raw_args or {})
        if _bench == "itbench_lite_finops":
            if n in {"run_python"}:
                path = str(
                    a.get("path") or a.get("file_path") or attachment_path or ""
                ).strip()
                return (
                    "analyze_finops_cost_anomaly",
                    {
                        "path": path,
                        "anomaly_date": str(
                            a.get("date") or a.get("anomaly_date") or ""
                        ),
                        "account_id": str(
                            a.get("account_id") or a.get("account") or ""
                        ),
                    },
                )
        if _bench == "itbench_lite_sre":
            if n in {"read_file", "summarize_csv", "run_python"}:
                path = str(
                    a.get("path") or a.get("file_path") or attachment_path or ""
                ).strip()
                if path:
                    return ("nl2kubectl", {"nl_query": f"inspect file {path}"})
                return (
                    "nl2kubectl",
                    {"nl_query": "inspect kubernetes events and objects"},
                )
        return None

    def read_attachment(path: str = "") -> str:
        target = str(path or attachment_path or "")
        if not target:
            return "No attachment for this task"
        return read_file(target)

    dynamic_tools["read_attachment"] = read_attachment
    attachment_preview = "No attachment"
    if attachment_path and not attachment_image_b64:
        # Non-vision path: provide a small attachment preview to bootstrap tool planning.
        attachment_preview = read_attachment()

    image_context = ""
    if attachment_image_b64:
        image_context = (
            "An image attachment is included in this message. "
            "Inspect the image directly using vision capabilities first. "
            "Do not rely on OCR-only conversion unless absolutely necessary."
        )

    user_msg = {
        "role": "user",
        "content": (
            f"Question:\n{question}\n\n"
            "Attachment available: "
            f"{'yes' if attachment_path else 'no'}\n"
            "Use provided image bytes directly when present; do not rely on local file paths.\n"
            f"Attachment preview:\n{_truncate(attachment_preview, 5000)}\n\n"
            f"{image_context}\n\n"
            "Use tools as needed, then return FINAL_ANSWER."
        ),
        "images": [],
    }
    if attachment_image_b64:
        user_msg["images"] = [attachment_image_b64]

    messages = [
        {"role": "system", "content": build_tool_prompt(tool_prompt_profile)},
        user_msg,
    ]
    trace = []
    tool_errors = 0
    step_count = 0
    tool_call_counts: dict[str, int] = {}
    tool_history: list[str] = []
    invalid_format_count = 0
    format_violation_count = 0
    tool_result_role = _preferred_tool_result_role(host, model)

    def tool_sig(name: str, raw_args: dict) -> str:
        adapted = adapt_tool_args(name, raw_args, attachment_path)
        return f"{name}:{json.dumps(adapted, sort_keys=True, ensure_ascii=False)}"

    for _ in range(max_steps):
        step_count += 1
        reply = ollama_chat_fn(host, model, messages, timeout)
        kind, payload = parse_tool_or_answer(reply)
        trace.append(reply)
        if kind == "final":
            invalid_format_count = 0
            if not reply.strip().upper().startswith("FINAL_ANSWER:"):
                format_violation_count += 1
                if format_violation_count >= 3:
                    forced = force_final_answer(
                        ollama_chat_fn=ollama_chat_fn,
                        host=host,
                        model=model,
                        raw_text=reply,
                        timeout=timeout,
                    )
                    return (
                        "\n---\n".join(trace),
                        forced,
                        {
                            "steps_used": step_count,
                            "max_steps_reached": 1,
                            "tool_errors": tool_errors,
                            "media_to_text_used": media_to_text_used,
                            "read_attachment_used": read_attachment_used,
                            "image_tool_used": image_tool_used,
                            "audio_tool_used": audio_tool_used,
                            "video_tool_used": video_tool_used,
                        },
                    )
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": "user",
                        "content": 'Format violation. Reply EXACTLY in one of: FINAL_ANSWER: <answer> OR TOOL_CALL: {"tool":"...","args":{...}}',
                    }
                )
                continue
            format_violation_count = 0
            final_payload = str(payload)
            return (
                "\n---\n".join(trace[:-1]),
                final_payload,
                {
                    "steps_used": step_count,
                    "max_steps_reached": 0,
                    "tool_errors": tool_errors,
                    "media_to_text_used": media_to_text_used,
                    "read_attachment_used": read_attachment_used,
                    "image_tool_used": image_tool_used,
                    "audio_tool_used": audio_tool_used,
                    "video_tool_used": video_tool_used,
                },
            )
        if kind == "tool":
            invalid_format_count = 0
            format_violation_count = 0
            total_calls_in_reply = len(re.findall(r"(?i)\btool_call\s*:", reply))
            if total_calls_in_reply > 1:
                parse_obs = (
                    "Tool error: multiple TOOL_CALL blocks in one response. "
                    "Only one tool call is allowed per turn."
                )
                tool_errors += 1
                trace.append(f"TOOL_RESULT[parse_error]:\n{parse_obs}")
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": tool_result_role,
                        "content": (
                            "TOOL_RESULT indicates parsing failure:\n"
                            f"{parse_obs}\n"
                            f"{tool_schema_hint(extract_declared_tool_name(reply))}\n\n"
                            'Retry with exactly one tool call JSON: TOOL_CALL: {"tool":"...","args":{...}}'
                        ),
                    }
                )
                continue
            if not isinstance(payload, dict):
                tool = ""
                args = {}
            else:
                tool = normalize_tool_name(payload.get("tool", ""))
                args = payload.get("args", {}) or {}

            # Loop guard: avoid repeated identical tool call spirals.
            sig = tool_sig(tool, args)
            call_n = tool_call_counts.get(sig, 0) + 1
            tool_call_counts[sig] = call_n
            if call_n >= 3:
                loop_obs = (
                    "Tool error: repeated identical TOOL_CALL detected. "
                    "Do not repeat the same tool+args; use a different step or provide FINAL_ANSWER."
                )
                tool_errors += 1
                trace.append(f"TOOL_RESULT[loop_guard]:\n{loop_obs}")
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": tool_result_role,
                        "content": (
                            "TOOL_RESULT indicates loop prevention:\n"
                            f"{loop_obs}\n\n"
                            "Stop repeating the same call. Either call a different tool/args or output FINAL_ANSWER: <answer>."
                        ),
                    }
                )
                continue

            if tool not in dynamic_tools:
                remapped = _remap_unknown_tool_for_benchmark(tool, args)
                if remapped and remapped[0] in dynamic_tools:
                    tool, args = remapped
                    obs, retries_failed = execute_tool_with_retries(
                        tool=tool,
                        args=args,
                        attachment_path=attachment_path,
                        dynamic_tools=dynamic_tools,
                        max_attempts=2,
                    )
                    tool_errors += int(retries_failed)
                else:
                    obs = f"Unknown tool: {tool}\n{tool_schema_hint(tool)}"
                    tool_errors += 1
            else:
                obs, retries_failed = execute_tool_with_retries(
                    tool=tool,
                    args=args,
                    attachment_path=attachment_path,
                    dynamic_tools=dynamic_tools,
                    max_attempts=2,
                )
                tool_errors += int(retries_failed)

            tool_history.append(tool)

            if tool == "read_attachment":
                read_attachment_used += 1
            if tool in {"read_file", "read_attachment"}:
                p = str(
                    adapt_tool_args(tool, args, attachment_path).get(
                        "path", attachment_path
                    )
                ).lower()
                if p.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                    image_tool_used += 1
                if p.endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")):
                    audio_tool_used += 1
                if p.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")):
                    video_tool_used += 1
            trace.append(f"TOOL_RESULT[{tool}]:\n{_truncate(obs, 2000)}")

            messages.append({"role": "assistant", "content": reply})
            if is_tool_error_output(obs):
                messages.append(
                    {
                        "role": tool_result_role,
                        "content": (
                            "TOOL_RESULT indicates failure:\n"
                            f"{_truncate(obs, 3000)}\n\n"
                            f"{tool_schema_hint(tool)}\n"
                            "Do not output this failure message as final answer. "
                            "Try a different tool or a different argument schema."
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": tool_result_role,
                        "content": (f"TOOL_RESULT[{tool}]:\n{_truncate(obs, 10000)}"),
                    }
                )
            continue

        if re.search(r"(?i)\btool_call\s*:", reply):
            invalid_format_count = 0
            format_violation_count += 1
            parse_obs = "Tool error: malformed TOOL_CALL JSON or unsupported structure"
            tool_errors += 1
            trace.append(f"TOOL_RESULT[parse_error]:\n{parse_obs}")
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": tool_result_role,
                    "content": (
                        "TOOL_RESULT indicates parsing failure:\n"
                        f"{parse_obs}\n"
                        f"{tool_schema_hint(extract_declared_tool_name(reply))}\n\n"
                        'Retry with valid TOOL_CALL JSON: TOOL_CALL: {"tool":"...","args":{...}}'
                    ),
                }
            )
            if format_violation_count >= 3:
                forced = _best_effort_final_answer(
                    ollama_chat_fn=ollama_chat_fn,
                    host=host,
                    model=model,
                    question=question,
                    trace=trace,
                    timeout=timeout,
                )
                return (
                    "\n---\n".join(trace),
                    forced,
                    {
                        "steps_used": step_count,
                        "max_steps_reached": 1,
                        "tool_errors": tool_errors,
                        "media_to_text_used": media_to_text_used,
                        "read_attachment_used": read_attachment_used,
                        "image_tool_used": image_tool_used,
                        "audio_tool_used": audio_tool_used,
                        "video_tool_used": video_tool_used,
                    },
                )
            continue

        invalid_format_count += 1
        format_violation_count += 1
        if invalid_format_count >= 3:
            forced = _best_effort_final_answer(
                ollama_chat_fn=ollama_chat_fn,
                host=host,
                model=model,
                question=question,
                trace=trace,
                timeout=timeout,
            )
            return (
                "\n---\n".join(trace),
                forced,
                {
                    "steps_used": step_count,
                    "max_steps_reached": 1,
                    "tool_errors": tool_errors,
                    "media_to_text_used": media_to_text_used,
                    "read_attachment_used": read_attachment_used,
                    "image_tool_used": image_tool_used,
                    "audio_tool_used": audio_tool_used,
                    "video_tool_used": video_tool_used,
                },
            )

        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Invalid format. Respond with either:\n"
                    'TOOL_CALL: {"tool":"...","args":{...}}\n'
                    "or FINAL_ANSWER: <answer>"
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": "Stop tool use now. Provide FINAL_ANSWER: <answer> based on gathered evidence.",
        }
    )
    last = ollama_chat_fn(host, model, messages, timeout)
    trace.append(last)
    kind, payload = parse_tool_or_answer(last)

    # Never end in the middle of a tool call: execute it and request final answer once more.
    if kind == "tool":
        total_calls_in_last = len(re.findall(r"(?i)\btool_call\s*:", last))
        if total_calls_in_last > 1:
            parse_obs = (
                "Tool error: multiple TOOL_CALL blocks in one response. "
                "Only one tool call is allowed per turn."
            )
            tool_errors += 1
            trace.append(f"TOOL_RESULT[parse_error]:\n{parse_obs}")
            messages.append({"role": "assistant", "content": last})
            messages.append(
                {
                    "role": tool_result_role,
                    "content": (
                        "TOOL_RESULT indicates parsing failure:\n"
                        f"{parse_obs}\n"
                        f"{tool_schema_hint(extract_declared_tool_name(last))}\n\n"
                        "Now provide FINAL_ANSWER: <answer> only."
                    ),
                }
            )
            last2 = ollama_chat_fn(host, model, messages, timeout)
            trace.append(last2)
            kind2, payload2 = parse_tool_or_answer(last2)
            if kind2 == "final":
                return (
                    "\n---\n".join(trace[:-1]),
                    str(payload2),
                    {
                        "steps_used": step_count,
                        "max_steps_reached": 1,
                        "tool_errors": tool_errors,
                        "media_to_text_used": media_to_text_used,
                        "read_attachment_used": read_attachment_used,
                        "image_tool_used": image_tool_used,
                        "audio_tool_used": audio_tool_used,
                        "video_tool_used": video_tool_used,
                    },
                )
            return (
                "\n---\n".join(trace),
                _best_effort_final_answer(
                    ollama_chat_fn=ollama_chat_fn,
                    host=host,
                    model=model,
                    question=question,
                    trace=trace,
                    timeout=timeout,
                ),
                {
                    "steps_used": step_count,
                    "max_steps_reached": 1,
                    "tool_errors": tool_errors,
                    "media_to_text_used": media_to_text_used,
                    "read_attachment_used": read_attachment_used,
                    "image_tool_used": image_tool_used,
                    "audio_tool_used": audio_tool_used,
                    "video_tool_used": video_tool_used,
                },
            )
        if not isinstance(payload, dict):
            tool = ""
            args = {}
        else:
            tool = normalize_tool_name(payload.get("tool", ""))
            args = payload.get("args", {}) or {}

        if tool not in dynamic_tools:
            remapped = _remap_unknown_tool_for_benchmark(tool, args)
            if remapped and remapped[0] in dynamic_tools:
                tool, args = remapped
                obs, retries_failed = execute_tool_with_retries(
                    tool=tool,
                    args=args,
                    attachment_path=attachment_path,
                    dynamic_tools=dynamic_tools,
                    max_attempts=2,
                )
                tool_errors += int(retries_failed)
            else:
                obs = f"Unknown tool: {tool}\n{tool_schema_hint(tool)}"
                tool_errors += 1
        else:
            obs, retries_failed = execute_tool_with_retries(
                tool=tool,
                args=args,
                attachment_path=attachment_path,
                dynamic_tools=dynamic_tools,
                max_attempts=2,
            )
            tool_errors += int(retries_failed)

        trace.append(f"TOOL_RESULT[{tool or 'parse_error'}]:\n{_truncate(obs, 2000)}")
        messages.append({"role": "assistant", "content": last})
        messages.append(
            {
                "role": tool_result_role,
                "content": (
                    f"TOOL_RESULT[{tool or 'parse_error'}]:\n{_truncate(obs, 10000)}"
                ),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "Now provide FINAL_ANSWER: <answer> only.",
            }
        )
        last2 = ollama_chat_fn(host, model, messages, timeout)
        trace.append(last2)
        kind2, payload2 = parse_tool_or_answer(last2)
        if kind2 == "final":
            return (
                "\n---\n".join(trace[:-1]),
                str(payload2),
                {
                    "steps_used": step_count,
                    "max_steps_reached": 1,
                    "tool_errors": tool_errors,
                    "media_to_text_used": media_to_text_used,
                    "read_attachment_used": read_attachment_used,
                    "image_tool_used": image_tool_used,
                    "audio_tool_used": audio_tool_used,
                    "video_tool_used": video_tool_used,
                },
            )
        return (
            "\n---\n".join(trace),
            _best_effort_final_answer(
                ollama_chat_fn=ollama_chat_fn,
                host=host,
                model=model,
                question=question,
                trace=trace,
                timeout=timeout,
            ),
            {
                "steps_used": step_count,
                "max_steps_reached": 1,
                "tool_errors": tool_errors,
                "media_to_text_used": media_to_text_used,
                "read_attachment_used": read_attachment_used,
                "image_tool_used": image_tool_used,
                "audio_tool_used": audio_tool_used,
                "video_tool_used": video_tool_used,
            },
        )

    if kind == "final":
        final_payload = str(payload)
        return (
            "\n---\n".join(trace[:-1]),
            final_payload,
            {
                "steps_used": step_count,
                "max_steps_reached": 1,
                "tool_errors": tool_errors,
                "media_to_text_used": media_to_text_used,
                "read_attachment_used": read_attachment_used,
                "image_tool_used": image_tool_used,
                "audio_tool_used": audio_tool_used,
                "video_tool_used": video_tool_used,
            },
        )
    final_guess = str(last).strip()
    if "TOOL_CALL" in final_guess.upper() or not final_guess:
        final_guess = _best_effort_final_answer(
            ollama_chat_fn=ollama_chat_fn,
            host=host,
            model=model,
            question=question,
            trace=trace,
            timeout=timeout,
        )
    return (
        "\n---\n".join(trace),
        final_guess,
        {
            "steps_used": step_count,
            "max_steps_reached": 1,
            "tool_errors": tool_errors,
            "media_to_text_used": media_to_text_used,
            "read_attachment_used": read_attachment_used,
            "image_tool_used": image_tool_used,
            "audio_tool_used": audio_tool_used,
            "video_tool_used": video_tool_used,
        },
    )
