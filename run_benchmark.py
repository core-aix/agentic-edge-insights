#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import string
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from agent import (
    configure_docling_quiet,
    configure_tool_cache,
    solve_with_tools,
)

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


SYSTEM_PROMPT = (
    "You are a careful agent. Solve the task and return the final answer only. "
    "No explanation, no units unless the answer requires them."
)

VL_STRICT_PROMPT = (
    "Return ONLY the final answer string. "
    "No analysis, no thinking, no preface, no markdown. "
    "If uncertain, still output your single best final answer only."
)


def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("\u2212", "-")
    value = re.sub(r"\*+|`+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def canonical_text(value: str) -> str:
    value = normalize_text(value)
    value = value.replace(",", "")
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_float(value: str):
    cleaned = normalize_text(value).replace(",", "")
    cleaned = cleaned.replace("percent", "").replace("%", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None
    match = NUM_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def strip_answer_prefix(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", value)
    return value.strip()


def prediction_candidates(pred: str) -> list[str]:
    text = str(pred)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = [text]
    if lines:
        candidates.append(lines[0])
        candidates.append(lines[-1])
    if ":" in text:
        candidates.append(text.split(":")[-1])

    out = []
    for c in candidates:
        c = strip_answer_prefix(c)
        if c and c not in out:
            out.append(c)
    return out


def is_correct(pred: str, gold: str) -> bool:
    gold_s = strip_answer_prefix(gold)
    gold_n = canonical_text(gold_s)
    gold_f = parse_float(gold_s)
    for candidate in prediction_candidates(pred):
        if canonical_text(candidate) == gold_n:
            return True
        pred_f = parse_float(candidate)
        if pred_f is not None and gold_f is not None:
            if math.isclose(pred_f, gold_f, rel_tol=1e-6, abs_tol=1e-6):
                return True
    return False


def itbench_root_cause_scorer(pred: str, eval_meta: dict) -> bool:
    expected = [
        str(x).strip()
        for x in (eval_meta.get("expected_names") or [])
        if str(x).strip()
    ]
    if not expected:
        return False
    norm_pred = normalize_text(pred)
    if not norm_pred:
        return False
    domain = str(eval_meta.get("domain", "") or "").strip().lower()

    if domain == "finops":
        parts = [p.strip() for p in re.split(r"[,;\n|]", str(pred or "")) if p.strip()]
        cand = {normalize_text(p) for p in parts if normalize_text(p)}
        exp = {normalize_text(x) for x in expected if normalize_text(x)}
        if exp and cand:
            return exp.issubset(cand) and len(cand) <= max(len(exp) + 3, 5)
        return all(normalize_text(name) in norm_pred for name in expected)

    alt_names = [
        str(x).strip()
        for x in (eval_meta.get("expected_alt_names") or [])
        if str(x).strip()
    ]
    exp_norm = [normalize_text(x) for x in expected]
    alt_norm = [normalize_text(x) for x in alt_names]
    for i, eid in enumerate(exp_norm):
        alt = alt_norm[i] if i < len(alt_norm) else ""
        if eid in norm_pred:
            continue
        if alt and alt in norm_pred:
            continue
        return False
    return True


def score_prediction(
    pred: str, gold: str, scorer: str, eval_meta: dict | None = None
) -> bool:
    meta = eval_meta or {}
    if scorer == "itbench_root_cause":
        return itbench_root_cause_scorer(pred, meta)
    return is_correct(pred, gold)


def maybe_repair_itbench_prediction(
    pred: str,
    scorer: str,
    eval_meta: dict | None,
    draft: str,
) -> str:
    if scorer != "itbench_root_cause":
        return pred
    meta = eval_meta or {}
    if str(meta.get("domain", "") or "").strip().lower() != "sre":
        return pred

    deny = {
        "default",
        "kube-system",
        "ingress-nginx",
        "monitoring",
        "logging",
        "observability",
        "ad",
    }
    deny_norm = {normalize_text(x) for x in deny}
    deny_substrings = (
        "topology-monitor",
        "otel-collector",
        "prometheus-alert",
        "node-exporter",
        "grafana",
        "jaeger",
        "loki",
        "root-cause_entity_group_id",
    )
    uuid_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        flags=re.IGNORECASE,
    )

    def is_noise_entity(v: str) -> bool:
        s = normalize_text(v)
        if not s:
            return True
        if s in deny_norm:
            return True
        if any(tok in s for tok in deny_substrings):
            return True
        if s.startswith("pvc-"):
            return True
        if bool(uuid_re.match(s)):
            return True
        return False

    parts = [p.strip() for p in re.split(r"[,;\n|]", str(pred or "")) if p.strip()]
    cleaned = [p for p in parts if not is_noise_entity(p)]
    if 0 < len(cleaned) <= 3:
        return ", ".join(cleaned)

    m = re.search(
        r"TOOL_RESULT\[rank_sre_root_cause_candidates\]:\n(\{.*?\})",
        str(draft or ""),
        flags=re.S,
    )
    if m:
        try:
            obj = json.loads(m.group(1))
            ranked = [
                str(x).strip()
                for x in (obj.get("ranked_candidates") or [])
                if str(x).strip()
            ]
            ranked = [x for x in ranked if not is_noise_entity(x)]
            if ranked:
                return ", ".join(ranked[:2])
        except Exception:  # noqa: BLE001
            pass

    if cleaned:
        return ", ".join(cleaned[:2])

    return pred


def env_int(name: str, default: int) -> int:
    try:
        raw = str(os.environ.get(name, "")).strip()
        if raw == "":
            return int(default)
        return int(raw)
    except Exception:  # noqa: BLE001
        return int(default)


def _try_install_tesseract_linux() -> tuple[bool, str]:
    installers = []
    use_sudo = (
        shutil.which("sudo") is not None and getattr(os, "geteuid", lambda: 1)() != 0
    )

    def wrap(cmd: list[str]) -> list[str]:
        return (["sudo"] + cmd) if use_sudo else cmd

    if shutil.which("apt-get"):
        installers.append(
            [
                wrap(["apt-get", "update"]),
                wrap(["apt-get", "install", "-y", "tesseract-ocr"]),
            ]
        )
    if shutil.which("dnf"):
        installers.append([[wrap(["dnf", "install", "-y", "tesseract"])][0]])
    if shutil.which("yum"):
        installers.append([[wrap(["yum", "install", "-y", "tesseract"])][0]])
    if shutil.which("pacman"):
        installers.append([[wrap(["pacman", "-Sy", "--noconfirm", "tesseract"])][0]])
    if shutil.which("zypper"):
        installers.append(
            [[wrap(["zypper", "--non-interactive", "install", "tesseract-ocr"])][0]]
        )
    if shutil.which("apk"):
        installers.append([[wrap(["apk", "add", "tesseract-ocr"])][0]])

    if not installers:
        return False, "no supported package manager found"

    for sequence in installers:
        ok = True
        err = ""
        for cmd in sequence:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, check=False
            )
            if proc.returncode != 0:
                ok = False
                err = (proc.stderr or proc.stdout or "").strip()
                break
        if ok and shutil.which("tesseract"):
            return True, "installed"
        if err:
            last_err = err
    return False, locals().get("last_err", "installation command failed")


def ensure_linux_tesseract(mode: str) -> None:
    if sys.platform != "linux":
        return
    if shutil.which("tesseract"):
        return

    msg = (
        "[setup] tesseract not found on Linux. "
        "Docling OCR quality may degrade without it."
    )

    mode = str(mode or "prompt").strip().lower()
    if mode not in {"prompt", "auto", "skip"}:
        mode = "prompt"

    should_install = mode == "auto"
    if mode == "prompt":
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                ans = input(f"{msg} Install now? [Y/n]: ").strip().lower()
            except Exception:  # noqa: BLE001
                ans = ""
            should_install = ans in {"", "y", "yes"}
        else:
            print(msg + " Non-interactive session; skipping install prompt.")
            should_install = False

    if should_install:
        ok, detail = _try_install_tesseract_linux()
        if ok:
            print("[setup] tesseract installed successfully.")
            return
        print(f"[setup] failed to auto-install tesseract: {detail}")

    if not shutil.which("tesseract"):
        print(
            "[setup] continuing without tesseract. "
            "Install manually (e.g., apt-get install tesseract-ocr) for better OCR on Linux."
        )


def ollama_chat(host: str, model: str, messages: list[dict], timeout: int) -> str:
    num_ctx = get_model_max_context(host, model)
    base_num_predict = get_model_max_output_tokens(host, model, num_ctx)
    url = f"{host.rstrip('/')}/api/chat"

    def make_payload() -> dict:
        num_predict = min(int(base_num_predict), int(OLLAMA_MAX_NUM_PREDICT))
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": float(OLLAMA_TEMPERATURE),
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
        if OLLAMA_RANDOM_SEED is not None:
            payload["options"]["seed"] = int(OLLAMA_RANDOM_SEED)
        if str(OLLAMA_KEEP_ALIVE).strip():
            payload["keep_alive"] = str(OLLAMA_KEEP_ALIVE).strip()
        return payload

    payload_messages = messages
    if model_uses_prompt_only_template(host, model):
        merged = []
        images = []
        for m in messages:
            role = str(m.get("role", "user") or "user").upper()
            content = str(m.get("content", "") or "").strip()
            if content:
                merged.append(f"[{role}]\n{content}")
            if role == "USER" and isinstance(m.get("images"), list) and m.get("images"):
                images = list(m.get("images") or [])
        merged_prompt = "\n\n".join(merged).strip()
        if merged_prompt:
            merged_prompt += "\n\n[ASSISTANT]\n"
        if images:
            payload_messages = [
                {"role": "user", "content": merged_prompt, "images": images}
            ]
        else:
            payload_messages = [{"role": "user", "content": merged_prompt}]

    payload = make_payload()
    payload["messages"] = payload_messages
    retries = max(0, int(OLLAMA_CHAT_RETRIES))
    max_req_timeout = int(OLLAMA_MAX_REQUEST_TIMEOUT)
    req_timeout = min(int(timeout), max_req_timeout) if timeout > 0 else max_req_timeout
    has_images = any(
        isinstance(m.get("images"), list) and m.get("images") for m in messages
    )
    is_vl_model = "vl" in str(model or "").lower()
    step_timeout = int(
        OLLAMA_IMAGE_STEP_TIMEOUT
        if (has_images or is_vl_model)
        else OLLAMA_STEP_TIMEOUT
    )
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        start = time.monotonic()
        try:
            read_timeout = (
                min(req_timeout, max(5, step_timeout))
                if step_timeout > 0
                else min(req_timeout, 30)
            )
            resp = requests.post(
                url,
                json=payload,
                timeout=(20, read_timeout),
                stream=True,
            )
            if resp.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                if attempt < retries:
                    time.sleep(min(10, 2 * (attempt + 1)))
                    continue
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if step_timeout > 0 and (time.monotonic() - start) >= step_timeout:
                    content = "".join(content_parts).strip()
                    if content:
                        return content
                    thinking = "".join(thinking_parts).strip()
                    if thinking:
                        return thinking
                    break
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = evt.get("message", {}) or {}
                c = str(msg.get("content", "") or "")
                t = str(msg.get("thinking", "") or "")
                if c:
                    content_parts.append(c)
                if t:
                    thinking_parts.append(t)
                if evt.get("done"):
                    break

            content = "".join(content_parts).strip()
            if content:
                return content
            thinking = "".join(thinking_parts).strip()
            return thinking
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            content = "".join(content_parts).strip()
            if content:
                return content
            thinking = "".join(thinking_parts).strip()
            if thinking:
                return thinking
            last_exc = exc
            if attempt < retries:
                time.sleep(min(10, 2 * (attempt + 1)))
                continue
            raise
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            code = getattr(exc.response, "status_code", 0)
            if code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(min(10, 2 * (attempt + 1)))
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("ollama_chat failed without response")


CTX_CACHE: dict[str, int] = {}
OUT_CACHE: dict[str, int] = {}
TOOL_SUPPORT_CACHE: dict[str, bool] = {}
TEMPLATE_CACHE: dict[str, str] = {}

OLLAMA_CHAT_RETRIES = 2
OLLAMA_MAX_REQUEST_TIMEOUT = 240
OLLAMA_STEP_TIMEOUT = 120
OLLAMA_IMAGE_STEP_TIMEOUT = 420
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_RANDOM_SEED: int | None = None
OLLAMA_TEMPERATURE = 0.2
OLLAMA_MAX_NUM_CTX = 262144
OLLAMA_MAX_NUM_PREDICT = 32768


def get_model_max_context(host: str, model: str) -> int:
    key = f"{host}|{model}"
    if key in CTX_CACHE:
        return CTX_CACHE[key]

    fallback = min(32768, int(OLLAMA_MAX_NUM_CTX))
    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/show",
            json={"model": model},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            candidates = []

            details = data.get("details", {}) or {}
            if isinstance(details, dict):
                for k in ["context_length", "num_ctx", "n_ctx"]:
                    v = details.get(k)
                    if isinstance(v, int) and v > 0:
                        candidates.append(v)

            model_info = data.get("model_info", {}) or {}
            if isinstance(model_info, dict):
                for k, v in model_info.items():
                    lk = str(k).lower()
                    if (
                        "context_length" in lk or lk.endswith(".n_ctx_train")
                    ) and isinstance(v, int):
                        if v > 0:
                            candidates.append(v)

            if candidates:
                CTX_CACHE[key] = int(min(max(candidates), int(OLLAMA_MAX_NUM_CTX)))
                return CTX_CACHE[key]
    except Exception:  # noqa: BLE001
        pass

    CTX_CACHE[key] = fallback
    return fallback


def get_model_template(host: str, model: str) -> str:
    key = f"{host}|{model}"
    if key in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[key]
    tmpl = ""
    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/show",
            json={"model": model},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            tmpl = str(data.get("template", "") or "")
    except Exception:  # noqa: BLE001
        tmpl = ""
    TEMPLATE_CACHE[key] = tmpl
    return tmpl


def model_uses_prompt_only_template(host: str, model: str) -> bool:
    t = get_model_template(host, model).strip()
    return t in {"{{ .Prompt }}", "{{.Prompt}}"}


def get_model_max_output_tokens(host: str, model: str, num_ctx: int) -> int:
    key = f"{host}|{model}"
    if key in OUT_CACHE:
        return OUT_CACHE[key]

    # Prefer a large but bounded output budget.
    fallback = int(min(num_ctx, int(OLLAMA_MAX_NUM_PREDICT)))
    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/show",
            json={"model": model},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            candidates = []

            details = data.get("details", {}) or {}
            if isinstance(details, dict):
                for k in ["max_output_tokens", "num_predict", "n_predict"]:
                    v = details.get(k)
                    if isinstance(v, int) and v > 0:
                        candidates.append(v)

            model_info = data.get("model_info", {}) or {}
            if isinstance(model_info, dict):
                for k, v in model_info.items():
                    lk = str(k).lower()
                    if any(
                        tag in lk for tag in ["max_output", "n_predict", "num_predict"]
                    ):
                        if isinstance(v, int) and v > 0:
                            candidates.append(v)

            if candidates:
                OUT_CACHE[key] = int(
                    min(max(candidates), num_ctx, int(OLLAMA_MAX_NUM_PREDICT))
                )
                return OUT_CACHE[key]
    except Exception:  # noqa: BLE001
        pass

    OUT_CACHE[key] = fallback
    return fallback


def model_has_native_tool_support(host: str, model: str) -> bool:
    key = f"{host}|{model}"
    if key in TOOL_SUPPORT_CACHE:
        return TOOL_SUPPORT_CACHE[key]

    # Conservative default: assume native tool-call support unless metadata says otherwise.
    supported = True
    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/show",
            json={"model": model},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            caps = [str(x).lower() for x in (data.get("capabilities") or [])]
            if caps:
                tool_markers = {
                    "tools",
                    "tool",
                    "function_calling",
                    "function-calling",
                }
                supported = any(m in caps for m in tool_markers)
    except Exception:  # noqa: BLE001
        supported = True

    TOOL_SUPPORT_CACHE[key] = bool(supported)
    return TOOL_SUPPORT_CACHE[key]


def autotune_parallelism(
    host: str,
    model: str,
    timeout: int,
    max_parallelism: int,
) -> int:
    candidates = [1, 2, 4, 6, 8, 12, 16]
    candidates = [c for c in candidates if c <= max_parallelism]
    best = 1
    best_tput = 0.0

    def probe_once() -> bool:
        try:
            out = ollama_chat(
                host,
                model,
                [
                    {"role": "system", "content": "Reply with 1."},
                    {"role": "user", "content": "Reply with 1."},
                ],
                timeout,
            )
            return bool(out)
        except Exception:  # noqa: BLE001
            return False

    for c in candidates:
        n = max(4, c * 2)
        start = time.time()
        ok = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            futures = [ex.submit(probe_once) for _ in range(n)]
            for f in concurrent.futures.as_completed(futures):
                ok += int(f.result())
        elapsed = max(1e-6, time.time() - start)
        tput = ok / elapsed
        success = ok / n
        print(
            f"[autotune] workers={c} success={success:.2f} throughput={tput:.2f} req/s"
        )
        if success >= 0.9 and tput > best_tput:
            best_tput = tput
            best = c
    print(f"[autotune] selected parallelism={best}")
    return best


def solve_question_simple(
    host: str, model: str, question: str, timeout: int, attachment_path: str
) -> tuple[str, str]:
    def looks_meta(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return True
        markers = [
            "we are given",
            "candidate answer",
            "instructions",
            "i will",
            "let's",
            "as an ai",
            "based on the prompt",
            "cannot provide",
        ]
        if any(m in s for m in markers):
            return True
        # Long narrative outputs are usually low-confidence for exact matching.
        return len(s.split()) > 20

    def is_concise_answer(text: str) -> bool:
        s = str(text or "").strip()
        return bool(s) and len(s.split()) <= 12 and len(s) <= 80

    image_b64 = maybe_attachment_image_b64(host, attachment_path, model)
    # Two-pass self-refine loop to keep the protocol agentic.
    draft = ollama_chat(
        host,
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{question}\n\n"
                    "Think carefully. Return only your best final answer."
                ),
                **({"images": [image_b64]} if image_b64 else {}),
            },
        ],
        timeout,
    )

    final = ollama_chat(
        host,
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{question}\n\n"
                    f"Candidate answer: {draft}\n"
                    "Verify and correct if needed. Return only the final answer."
                ),
                **({"images": [image_b64]} if image_b64 else {}),
            },
        ],
        timeout,
    )
    # Guard against refinement-step degeneration where draft is good but
    # final answer becomes meta/instructional text.
    if is_concise_answer(draft) and looks_meta(final):
        final = draft
    return draft, final


def solve_question_tool(
    host: str,
    model: str,
    question: str,
    attachment_path: str,
    timeout: int,
    max_steps: int,
    tool_prompt_profile: str,
) -> tuple[str, str, dict]:
    image_b64 = maybe_attachment_image_b64(host, attachment_path, model)
    return solve_with_tools(
        ollama_chat_fn=ollama_chat,
        host=host,
        model=model,
        question=question,
        attachment_path=attachment_path,
        attachment_image_b64=image_b64,
        timeout=timeout,
        max_steps=max_steps,
        tool_prompt_profile=tool_prompt_profile,
    )


def execute_job(job: dict) -> dict:
    start = time.time()
    error = ""
    draft, pred = "", ""
    steps_used = 0
    max_steps_reached = 0
    tool_errors = 0
    image_sent = 0
    media_to_text_used = 0
    read_attachment_used = 0
    image_tool_used = 0
    audio_tool_used = 0
    video_tool_used = 0
    attachment_path = str(job.get("file_path", "") or "")
    ap_lower = attachment_path.lower()
    attachment_is_image = int(
        ap_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))
    )
    attachment_is_audio = int(
        ap_lower.endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"))
    )
    attachment_is_video = int(
        ap_lower.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"))
    )
    try:
        prompt = job.get("augmented_question") or job["question"]
        max_steps = job["max_steps"]
        level_steps = job.get("level_steps", {})
        if isinstance(level_steps, dict):
            max_steps = int(level_steps.get(str(job["level"]), max_steps))

        use_tool_mode = job["agent_mode"] == "tool"
        native_tools = model_has_native_tool_support(job["host"], job["model"])
        if use_tool_mode and ("vl" in str(job["model"]).lower()):
            # VL models that are explicitly tagged as VL frequently perform better on
            # direct multimodal answering than strict tool protocol.
            use_tool_mode = False
        if use_tool_mode and attachment_is_image and image_sent == 0:
            image_sent = int(
                bool(
                    maybe_attachment_image_b64(
                        job["host"], attachment_path, job["model"]
                    )
                )
            )
        if (
            use_tool_mode
            and attachment_is_image
            and image_sent == 1
            and not native_tools
        ):
            # Vision-only models without native tool support should solve image tasks
            # from image bytes directly, not via tool protocol.
            use_tool_mode = False

        if use_tool_mode:
            base_profile = "native" if native_tools else "simple"
            benchmark_name = str(job.get("benchmark", "itbench_lite") or "itbench_lite")
            task_type = str(job.get("task_type", "") or "")
            if benchmark_name == "itbench_lite" and "finops" in task_type:
                benchmark_name = "itbench_lite_finops"
            elif benchmark_name == "itbench_lite" and "sre" in task_type:
                benchmark_name = "itbench_lite_sre"
            tool_prompt_profile = f"{base_profile}:{benchmark_name}"
            draft, pred, meta = solve_question_tool(
                job["host"],
                job["model"],
                prompt,
                attachment_path,
                job["timeout"],
                max_steps,
                tool_prompt_profile,
            )
            image_sent = int(
                bool(
                    maybe_attachment_image_b64(
                        job["host"], attachment_path, job["model"]
                    )
                )
            )
            steps_used = int(meta.get("steps_used", 0))
            max_steps_reached = int(meta.get("max_steps_reached", 0))
            tool_errors = int(meta.get("tool_errors", 0))
            media_to_text_used = int(meta.get("media_to_text_used", 0))
            read_attachment_used = int(meta.get("read_attachment_used", 0))
            image_tool_used = int(meta.get("image_tool_used", 0))
            audio_tool_used = int(meta.get("audio_tool_used", 0))
            video_tool_used = int(meta.get("video_tool_used", 0))
        else:
            image_sent = int(
                bool(
                    maybe_attachment_image_b64(
                        job["host"], attachment_path, job["model"]
                    )
                )
            )
            draft, pred = solve_question_simple(
                job["host"],
                job["model"],
                prompt,
                job["timeout"],
                attachment_path,
            )
            steps_used = 2
        pred = canonicalize_final_answer(pred)
        scorer = job.get("scorer", "enhanced")
        pred = maybe_repair_itbench_prediction(
            pred,
            scorer,
            job.get("eval_meta", {}),
            draft,
        )
        correct = score_prediction(
            pred,
            job["gold"],
            scorer,
            job.get("eval_meta", {}),
        )
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        correct = False

    elapsed = time.time() - start
    failure_reason = ""
    if not correct:
        reasons = []
        if error:
            reasons.append("runtime_error")
        if tool_errors > 0:
            reasons.append("tool_error")
        if max_steps_reached:
            reasons.append("max_steps_reached")
        if not reasons:
            reasons.append("wrong_answer")
        failure_reason = "+".join(reasons)

    return {
        "run": job["run"],
        "repeat_idx": int(job.get("repeat_idx", 0)),
        "repeat_seed": int(job.get("repeat_seed", 0)),
        "model": job["model"],
        "task_id": job["task_id"],
        "level": job["level"],
        "gold": job["gold"],
        "pred": pred,
        "correct": int(correct),
        "task_type": job["task_type"],
        "file_name": job.get("file_name", ""),
        "file_path": job.get("file_path", ""),
        "draft": draft,
        "latency_sec": elapsed,
        "steps_used": steps_used,
        "max_steps_reached": max_steps_reached,
        "tool_errors": tool_errors,
        "image_sent": image_sent,
        "attachment_is_image": attachment_is_image,
        "attachment_is_audio": attachment_is_audio,
        "attachment_is_video": attachment_is_video,
        "media_to_text_used": media_to_text_used,
        "read_attachment_used": read_attachment_used,
        "image_tool_used": image_tool_used,
        "audio_tool_used": audio_tool_used,
        "video_tool_used": video_tool_used,
        "failure_reason": failure_reason,
        "error": error,
        "eval_meta": json.dumps(job.get("eval_meta", {}), ensure_ascii=False),
        "family": infer_family(job["model"]),
        "is_coder": int("coder" in str(job["model"]).lower()),
        "is_vision": int(is_vision_model(job["host"], job["model"])),
    }


def canonicalize_final_answer(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s

    # Common wrappers
    s = re.sub(r"^```[a-zA-Z0-9_]*\n", "", s)
    s = s.replace("```", "").strip()
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"\*(.*?)\*", r"\1", s)

    patterns = [
        r"(?is)^\s*final\s+answer\s*[:\-]\s*(.+)$",
        r"(?is)^\s*answer\s*[:\-]\s*(.+)$",
        r"(?is)^\s*the\s+answer\s+is\s+(.+)$",
        r"(?is)^\s*based\s+on\s+.*?,\s*(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, s)
        if m:
            s = m.group(1).strip()
            break

    # If first line is concise, prefer it.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) > 1 and len(lines[0]) <= 120:
        s = lines[0]

    # Remove leading narrative phrases that hurt exact-match evaluation.
    s = re.sub(
        r"(?is)^\s*(based on|from the evidence|according to|i found that|it appears that)\b[^:]{0,120}:\s*",
        "",
        s,
    ).strip()

    # Yes/No style answers should be canonicalized.
    low = s.lower()
    if low.startswith("yes"):
        return "Yes"
    if low.startswith("no"):
        return "No"

    # Fraction-list style answers should be normalized to clean comma-separated tokens.
    if "/" in s and "," in s:
        fracs = re.findall(r"\b\d+\s*/\s*\d+\b", s)
        if len(fracs) >= 2:
            seen = set()
            uniq = []
            for f in fracs:
                token = f.replace(" ", "")
                if token in seen:
                    continue
                seen.add(token)
                uniq.append(token)
            if uniq:
                return ",".join(uniq)

    # Trim trailing explanatory sentence after first period for short answers.
    if "." in s and len(s) > 40 and "," not in s and ";" not in s:
        first = s.split(".", 1)[0].strip()
        if 1 <= len(first.split()) <= 8:
            s = first

    # Benchmark answers are typically concise; compress long narrative outputs.
    words = s.split()
    if len(words) > 24 or len(s) > 220:
        if "." in s:
            s = s.split(".", 1)[0].strip()
        elif ";" in s:
            s = s.split(";", 1)[0].strip()
        elif "," in s and len(words) > 32:
            s = s.split(",", 1)[0].strip()
        words = s.split()
        if len(words) > 24:
            s = " ".join(words[:24]).strip()

    return s.strip('"').strip()


@dataclass
class RunSpec:
    name: str
    models: list[str]


MODEL_SIZE_B = {
    "gemma3:1b": 1.0,
    "mistral:7b": 7,
    "ministral-3:3b": 3.8,
    "ministral-3:8b": 8.9,
    "ministral-3:14b": 13.9,
    "mistral-small3.2:24b": 24.0,
    "gemma3:4b": 4.3,
    "gemma3:12b": 12.2,
    "gemma3:27b": 27.4,
    "qwen2.5-coder:7b": 7.6,
    "qwen2.5-coder:14b": 14.8,
    "qwen3-coder:30b": 30.5,
    "qwen2.5-coder:32b": 32.8,
    "qwen3:30b": 30.5,
    "qwen3-vl:2b": 2.0,
    "qwen3-vl:4b": 4.0,
    "qwen3-vl:8b": 8.0,
    "qwen3-vl:30b": 30.0,
    "qwen3-vl:32b": 32.0,
    "qwen2.5vl:32b": 32.0,
    "qwen3-vl:32b": 32.0,
    "qwen3-coder-next:latest": 79.7,
    "llama3.3:70b": 70.6,
    "deepseek-coder:33b": 33.0,
    "qwen2.5-coder:3b": 3.1,
    "gpt-oss:120b": 116.8,
}


def parse_models_arg(models_arg: str) -> list[str]:
    return [m.strip() for m in (models_arg or "").split(",") if m.strip()]


def order_models_capability_first(models: list[str]) -> list[str]:
    # Deterministic high-capability-first order for partial-run usefulness.
    preferred = [
        "qwen3-coder-next:latest",
        "qwen2.5-coder:14b",
        "qwen2.5-coder:7b",
    ]
    pref_rank = {m: i for i, m in enumerate(preferred)}

    def key_fn(m: str):
        if m in pref_rank:
            return (0, pref_rank[m], m)
        size = MODEL_SIZE_B.get(m, 0)
        return (1, -float(size), m)

    return sorted(list(dict.fromkeys(models)), key=key_fn)


VISION_CACHE: dict[str, bool] = {}


def is_vision_model(host: str, model: str) -> bool:
    key = f"{host}|{model}"
    if key in VISION_CACHE:
        return VISION_CACHE[key]

    m = model.lower()
    hinted = ("vl" in m) or ("vision" in m)

    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/show",
            json={"model": model},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            caps = [str(x).lower() for x in (data.get("capabilities") or [])]
            if caps:
                VISION_CACHE[key] = "vision" in caps
                return VISION_CACHE[key]
            text = json.dumps(data).lower()
            meta_vision = ("mmproj" in text) or ("projector" in text)
            VISION_CACHE[key] = bool(hinted or meta_vision)
            return VISION_CACHE[key]
    except Exception:  # noqa: BLE001
        pass

    VISION_CACHE[key] = bool(hinted)
    return VISION_CACHE[key]


def maybe_attachment_image_b64(host: str, path: str, model: str) -> str:
    if not path or not is_vision_model(host, model):
        return ""
    p = Path(path)
    if (not p.exists()) or p.is_dir():
        return ""
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
        return ""
    try:
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:  # noqa: BLE001
        return ""


def infer_family(model: str) -> str:
    m = model.lower()
    if m.startswith("qwen"):
        return "qwen"
    if m.startswith("gemma3"):
        return "gemma3"
    if m.startswith("ministral"):
        return "ministral"
    return "other"


def _collect_itbench_lite_scenarios(domains: set[str]) -> dict[str, list[str]]:
    repo = "ibm-research/ITBench-Lite"
    files = list_repo_files(repo_id=repo, repo_type="dataset")
    gt_files = [f for f in files if f.endswith("/ground_truth.yaml")]
    grouped: dict[str, list[str]] = {"finops": [], "sre": []}
    for path in gt_files:
        domain = ""
        if "/finops/" in path:
            domain = "finops"
        elif "/sre/" in path:
            domain = "sre"
        if domain and domain in domains:
            grouped[domain].append(path)
    return grouped


def _download_itbench_file(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)


def _load_yaml_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data
    return {}


def _itbench_gold_from_ground_truth(ground_truth: dict, domain: str) -> list[str]:
    if domain == "finops":
        out = []
        for item in ground_truth.get("resource") or []:
            r = item.get("resource", {}) if isinstance(item, dict) else {}
            if bool(r.get("root_cause", False)):
                name = str(r.get("name", "") or "").strip()
                if name:
                    out.append(name)
        return out

    if domain == "sre":
        out = []
        for g in ground_truth.get("groups") or []:
            if isinstance(g, dict) and bool(g.get("root_cause", False)):
                gid = str(g.get("id", "") or "").strip()
                if gid:
                    out.append(gid)
        return out

    return []


def load_itbench_lite_subset(
    domains: set[str],
    per_level: int,
    seed: int,
    all_tasks: bool,
) -> list[dict]:
    repo = "ibm-research/ITBench-Lite"
    grouped = _collect_itbench_lite_scenarios(domains)
    rng = random.Random(seed)
    selected_paths = []
    for domain in sorted(grouped.keys()):
        paths = list(grouped[domain])
        rng.shuffle(paths)
        if all_tasks:
            take = len(paths)
        else:
            take = min(per_level, len(paths))
        print(f"Selecting {take} ITBench-Lite {domain} scenarios (pool={len(paths)})")
        selected_paths.extend(paths[:take])

    rows = []
    for gt_rel in selected_paths:
        gt_path = _download_itbench_file(repo, gt_rel)
        gt = _load_yaml_file(gt_path)
        domain = "finops" if "/finops/" in gt_rel else "sre"
        scenario_root = gt_rel.rsplit("/ground_truth.yaml", 1)[0]
        scenario_name = scenario_root.split("/")[-1]
        expected_names = _itbench_gold_from_ground_truth(gt, domain)
        expected_alt_names: list[str] = []
        if domain == "sre":
            for g in gt.get("groups") or []:
                if isinstance(g, dict) and bool(g.get("root_cause", False)):
                    nm = str(g.get("name", "") or "").strip()
                    if nm:
                        expected_alt_names.append(nm)
        if not expected_names:
            continue

        if domain == "finops":
            anomaly_rel = f"{scenario_root}/anomaly.json"
            data_rel = f"{scenario_root}/data.csv"
            anomaly_path = _download_itbench_file(repo, anomaly_rel)
            data_path = _download_itbench_file(repo, data_rel)
            try:
                with open(anomaly_path, encoding="utf-8") as f:
                    anomaly = json.load(f)
            except Exception:  # noqa: BLE001
                anomaly = {}
            question = (
                "ITBench-Lite FinOps root-cause analysis. "
                f"Given anomaly details {json.dumps(anomaly, ensure_ascii=False)} and cost data file "
                f"at path {data_path}, identify the anomalous resource name(s) causing the cost spike. "
                "Return only the resource name(s), comma-separated if multiple."
            )
            file_path = data_path
            file_name = Path(data_path).name
            task_type = "itbench_finops_root_cause"
        else:
            k8s_events_rel = f"{scenario_root}/k8s_events_raw.tsv"
            k8s_objects_rel = f"{scenario_root}/k8s_objects_raw.tsv"
            k8s_events_path = _download_itbench_file(repo, k8s_events_rel)
            k8s_objects_path = _download_itbench_file(repo, k8s_objects_rel)
            scenario_dir = str(Path(k8s_events_path).parent)
            question = (
                "ITBench-Lite SRE fault-localization from static snapshot files. "
                f"Analyze Kubernetes events at {k8s_events_path}, Kubernetes objects at {k8s_objects_path}, "
                f"and any additional files under scenario directory {scenario_dir}. "
                "Use local files only; do not use web tools. "
                "Start with summarize_sre_candidates and rank_sre_root_cause_candidates. "
                "Identify the root-cause entity group ID(s), not namespaces. "
                "IDs are entity identifiers (for example object-name-like IDs or UUID-like IDs). "
                "Return only the ID value(s), comma-separated if multiple, with no explanation."
            )
            file_path = k8s_events_path
            file_name = Path(k8s_events_path).name
            task_type = "itbench_sre_root_cause"

        rows.append(
            {
                "task_id": f"itbench-lite:{domain}:{scenario_name}",
                "question": question,
                "level": 1,
                "gold": ", ".join(expected_names),
                "task_type": task_type,
                "file_name": file_name,
                "file_path": file_path,
                "augmented_question": question,
                "eval_meta": {
                    "expected_names": expected_names,
                    "expected_alt_names": expected_alt_names,
                    "domain": domain,
                },
            }
        )
    return rows


def load_selected_questions(args) -> list[dict]:
    domains = {
        d.strip().lower()
        for d in str(args.itbench_domains).split(",")
        if d.strip().lower() in {"finops", "sre"}
    }
    if not domains:
        domains = {"finops"}
    return load_itbench_lite_subset(
        domains=domains,
        per_level=args.per_level,
        seed=args.seed,
        all_tasks=args.all_tasks,
    )


def resolve_scorer(scorer_arg: str) -> str:
    s = str(scorer_arg or "auto").strip().lower()
    if s != "auto":
        return s
    return "itbench_root_cause"


def run_benchmark(
    host: str,
    spec: RunSpec,
    questions: list[dict],
    timeout: int,
    out_dir: Path,
    agent_mode: str,
    max_steps: int,
    parallelism: int,
    model_parallelism: int,
    task_isolation: str,
    scorer: str,
    level_steps: dict,
    benchmark: str = "itbench_lite",
    repeat_idx: int = 0,
    repeat_seed: int = 42,
):
    def shutdown_executor_now(executor: concurrent.futures.Executor) -> None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        processes = getattr(executor, "_processes", None)
        if isinstance(processes, dict):
            for proc in list(processes.values()):
                try:
                    if proc is not None and proc.is_alive():
                        proc.terminate()
                except Exception:  # noqa: BLE001
                    pass

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"results_{spec.name}.csv"
    completed = set()
    if result_path.exists():
        try:
            old = pd.read_csv(result_path)
            if "repeat_idx" in old.columns:
                completed = set(
                    zip(
                        old["model"].astype(str),
                        old["task_id"].astype(str),
                        old["repeat_idx"].astype(int),
                        strict=False,
                    )
                )
            else:
                completed = set(
                    (m, t, 0)
                    for m, t in zip(
                        old["model"].astype(str),
                        old["task_id"].astype(str),
                        strict=False,
                    )
                )
            print(f"[{spec.name}] resume mode: {len(completed)} completed rows found")
        except Exception:  # noqa: BLE001
            completed = set()

    records = []

    # We parallelize across questions per model, and can also run multiple models
    # concurrently (useful when server has multiple GPUs and model placement is split).
    def run_one_model(model: str):
        model_records = []
        jobs = []
        for q in questions:
            if (str(model), str(q["task_id"]), int(repeat_idx)) in completed:
                continue
            jobs.append(
                {
                    "run": spec.name,
                    "host": host,
                    "model": model,
                    "question": q["question"],
                    "augmented_question": q.get("augmented_question", ""),
                    "file_path": q.get("file_path", ""),
                    "task_id": q["task_id"],
                    "level": q["level"],
                    "gold": q["gold"],
                    "task_type": q["task_type"],
                    "file_name": q.get("file_name", ""),
                    "eval_meta": q.get("eval_meta", {}),
                    "timeout": timeout,
                    "agent_mode": agent_mode,
                    "max_steps": max_steps,
                    "scorer": scorer,
                    "benchmark": benchmark,
                    "level_steps": level_steps,
                    "repeat_idx": int(repeat_idx),
                    "repeat_seed": int(repeat_seed),
                }
            )
        print(
            f"[{spec.name}] starting model {model} with "
            f"{len(jobs)} tasks @ parallelism={parallelism}"
        )
        if parallelism <= 1:
            for job in jobs:
                rec = execute_job(job)
                model_records.append(rec)
                pd.DataFrame([rec]).to_csv(
                    result_path,
                    mode="a",
                    index=False,
                    header=not result_path.exists(),
                )
                print(
                    f"[{spec.name}] {rec['model']} level={rec['level']} "
                    f"correct={rec['correct']} latency={rec['latency_sec']:.2f}s"
                )
        else:
            executor_cls = (
                concurrent.futures.ProcessPoolExecutor
                if task_isolation == "process"
                else concurrent.futures.ThreadPoolExecutor
            )
            ex = executor_cls(max_workers=parallelism)
            try:
                futures = [ex.submit(execute_job, job) for job in jobs]
                for fut in concurrent.futures.as_completed(futures):
                    rec = fut.result()
                    model_records.append(rec)
                    pd.DataFrame([rec]).to_csv(
                        result_path,
                        mode="a",
                        index=False,
                        header=not result_path.exists(),
                    )
                    print(
                        f"[{spec.name}] {rec['model']} level={rec['level']} "
                        f"correct={rec['correct']} latency={rec['latency_sec']:.2f}s"
                    )
            except KeyboardInterrupt:
                shutdown_executor_now(ex)
                raise
            finally:
                shutdown_executor_now(ex)
        return model_records

    if model_parallelism <= 1:
        for model in spec.models:
            records.extend(run_one_model(model))
    else:
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=model_parallelism)
        try:
            futures = [ex.submit(run_one_model, model) for model in spec.models]
            for fut in concurrent.futures.as_completed(futures):
                records.extend(fut.result())
        except KeyboardInterrupt:
            shutdown_executor_now(ex)
            raise
        finally:
            shutdown_executor_now(ex)
    if result_path.exists():
        df = pd.read_csv(result_path)
    else:
        df = pd.DataFrame(records)
        df.to_csv(result_path, index=False)
    return df


def summarize(df: pd.DataFrame, out_dir: Path, tag: str, benchmark: str = "ITBench-Lite"):
    if "repeat_idx" in df.columns:
        rep_col = "repeat_idx"
    else:
        rep_col = "__repeat_idx"
        df = df.copy()
        df[rep_col] = 0

    per_repeat_model = df.groupby(["model", rep_col], as_index=False).agg(
        accuracy=("correct", "mean"), latency_sec=("latency_sec", "mean")
    )
    model_summary = (
        per_repeat_model.groupby("model", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            latency_sec=("latency_sec", "mean"),
            latency_std=("latency_sec", "std"),
            n_repeats=("accuracy", "count"),
        )
        .sort_values("accuracy", ascending=False)
    )
    model_summary["accuracy_std"] = model_summary["accuracy_std"].fillna(0.0)
    model_summary["latency_std"] = model_summary["latency_std"].fillna(0.0)

    per_repeat_level = df.groupby(["model", "level", rep_col], as_index=False).agg(
        accuracy=("correct", "mean"), latency_sec=("latency_sec", "mean")
    )
    level_summary = (
        per_repeat_level.groupby(["model", "level"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            latency_sec=("latency_sec", "mean"),
            latency_std=("latency_sec", "std"),
            n_repeats=("accuracy", "count"),
        )
        .sort_values(["level", "accuracy"], ascending=[True, False])
    )
    level_summary["accuracy_std"] = level_summary["accuracy_std"].fillna(0.0)
    level_summary["latency_std"] = level_summary["latency_std"].fillna(0.0)

    per_repeat_type = df.groupby(["model", "task_type", rep_col], as_index=False).agg(
        accuracy=("correct", "mean"),
        latency_sec=("latency_sec", "mean"),
        n=("correct", "count"),
    )
    type_summary = (
        per_repeat_type.groupby(["model", "task_type"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            latency_sec=("latency_sec", "mean"),
            latency_std=("latency_sec", "std"),
            n=("n", "mean"),
            n_repeats=("accuracy", "count"),
        )
        .sort_values(["task_type", "accuracy"], ascending=[True, False])
    )
    type_summary["accuracy_std"] = type_summary["accuracy_std"].fillna(0.0)
    type_summary["latency_std"] = type_summary["latency_std"].fillna(0.0)
    type_overall = (
        df.groupby("task_type", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            latency_sec=("latency_sec", "mean"),
            n=("correct", "count"),
        )
        .sort_values("task_type")
    )
    family_summary = (
        df.groupby("family", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            latency_sec=("latency_sec", "mean"),
            n=("correct", "count"),
        )
        .sort_values("accuracy", ascending=False)
    )
    coder_summary = (
        df.groupby("is_coder", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            latency_sec=("latency_sec", "mean"),
            n=("correct", "count"),
        )
        .sort_values("is_coder")
    )
    vision_summary = (
        df.groupby("is_vision", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            latency_sec=("latency_sec", "mean"),
            n=("correct", "count"),
        )
        .sort_values("is_vision")
    )
    failures = df[df["correct"] == 0].copy()
    if len(failures) > 0:
        failure_summary = (
            failures.groupby("failure_reason", as_index=False)
            .agg(count=("failure_reason", "count"))
            .sort_values("count", ascending=False)
        )
        failure_summary["pct_of_failures"] = failure_summary["count"] / len(failures)

        failure_by_model = (
            failures.groupby(["model", "failure_reason"], as_index=False)
            .agg(count=("failure_reason", "count"))
            .sort_values(["model", "count"], ascending=[True, False])
        )
        totals = failure_by_model.groupby("model", as_index=False).agg(
            total=("count", "sum")
        )
        failure_by_model = failure_by_model.merge(totals, on="model", how="left")
        failure_by_model["pct_within_model_failures"] = (
            failure_by_model["count"] / failure_by_model["total"]
        )
    else:
        failure_summary = pd.DataFrame(
            [{"failure_reason": "none", "count": 0, "pct_of_failures": 0.0}]
        )
        failure_by_model = pd.DataFrame(
            columns=[
                "model",
                "failure_reason",
                "count",
                "total",
                "pct_within_model_failures",
            ]
        )

    model_summary.to_csv(out_dir / f"summary_model_{tag}.csv", index=False)
    level_summary.to_csv(out_dir / f"summary_level_{tag}.csv", index=False)
    type_summary.to_csv(out_dir / f"summary_task_type_{tag}.csv", index=False)
    type_overall.to_csv(out_dir / f"summary_task_type_overall_{tag}.csv", index=False)
    family_summary.to_csv(out_dir / f"summary_family_{tag}.csv", index=False)
    coder_summary.to_csv(out_dir / f"summary_coder_{tag}.csv", index=False)
    vision_summary.to_csv(out_dir / f"summary_vision_{tag}.csv", index=False)
    failure_summary.to_csv(out_dir / f"summary_failure_types_{tag}.csv", index=False)
    failure_by_model.to_csv(
        out_dir / f"summary_failure_types_by_model_{tag}.csv", index=False
    )

    plt.figure(figsize=(10, 4))
    plt.bar(
        model_summary["model"],
        model_summary["accuracy"],
        yerr=model_summary["accuracy_std"],
        capsize=4,
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"{benchmark} accuracy by model ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_{tag}.png", dpi=200)
    plt.close()

    level_with_size = level_summary.copy()
    level_with_size["size_b"] = level_with_size["model"].map(MODEL_SIZE_B)
    level_with_size = level_with_size.dropna(subset=["size_b"])
    plt.figure(figsize=(7, 5))
    for level in sorted(level_with_size["level"].unique()):
        part = level_with_size[level_with_size["level"] == level].sort_values("size_b")
        plt.errorbar(
            part["size_b"],
            part["accuracy"],
            yerr=part["accuracy_std"],
            marker="o",
            label=f"Level {level}",
            capsize=3,
        )
    plt.xlabel("Model size (B params)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs size by difficulty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"size_vs_difficulty_{tag}.png", dpi=200)
    plt.close()

    pivot = level_summary.pivot(
        index="model", columns="level", values="accuracy"
    ).fillna(0)
    plt.figure(figsize=(10, 5))
    for level in sorted(pivot.columns):
        std_part = (
            level_summary[level_summary["level"] == level]
            .set_index("model")
            .reindex(pivot.index)["accuracy_std"]
            .fillna(0.0)
        )
        plt.errorbar(
            pivot.index,
            pivot[level],
            yerr=std_part,
            marker="o",
            label=f"Level {level}",
            capsize=3,
        )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by difficulty level ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_by_level_{tag}.png", dpi=200)
    plt.close()

    type_pivot = type_summary.pivot(
        index="model", columns="task_type", values="accuracy"
    ).fillna(0)
    plt.figure(figsize=(11, 5))
    for task_type in sorted(type_pivot.columns):
        std_part = (
            type_summary[type_summary["task_type"] == task_type]
            .set_index("model")
            .reindex(type_pivot.index)["accuracy_std"]
            .fillna(0.0)
        )
        plt.errorbar(
            type_pivot.index,
            type_pivot[task_type],
            yerr=std_part,
            marker="o",
            label=task_type,
            capsize=3,
        )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by task type ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_by_task_type_{tag}.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument(
        "--benchmark",
        choices=["itbench_lite"],
        default="itbench_lite",
    )
    parser.add_argument("--per-level", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Number of repeated runs per experiment/model with different seeds.",
    )
    parser.add_argument(
        "--repeat-seed-stride",
        type=int,
        default=1000,
        help="Seed increment between repeated runs.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--itbench-domains", default="finops")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--agent-mode", choices=["simple", "tool"], default="tool")
    parser.add_argument(
        "--require-native-tool-support",
        type=int,
        choices=[0, 1],
        default=1,
        help="When agent-mode=tool, require native tool metadata support (1) or allow all models (0).",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--parallelism", default="auto")
    parser.add_argument("--max-parallelism", type=int, default=12)
    parser.add_argument("--model-parallelism", type=int, default=1)
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model list (required), e.g. qwen2.5-coder:7b,qwen2.5-coder:14b",
    )
    parser.add_argument("--task-types", default="")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument(
        "--tool-cache-mode",
        choices=["auto", "replay", "off"],
        default=str(os.environ.get("BENCHMARK_TOOL_CACHE_MODE", "auto")).strip().lower(),
    )
    parser.add_argument(
        "--tool-cache-file", default=os.environ.get("BENCHMARK_TOOL_CACHE_FILE", "")
    )
    parser.add_argument(
        "--task-isolation", choices=["thread", "process"], default="process"
    )
    parser.add_argument(
        "--scorer",
        choices=[
            "auto",
            "enhanced",
            "itbench_root_cause",
        ],
        default="auto",
    )
    parser.add_argument("--level2-steps", type=int, default=12)
    parser.add_argument("--level3-steps", type=int, default=20)
    parser.add_argument(
        "--ollama-chat-retries",
        type=int,
        default=env_int("BENCHMARK_OLLAMA_CHAT_RETRIES", 2),
    )
    parser.add_argument(
        "--ollama-max-request-timeout",
        type=int,
        default=env_int("BENCHMARK_OLLAMA_MAX_REQUEST_TIMEOUT", 240),
    )
    parser.add_argument(
        "--ollama-step-timeout",
        type=int,
        default=env_int("BENCHMARK_OLLAMA_STEP_TIMEOUT", 120),
    )
    parser.add_argument(
        "--ollama-image-step-timeout",
        type=int,
        default=env_int("BENCHMARK_OLLAMA_IMAGE_STEP_TIMEOUT", 420),
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default=os.environ.get("BENCHMARK_OLLAMA_KEEP_ALIVE", "10m"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("BENCHMARK_OLLAMA_TEMPERATURE", "0.2")),
        help="Sampling temperature for Ollama chat requests.",
    )
    parser.add_argument(
        "--docling-quiet",
        type=int,
        choices=[0, 1],
        default=env_int("BENCHMARK_DOCLING_QUIET", 1),
        help="Suppress noisy Docling warnings/logs (1=on, 0=off).",
    )
    parser.add_argument(
        "--linux-tesseract-mode",
        choices=["prompt", "auto", "skip"],
        default=str(os.environ.get("BENCHMARK_LINUX_TESSERACT_MODE", "prompt"))
        .strip()
        .lower(),
        help=(
            "Linux-only behavior when tesseract is missing: "
            "prompt user, auto-install, or skip installation."
        ),
    )
    args = parser.parse_args()

    global OLLAMA_CHAT_RETRIES
    global OLLAMA_MAX_REQUEST_TIMEOUT
    global OLLAMA_STEP_TIMEOUT
    global OLLAMA_IMAGE_STEP_TIMEOUT
    global OLLAMA_KEEP_ALIVE
    global OLLAMA_RANDOM_SEED
    global OLLAMA_TEMPERATURE
    OLLAMA_CHAT_RETRIES = max(0, int(args.ollama_chat_retries))
    OLLAMA_MAX_REQUEST_TIMEOUT = max(1, int(args.ollama_max_request_timeout))
    OLLAMA_STEP_TIMEOUT = max(0, int(args.ollama_step_timeout))
    OLLAMA_IMAGE_STEP_TIMEOUT = max(0, int(args.ollama_image_step_timeout))
    OLLAMA_KEEP_ALIVE = str(args.ollama_keep_alive or "").strip()
    OLLAMA_RANDOM_SEED = int(args.seed)
    OLLAMA_TEMPERATURE = float(args.temperature)
    configure_docling_quiet(bool(int(args.docling_quiet)))
    ensure_linux_tesseract(args.linux_tesseract_mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_file = args.tool_cache_file or str(out_dir / "tool_cache.json")
    configure_tool_cache(mode=args.tool_cache_mode, file_path=cache_file)
    print(f"Tool cache mode={args.tool_cache_mode} file={cache_file}")
    questions = load_selected_questions(args)
    scorer = resolve_scorer(args.scorer)
    print(f"Benchmark={args.benchmark} scorer={scorer}")

    if args.task_types:
        keep = {x.strip() for x in args.task_types.split(",") if x.strip()}
        questions = [q for q in questions if q.get("task_type") in keep]
        print(
            f"Filtered questions by task type {sorted(keep)}: {len(questions)} remain"
        )
    if args.max_questions > 0 and len(questions) > args.max_questions:
        random.Random(args.seed).shuffle(questions)
        questions = questions[: args.max_questions]
        print(f"Truncated questions to max {args.max_questions}")

    with open(out_dir / "selected_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    override_models = parse_models_arg(args.models)
    if not override_models:
        parser.error("--models is required and must contain at least one non-empty model name")
    run_names = ["all"]
    level_steps = {"2": args.level2_steps, "3": args.level3_steps}
    merged = []
    try:
        for run_name in run_names:
            spec = RunSpec(name=run_name, models=override_models)
            if args.agent_mode == "tool" and int(args.require_native_tool_support) == 1:
                native_models = []
                skipped_models = []
                for m in spec.models:
                    if model_has_native_tool_support(args.host, m):
                        native_models.append(m)
                    else:
                        skipped_models.append(m)
                if skipped_models:
                    print(
                        "[filter] skipped non-native-tool models: "
                        + ", ".join(skipped_models)
                    )
                if not native_models:
                    print(
                        f"[{spec.name}] no native-tool models remain after filtering; skipping run"
                    )
                    continue
                spec = RunSpec(name=spec.name, models=native_models)
            spec = RunSpec(
                name=spec.name, models=order_models_capability_first(spec.models)
            )
            print(f"[{spec.name}] model order: {', '.join(spec.models)}")
            if str(args.parallelism).lower() == "auto":
                parallelism = autotune_parallelism(
                    args.host,
                    spec.models[0],
                    args.timeout,
                    args.max_parallelism,
                )
            else:
                parallelism = int(args.parallelism)
            repeat_frames = []
            for repeat_idx in range(max(1, int(args.repeat_count))):
                repeat_seed = int(args.seed) + int(args.repeat_seed_stride) * repeat_idx
                OLLAMA_RANDOM_SEED = repeat_seed
                print(
                    f"[{spec.name}] repeat {repeat_idx + 1}/{args.repeat_count} seed={repeat_seed}"
                )
                df = run_benchmark(
                    args.host,
                    spec,
                    questions,
                    args.timeout,
                    out_dir,
                    args.agent_mode,
                    args.max_steps,
                    parallelism,
                    args.model_parallelism,
                    args.task_isolation,
                    scorer,
                    level_steps,
                    benchmark=args.benchmark,
                    repeat_idx=repeat_idx,
                    repeat_seed=repeat_seed,
                )
                repeat_frames.append(df)

            merged_run = pd.concat(repeat_frames, ignore_index=True)
            summarize(merged_run, out_dir, run_name, benchmark=args.benchmark)
            merged.append(merged_run)
        if merged:
            all_df = pd.concat(merged, ignore_index=True)
            all_df.to_csv(out_dir / "results_all.csv", index=False)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down workers...", flush=True)
        for child in multiprocessing.active_children():
            try:
                child.terminate()
            except Exception:  # noqa: BLE001
                pass
        for child in multiprocessing.active_children():
            try:
                child.join(timeout=1)
            except Exception:  # noqa: BLE001
                pass
        raise SystemExit(130) from None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down workers...", flush=True)
        for child in multiprocessing.active_children():
            try:
                child.terminate()
            except Exception:  # noqa: BLE001
                pass
        for child in multiprocessing.active_children():
            try:
                child.join(timeout=1)
            except Exception:  # noqa: BLE001
                pass
        raise SystemExit(130) from None
