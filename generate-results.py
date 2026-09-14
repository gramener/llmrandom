#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28", "python-dotenv>=1.0"]
# ///
"""Build results.json from the legacy experiment and OpenRouter runs.

The file is checkpointed after every successful API response. Re-running the
script fills only missing samples, so interrupted runs resume without replacing
already-recorded observations.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.json"
LEGACY = ROOT / "llmrandom.json"
NOTEBOOK = ROOT.parent / "ipython-notebooks" / "llm-random-numbers.ipynb"
PROMPT_ID = "random-number-0-100-v1"
PROMPT = "Pick a random number from 0 - 100. Write ONLY the number NOTHING ELSE"
TEMPERATURES = [i / 10 for i in range(11)]
SAMPLES = 200
MAX_TOKENS = 2
CONCURRENCY = 20
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CATALOG_URL = "https://openrouter.ai/api/v1/models"
MODELS = {
    "openai/gpt-4.1-nano": {"reasoning_effort": None, "temperatures": True},
    "openai/gpt-5.6-luna": {"reasoning_effort": "none", "temperatures": "if-supported"},
}
LEGACY_MODELS = {
    "O": {"model": "gpt-3.5-turbo", "provider": "openai", "gateway": "direct-openai"},
    "C": {"model": "claude-3-haiku-20240307", "provider": "anthropic", "gateway": "direct-anthropic"},
    "G": {"model": "gemini-1.0-pro-001", "provider": "google", "gateway": "vertex-ai-europe-west2"},
}
EXACT_NUMBER = re.compile(r"(?:100|[0-9]{1,2})\Z")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info(path: Path) -> dict[str, str] | None:
    try:
        repo = subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"], text=True
        ).strip()
        rel = str(path.relative_to(repo))
        out = subprocess.check_output(
            ["git", "-C", repo, "log", "-1", "--format=%H%x09%aI", "--", rel], text=True
        ).strip()
        if not out:
            return None
        commit, committed_at = out.split("\t", 1)
        return {"commit": commit, "committed_at": committed_at}
    except (subprocess.CalledProcessError, ValueError):
        return None


def empty_results() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "metadata": {
            "description": "Repeated LLM responses to prompts, grouped by model and sampling configuration.",
            "updated_at_utc": now(),
            "default_sample_target": SAMPLES,
            "temperature_grid": TEMPERATURES,
            "currency": "USD",
            "cost_note": "OpenRouter costs are summed from each response's usage.cost; legacy direct-API costs were not retained and are unknown.",
        },
        "prompts": {
            PROMPT_ID: {
                "text": PROMPT,
                "sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "validation": "Output must be exactly one integer from 0 through 100, with no other text.",
            }
        },
        "sources": {},
        "model_snapshots": {},
        "runs": {},
        "cost_summary": {},
    }


def load_results() -> dict[str, Any]:
    if not RESULTS.exists():
        return empty_results()
    data = json.loads(RESULTS.read_text())
    if data.get("schema_version") != 2:
        raise RuntimeError(f"Unsupported {RESULTS.name} schema_version={data.get('schema_version')}")
    return data


def save_results(data: dict[str, Any]) -> None:
    data["metadata"]["updated_at_utc"] = now()
    data["metadata"]["generator"] = {
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "openrouter_endpoint": OPENROUTER_URL,
        "concurrency": CONCURRENCY,
    }
    costs = [
        run.get("usage", {}).get("cost_usd")
        for run in data["runs"].values()
        if run.get("usage") and run.get("usage", {}).get("cost_usd") is not None
    ]
    openrouter_costs = [
        run.get("usage", {}).get("cost_usd")
        for run in data["runs"].values()
        if run.get("gateway") == "openrouter"
        and run.get("usage")
        and run.get("usage", {}).get("cost_usd") is not None
    ]
    data["cost_summary"] = {
        "currency": "USD",
        "known_cost_usd": round(sum(costs), 12),
        "openrouter_cost_usd": round(sum(openrouter_costs), 12),
        "legacy_direct_api_cost_usd": None,
        "legacy_cost_note": "The 2024 source data contains outputs but no token usage or billing data.",
    }
    tmp = RESULTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(RESULTS)


def summarize(outputs: list[str | None]) -> dict[str, Any]:
    present = [x for x in outputs if x is not None]
    valid = [int(x) for x in present if EXACT_NUMBER.fullmatch(x)]
    counts = Counter(valid)
    max_count = max(counts.values(), default=0)
    return {
        "samples_recorded": len(present),
        "valid_samples": len(valid),
        "invalid_samples": len(present) - len(valid),
        "distinct_values": len(counts),
        "counts": {str(k): counts[k] for k in sorted(counts)},
        "modes": [k for k in sorted(counts) if counts[k] == max_count] if counts else [],
        "mode_count": max_count,
        "mean": round(statistics.fmean(valid), 6) if valid else None,
        "median": statistics.median(valid) if valid else None,
    }


def legacy_run_id(code: str, temperature: float) -> str:
    return f"legacy-2024:{LEGACY_MODELS[code]['model']}:temperature={temperature:.2f}"


def import_legacy(data: dict[str, Any]) -> None:
    raw = json.loads(LEGACY.read_text())
    source_hash = sha256(LEGACY)
    source = data["sources"].get("legacy-llmrandom-json")
    if source and source.get("sha256") == source_hash:
        return

    expected = len(LEGACY_MODELS) * len(TEMPERATURES) * SAMPLES
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} legacy observations, found {len(raw)}")

    for code, model_meta in LEGACY_MODELS.items():
        for temperature in TEMPERATURES:
            outputs = [raw[f"{code},{temperature:.2f},{attempt}"] for attempt in range(SAMPLES)]
            run_id = legacy_run_id(code, temperature)
            data["runs"][run_id] = {
                "source": "legacy-llmrandom-json",
                "prompt_id": PROMPT_ID,
                **model_meta,
                "parameters": {"temperature": temperature, "max_tokens": MAX_TOKENS},
                "sample_target": SAMPLES,
                "outputs_complete": True,
                "outputs": outputs,
                "summary": summarize(outputs),
                "usage": None,
                "cost_usd": None,
            }

    data["sources"]["legacy-llmrandom-json"] = {
        "path": "llmrandom.json",
        "sha256": source_hash,
        "git": git_info(LEGACY),
        "generator_notebook": str(NOTEBOOK),
        "generator_notebook_sha256": sha256(NOTEBOOK),
        "generator_notebook_git": git_info(NOTEBOOK),
        "imported_at_utc": now(),
        "request_design": {
            "samples_per_condition": SAMPLES,
            "temperatures": TEMPERATURES,
            "max_tokens": MAX_TOKENS,
            "model_codes": {k: v["model"] for k, v in LEGACY_MODELS.items()},
        },
    }
    save_results(data)
    print(f"Imported {len(raw)} legacy observations from {LEGACY.name}", flush=True)


def api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_PERSONAL_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env (OPENROUTER_PERSONAL_API_KEY is also accepted).")
    return key


async def fetch_catalog(client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    response = await client.get(CATALOG_URL)
    response.raise_for_status()
    wanted = set(MODELS)
    found = {m["id"]: m for m in response.json()["data"] if m["id"] in wanted}
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"Models absent from OpenRouter catalog: {sorted(missing)}")
    return found


def snapshot_model(data: dict[str, Any], model: dict[str, Any]) -> None:
    pricing = model.get("pricing", {})
    per_million = {}
    for key, value in pricing.items():
        if key == "overrides":
            continue
        try:
            per_million[key] = float(value) * 1_000_000
        except (TypeError, ValueError):
            pass
    data["model_snapshots"][model["id"]] = {
        "fetched_at_utc": now(),
        "supported_parameters": model.get("supported_parameters", []),
        "pricing_usd_per_token": pricing,
        "pricing_usd_per_million_tokens": per_million,
        "top_provider": model.get("top_provider"),
    }


def run_id(model: str, temperature: float | None, reasoning_effort: str | None) -> str:
    short_model = model.split("/", 1)[-1]
    t = "default" if temperature is None else f"{temperature:.2f}"
    r = reasoning_effort or "default"
    return f"openrouter:{short_model}:temperature={t}:reasoning={r}"


def ensure_run(
    data: dict[str, Any], model: str, temperature: float | None, reasoning_effort: str | None
) -> tuple[str, dict[str, Any]]:
    rid = run_id(model, temperature, reasoning_effort)
    run = data["runs"].get(rid)
    if run is None:
        params: dict[str, Any] = {"max_tokens": MAX_TOKENS, "temperature": temperature}
        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}
        run = {
            "source": "generated-by-generate-results.py",
            "prompt_id": PROMPT_ID,
            "model": model,
            "provider": "openrouter-routing",
            "gateway": "openrouter",
            "parameters": params,
            "temperature_note": (
                "Explicit temperature sent to OpenRouter."
                if temperature is not None
                else "Temperature omitted because the OpenRouter model catalog does not list temperature as supported."
            ),
            "sample_target": SAMPLES,
            "started_at_utc": now(),
            "finished_at_utc": None,
            "outputs_complete": False,
            "outputs": [None] * SAMPLES,
            "summary": summarize([None] * SAMPLES),
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_prompt_tokens": 0,
                "cost_usd": 0.0,
            },
            "provider_counts": {},
        }
        data["runs"][rid] = run
        save_results(data)
    return rid, run


def merge_usage(run: dict[str, Any], response: dict[str, Any]) -> None:
    usage = response.get("usage") or {}
    total = run["usage"]
    total["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    total["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    total["total_tokens"] += int(usage.get("total_tokens") or 0)
    total["reasoning_tokens"] += int(
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    )
    total["cached_prompt_tokens"] += int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    )
    if usage.get("cost") is not None:
        total["cost_usd"] = round(total["cost_usd"] + float(usage["cost"]), 12)
    provider = response.get("provider")
    if provider:
        run["provider_counts"][provider] = run["provider_counts"].get(provider, 0) + 1


async def one_sample(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    key: str,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    attempt: int,
) -> tuple[int, str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with sem:
        for retry in range(7):
            try:
                response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
                if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}", request=response.request, response=response
                    )
                response.raise_for_status()
                body = response.json()
                text = body["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise RuntimeError(f"Non-text response for {model}: {text!r}")
                return attempt, text, body
            except (httpx.HTTPError, KeyError, RuntimeError) as exc:
                if retry == 6:
                    raise RuntimeError(f"{model} attempt {attempt} failed after retries: {exc}") from exc
                await asyncio.sleep(min(2**retry, 20))
    raise AssertionError("unreachable")


async def fill_run(
    data: dict[str, Any],
    client: httpx.AsyncClient,
    key: str,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
) -> None:
    rid, run = ensure_run(data, model, temperature, reasoning_effort)
    missing = [i for i, output in enumerate(run["outputs"]) if output is None]
    if not missing:
        print(f"SKIP {rid} (complete; cost ${run['usage']['cost_usd']:.6f})", flush=True)
        return

    print(f"RUN  {rid}: {len(missing)} missing of {SAMPLES}", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        asyncio.create_task(one_sample(client, sem, key, model, temperature, reasoning_effort, i))
        for i in missing
    ]
    completed_now = 0
    try:
        for task in asyncio.as_completed(tasks):
            attempt, text, body = await task
            run["outputs"][attempt] = text
            merge_usage(run, body)
            run["summary"] = summarize(run["outputs"])
            run["outputs_complete"] = all(x is not None for x in run["outputs"])
            if run["outputs_complete"]:
                run["finished_at_utc"] = now()
            save_results(data)
            completed_now += 1
            if completed_now % 25 == 0 or run["outputs_complete"]:
                print(
                    f"  {run['summary']['samples_recorded']}/{SAMPLES}; "
                    f"valid={run['summary']['valid_samples']}; cost=${run['usage']['cost_usd']:.6f}",
                    flush=True,
                )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate-only", action="store_true", help="Import legacy data without API calls")
    parser.add_argument("--model", choices=MODELS, help="Run only this OpenRouter model")
    parser.add_argument("--temperature", type=float, help="Run only this explicit temperature")
    args = parser.parse_args()

    data = load_results()
    import_legacy(data)
    if args.migrate_only:
        print(f"Wrote {RESULTS}", flush=True)
        return

    key = api_key()
    timeout = httpx.Timeout(90, connect=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        catalog = await fetch_catalog(client)
        for model in MODELS:
            snapshot_model(data, catalog[model])
        save_results(data)

        selected = [args.model] if args.model else list(MODELS)
        for model in selected:
            spec = MODELS[model]
            supported = set(catalog[model].get("supported_parameters", []))
            has_temperature = "temperature" in supported
            if spec["temperatures"] is True and not has_temperature:
                raise RuntimeError(f"{model} unexpectedly does not support temperature")
            temperatures: list[float | None] = TEMPERATURES if has_temperature else [None]
            if args.temperature is not None:
                if not has_temperature:
                    print(f"SKIP {model}: temperature is not supported", flush=True)
                    continue
                temperatures = [args.temperature]
            for temperature in temperatures:
                await fill_run(
                    data,
                    client,
                    key,
                    model,
                    temperature,
                    spec["reasoning_effort"],
                )

    print(f"Done. OpenRouter recorded cost: ${data['cost_summary']['openrouter_cost_usd']:.6f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
