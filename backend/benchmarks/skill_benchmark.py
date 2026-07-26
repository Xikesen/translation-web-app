"""Benchmark the simultaneous-interpretation skill (before vs after).

Runs the same zh<->en sentences through two conditions against an Ollama model
and reports latency and accuracy so you can see what the skill changes:

  - baseline : the old "low-latency subtitle engine" system prompt + a plain
               "Source text / Output only the translation" user prompt.
  - skill    : the simultaneous-interpreter system prompt + the interpreter
               user prompt (Previous source / Previous target / New segment).

Both conditions use identical decoding options so the only variables are the
prompts (and, if you pass --baseline-model / --skill-model, the model itself).

Metrics
  - latency : wall-clock per request (mean / p50 / p95), plus tokens/sec from
              Ollama timing fields.
  - accuracy: chrF (character n-gram F-score, 0-100). chrF is language-agnostic
              and handles Chinese well without tokenization. Optional --judge
              adds an LLM-as-judge adequacy score (0-100).

Usage
  python skill_benchmark.py                         # single model, prompt A/B
  python skill_benchmark.py --repeats 3 --judge
  python skill_benchmark.py --baseline-model gemma-4-e4b:q8_0 \
                            --skill-model gemma-si:q8_0       # model A/B
  python skill_benchmark.py --out results.json

Requires the Ollama server running (see README / start_mac.sh).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma-4-e4b:q8_0")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DATA_PATH = Path(__file__).parent / "data" / "zh_en_pairs.json"

LANG_NAMES = {"en": "English", "zh": "Chinese (Simplified)"}
DIRECTION_LANGS = {"en2zh": ("en", "zh"), "zh2en": ("zh", "en")}


# --------------------------------------------------------------------------- #
# Prompts: keep these aligned with backend/services/manager_engine.py
# --------------------------------------------------------------------------- #
def baseline_system_prompt(source_lang: str, target_lang: str) -> str:
    source_name = LANG_NAMES.get(source_lang, source_lang)
    target_name = LANG_NAMES.get(target_lang, target_lang)
    return (
        "You are a low-latency subtitle translation engine for streaming speech.\n"
        f"Translate from {source_name} to {target_name}.\n"
        "Return only the translated subtitle text.\n"
        "Preserve named entities, dates, numbers, negation, conditions, punctuation, and units faithfully.\n"
        "Do not explain, annotate, add notes, or invent missing content."
    )


def baseline_user_prompt(source_text: str) -> str:
    return f"Source text: {source_text}\nOutput only the translation."


def skill_system_prompt(source_lang: str, target_lang: str) -> str:
    source_name = LANG_NAMES.get(source_lang, source_lang)
    target_name = LANG_NAMES.get(target_lang, target_lang)
    return (
        "You are a professional conference simultaneous interpreter.\n"
        f"Interpret from {source_name} to {target_name}.\n"
        "Interpret incrementally, segment by segment, as speech arrives; do not wait for a full sentence.\n"
        "Follow the source word order as much as the target language naturally allows, so output stays in sync with the speaker.\n"
        "Translate only the new source segment; never re-translate or repeat content already interpreted.\n"
        "Use any previous source/target only to stay consistent in terminology, names, tense, and register.\n"
        "If the new segment is incomplete, output a short natural partial interpretation; do not invent how it ends.\n"
        "Preserve named entities, dates, numbers, negation, conditions, punctuation, and units faithfully.\n"
        "Do not explain, annotate, add notes, or invent missing content.\n"
        "Keep it concise and in a natural spoken register, and output only the interpreted text."
    )


def skill_user_prompt(source_text: str) -> str:
    return (
        "You are a professional real-time interpreter.\n\n"
        "Translate the new source segment into natural target language.\n"
        "Use previous context only for consistency.\n"
        "Do not repeat previous translation.\n"
        "Keep the translation concise.\n"
        "If the segment is incomplete, produce a short interpretable partial translation.\n"
        "Do not add information that is not supported by the source.\n\n"
        "Previous source:\n(none)\n\n"
        "Previous target:\n(none)\n\n"
        f"New stable source segment:\n{source_text}\n\n"
        "Output only the translation of the new segment."
    )


@dataclass
class Condition:
    name: str
    model: str
    system_fn: object
    user_fn: object


# --------------------------------------------------------------------------- #
# chrF (character n-gram F-score)
# --------------------------------------------------------------------------- #
def _char_ngrams(text: str, n: int) -> Counter:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < n:
        return Counter()
    return Counter("".join(chars[i : i + n]) for i in range(len(chars) - n + 1))


def chrf_score(hypothesis: str, reference: str, *, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF in [0, 100]. Averages per-order precision/recall, then F-beta."""
    hyp = hypothesis.strip().lower()
    ref = reference.strip().lower()
    if not hyp or not ref:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    for n in range(1, max_n + 1):
        hyp_ng = _char_ngrams(hyp, n)
        ref_ng = _char_ngrams(ref, n)
        if not hyp_ng or not ref_ng:
            continue
        overlap = sum((hyp_ng & ref_ng).values())
        precisions.append(overlap / max(1, sum(hyp_ng.values())))
        recalls.append(overlap / max(1, sum(ref_ng.values())))

    if not precisions or not recalls:
        return 0.0
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    if avg_p == 0.0 and avg_r == 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * avg_p + avg_r
    if denom == 0.0:
        return 0.0
    f = (1 + beta_sq) * avg_p * avg_r / denom
    return round(100.0 * f, 2)


# --------------------------------------------------------------------------- #
# Ollama calls
# --------------------------------------------------------------------------- #
def call_ollama(client: httpx.Client, model: str, system: str, user: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "top_p": 0.9, "num_predict": 128},
    }
    started = time.perf_counter()
    response = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    response.raise_for_status()
    data = response.json()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    text = str(data.get("message", {}).get("content", "")).strip()
    eval_count = int(data.get("eval_count", 0) or 0)
    eval_duration_ns = int(data.get("eval_duration", 0) or 0)
    tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0
    return {
        "text": text,
        "elapsed_ms": elapsed_ms,
        "eval_count": eval_count,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


def judge_adequacy(client: httpx.Client, model: str, source: str, reference: str, hypothesis: str) -> float | None:
    """LLM-as-judge adequacy score in [0, 100]. Best-effort; returns None on parse failure."""
    system = (
        "You are a strict bilingual translation evaluator. "
        "Score how accurately the candidate conveys the meaning of the source, "
        "using the reference as a guide. 100 = perfect meaning and fidelity "
        "(numbers, names, negation preserved), 0 = wrong or missing meaning. "
        "Reply with ONLY an integer from 0 to 100."
    )
    user = f"Source:\n{source}\n\nReference:\n{reference}\n\nCandidate:\n{hypothesis}\n\nScore (0-100):"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 16},
    }
    try:
        response = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()
        content = str(response.json().get("message", {}).get("content", ""))
        # Strip any reasoning block, then take the last number (the final score).
        content = re.sub(r"<think>.*?</think>", " ", content, flags=re.DOTALL)
        numbers = re.findall(r"\d{1,3}", content)
        if not numbers:
            return None
        return float(max(0, min(100, int(numbers[-1]))))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def summarize(latencies: list[float], chrfs: list[float], judges: list[float], tps: list[float]) -> dict:
    return {
        "count": len(chrfs),
        "chrf_mean": round(statistics.mean(chrfs), 2) if chrfs else 0.0,
        "latency_mean_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "tokens_per_sec_mean": round(statistics.mean(tps), 2) if tps else 0.0,
        "judge_mean": round(statistics.mean(judges), 2) if judges else None,
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_condition(
    client: httpx.Client,
    condition: Condition,
    pairs: list[dict],
    repeats: int,
    use_judge: bool,
    judge_model: str | None = None,
) -> dict:
    rows: list[dict] = []
    by_direction: dict[str, dict[str, list[float]]] = {
        d: {"lat": [], "chrf": [], "judge": [], "tps": []} for d in DIRECTION_LANGS
    }

    for pair in pairs:
        direction = pair["direction"]
        source_lang, target_lang = DIRECTION_LANGS[direction]
        system = condition.system_fn(source_lang, target_lang)
        user = condition.user_fn(pair["source"])

        per_item_lat: list[float] = []
        last_text = ""
        last_tps = 0.0
        for _ in range(repeats):
            result = call_ollama(client, condition.model, system, user)
            per_item_lat.append(result["elapsed_ms"])
            last_text = result["text"]
            last_tps = result["tokens_per_sec"]

        chrf = chrf_score(last_text, pair["reference"])
        judge = (
            judge_adequacy(client, judge_model or condition.model, pair["source"], pair["reference"], last_text)
            if use_judge
            else None
        )
        median_lat = statistics.median(per_item_lat)

        by_direction[direction]["lat"].append(median_lat)
        by_direction[direction]["chrf"].append(chrf)
        by_direction[direction]["tps"].append(last_tps)
        if judge is not None:
            by_direction[direction]["judge"].append(judge)

        rows.append(
            {
                "id": pair["id"],
                "direction": direction,
                "source": pair["source"],
                "reference": pair["reference"],
                "hypothesis": last_text,
                "latency_ms": round(median_lat, 1),
                "chrf": chrf,
                "judge": judge,
                "tokens_per_sec": last_tps,
            }
        )
        print(f"  [{condition.name}] {pair['id']}  chrF={chrf:5.1f}  {median_lat:6.0f}ms  -> {last_text[:48]}")

    direction_summary = {}
    all_lat, all_chrf, all_judge, all_tps = [], [], [], []
    for direction, buckets in by_direction.items():
        if not buckets["chrf"]:
            continue
        direction_summary[direction] = summarize(buckets["lat"], buckets["chrf"], buckets["judge"], buckets["tps"])
        all_lat += buckets["lat"]
        all_chrf += buckets["chrf"]
        all_judge += buckets["judge"]
        all_tps += buckets["tps"]

    return {
        "name": condition.name,
        "model": condition.model,
        "overall": summarize(all_lat, all_chrf, all_judge, all_tps),
        "by_direction": direction_summary,
        "rows": rows,
    }


def print_comparison(baseline: dict, skill: dict) -> None:
    def fmt(value, suffix=""):
        return f"{value}{suffix}" if value is not None else "n/a"

    print("\n" + "=" * 78)
    print(f"COMPARISON   baseline={baseline['model']}   skill={skill['model']}")
    print("=" * 78)

    scopes = ["overall"] + [f"by_direction:{d}" for d in DIRECTION_LANGS]
    header = f"{'scope':<22}{'metric':<18}{'baseline':>12}{'skill':>12}{'delta':>12}"
    for scope in scopes:
        if scope == "overall":
            b, s = baseline["overall"], skill["overall"]
            label = "overall"
        else:
            d = scope.split(":")[1]
            b = baseline["by_direction"].get(d)
            s = skill["by_direction"].get(d)
            label = d
            if not b or not s:
                continue
        print("-" * 78)
        print(header)
        for metric, suffix, higher_better in [
            ("chrf_mean", "", True),
            ("judge_mean", "", True),
            ("latency_mean_ms", "ms", False),
            ("latency_p95_ms", "ms", False),
            ("tokens_per_sec_mean", "", True),
        ]:
            bv = b.get(metric)
            sv = s.get(metric)
            if bv is None and sv is None:
                continue
            delta = None
            if isinstance(bv, (int, float)) and isinstance(sv, (int, float)):
                delta = round(sv - bv, 2)
                arrow = "+" if delta > 0 else ""
                good = (delta > 0) == higher_better if delta != 0 else None
                mark = "" if good is None else ("  ✓" if good else "  ✗")
                delta_str = f"{arrow}{delta}{suffix}{mark}"
            else:
                delta_str = "n/a"
            print(f"{label:<22}{metric:<18}{fmt(bv, suffix):>12}{fmt(sv, suffix):>12}{delta_str:>12}")
    print("=" * 78)


def write_csv(baseline: dict, skill: dict, path: Path) -> None:
    """Write per-item rows for both conditions to one CSV, plus a *_summary.csv."""
    fieldnames = [
        "condition",
        "model",
        "id",
        "direction",
        "latency_ms",
        "chrf",
        "judge",
        "tokens_per_sec",
        "source",
        "reference",
        "hypothesis",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in (baseline, skill):
            for row in result["rows"]:
                writer.writerow(
                    {
                        "condition": result["name"],
                        "model": result["model"],
                        "id": row["id"],
                        "direction": row["direction"],
                        "latency_ms": row["latency_ms"],
                        "chrf": row["chrf"],
                        "judge": "" if row["judge"] is None else row["judge"],
                        "tokens_per_sec": row["tokens_per_sec"],
                        "source": row["source"],
                        "reference": row["reference"],
                        "hypothesis": row["hypothesis"],
                    }
                )

    summary_path = path.with_name(path.stem + "_summary.csv")
    metrics = ["chrf_mean", "judge_mean", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "tokens_per_sec_mean"]
    scopes = ["overall"] + list(DIRECTION_LANGS)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "metric", "baseline", "skill", "delta"])
        for scope in scopes:
            b = baseline["overall"] if scope == "overall" else baseline["by_direction"].get(scope)
            s = skill["overall"] if scope == "overall" else skill["by_direction"].get(scope)
            if not b or not s:
                continue
            for metric in metrics:
                bv, sv = b.get(metric), s.get(metric)
                if bv is None and sv is None:
                    continue
                delta = round(sv - bv, 2) if isinstance(bv, (int, float)) and isinstance(sv, (int, float)) else ""
                writer.writerow([scope, metric, bv, sv, delta])

    print(f"\nCSV written to {path}")
    print(f"Summary CSV written to {summary_path}")


def plot_comparison(baseline: dict, skill: dict, path: Path) -> None:
    """Grouped bar charts: chrF (and judge) accuracy + latency, by scope."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plot. Install with: pip install matplotlib")
        return

    scopes = ["overall"] + list(DIRECTION_LANGS)
    scopes = [s for s in scopes if (s == "overall") or (s in baseline["by_direction"] and s in skill["by_direction"])]

    def value(result: dict, scope: str, metric: str) -> float:
        bucket = result["overall"] if scope == "overall" else result["by_direction"].get(scope, {})
        val = bucket.get(metric)
        return float(val) if isinstance(val, (int, float)) else 0.0

    has_judge = baseline["overall"].get("judge_mean") is not None
    n_panels = 3 if has_judge else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    x = range(len(scopes))
    width = 0.35

    def grouped(ax, metric: str, title: str, ylabel: str) -> None:
        b_vals = [value(baseline, s, metric) for s in scopes]
        s_vals = [value(skill, s, metric) for s in scopes]
        bars_b = ax.bar([i - width / 2 for i in x], b_vals, width, label="baseline", color="#9aa7b8")
        bars_s = ax.bar([i + width / 2 for i in x], s_vals, width, label="skill", color="#3b82f6")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels(scopes)
        ax.legend()
        for bars in (bars_b, bars_s):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.0f}" if height >= 10 else f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    grouped(axes[0], "chrf_mean", "Accuracy (chrF, higher is better)", "chrF (0-100)")
    grouped(axes[1], "latency_mean_ms", "Latency (lower is better)", "mean latency (ms)")
    if has_judge:
        grouped(axes[2], "judge_mean", "LLM judge adequacy (higher is better)", "judge (0-100)")

    fig.suptitle(f"Simultaneous-interpretation skill: baseline ({baseline['model']}) vs skill ({skill['model']})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nChart written to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the simultaneous-interpretation skill (before vs after).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model used for both conditions (prompt A/B mode)")
    parser.add_argument("--baseline-model", default=None, help="override model for the baseline condition")
    parser.add_argument("--skill-model", default=None, help="override model for the skill condition")
    parser.add_argument("--repeats", type=int, default=1, help="requests per item; median is used for latency")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge adequacy score (slower)")
    parser.add_argument("--judge-model", default=None, help="model used as the LLM judge (default: the model under test)")
    parser.add_argument("--data", default=str(DATA_PATH), help="path to the zh<->en dataset JSON")
    parser.add_argument("--limit", type=int, default=0, help="limit number of pairs (0 = all)")
    parser.add_argument("--out", default=None, help="write full results JSON to this path")
    parser.add_argument("--csv", default=None, help="write per-item + summary CSV to this path")
    parser.add_argument("--plot", default=None, help="write a comparison chart PNG to this path (needs matplotlib)")
    args = parser.parse_args()

    dataset = json.loads(Path(args.data).read_text(encoding="utf-8"))
    pairs = dataset["pairs"]
    if args.limit > 0:
        pairs = pairs[: args.limit]

    baseline = Condition(
        name="baseline",
        model=args.baseline_model or args.model,
        system_fn=baseline_system_prompt,
        user_fn=baseline_user_prompt,
    )
    skill = Condition(
        name="skill",
        model=args.skill_model or args.model,
        system_fn=skill_system_prompt,
        user_fn=skill_user_prompt,
    )

    judge_model = args.judge_model if args.judge else None

    print(f"Ollama: {OLLAMA_URL}")
    print(f"Pairs: {len(pairs)}  repeats: {args.repeats}  judge: {args.judge}"
          + (f"  judge_model: {judge_model}" if judge_model else ""))

    with httpx.Client(timeout=120.0) as client:
        # Health check + warmup so the first real call isn't penalized by model load.
        try:
            client.get(f"{OLLAMA_URL}/api/tags").raise_for_status()
        except Exception as exc:
            print(f"\nError: cannot reach Ollama at {OLLAMA_URL}: {exc}")
            print("Start it with `ollama serve` (see start_mac.sh).")
            return 1
        print("Warming up models...")
        warmup_models = {baseline.model, skill.model}
        if judge_model:
            warmup_models.add(judge_model)
        for model in warmup_models:
            try:
                call_ollama(client, model, "You are a translator.", "Translate to Chinese: hello")
            except Exception as exc:
                print(f"\nError: model '{model}' not usable: {exc}")
                print(f"Pull/build it first (e.g. `ollama pull {model}`).")
                return 1

        print("\nRunning baseline condition...")
        baseline_result = run_condition(client, baseline, pairs, args.repeats, args.judge, judge_model)
        print("\nRunning skill condition...")
        skill_result = run_condition(client, skill, pairs, args.repeats, args.judge, judge_model)

    print_comparison(baseline_result, skill_result)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(
            json.dumps(
                {"baseline": baseline_result, "skill": skill_result, "config": vars(args)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFull results written to {out_path}")

    if args.csv:
        write_csv(baseline_result, skill_result, Path(args.csv))

    if args.plot:
        plot_comparison(baseline_result, skill_result, Path(args.plot))

    return 0


if __name__ == "__main__":
    sys.exit(main())
