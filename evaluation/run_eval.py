from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import SupportPipeline, load_tickets  # noqa: E402


def summarize_results(results: List[Dict[str, Any]], decision_count: int) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total_tickets": 0, "decision_log_reconciled": decision_count == 0}

    fcr_count = sum(1 for r in results if r["resolved_first_contact"])
    esc_count = sum(1 for r in results if r["route"] == "escalate")
    latencies = sorted(float(r["latency_ms"]) for r in results)
    p95_idx = min(total - 1, int(0.95 * (total - 1)))

    # Intent accuracy when ground-truth labels exist in input records.
    labelled = [r for r in results if r.get("expected_intent")]
    intent_acc = (
        round(sum(1 for r in labelled if r["intent"] == r["expected_intent"]) / len(labelled), 4)
        if labelled
        else None
    )

    segments: Dict[str, Dict[str, Any]] = {}
    for seg_field in ("customer_tier", "customer_region", "language_fluency"):
        seg_map: Dict[str, Any] = {}
        for val in sorted({r[seg_field] for r in results}):
            sub = [r for r in results if r[seg_field] == val]
            seg_map[val] = {
                "count": len(sub),
                "fcr_rate": round(sum(1 for x in sub if x["resolved_first_contact"]) / len(sub), 4),
                "escalation_rate": round(sum(1 for x in sub if x["route"] == "escalate") / len(sub), 4),
            }
        segments[seg_field] = seg_map

    return {
        "total_tickets": total,
        "decision_log_entries": decision_count,
        "decision_log_reconciled": decision_count == total,
        "fcr_rate": round(fcr_count / total, 4),
        "escalation_rate": round(esc_count / total, 4),
        "intent_accuracy_on_labelled": intent_acc,
        "latency_ms": {
            "mean": round(sum(latencies) / total, 2),
            "median": latencies[total // 2],
            "p95": latencies[p95_idx],
        },
        "segments": segments,
    }


def run_evaluation(input_path: str, output_path: str, docs_path: str = "data/documentation.json", use_llm: bool = False) -> Dict[str, Any]:
    in_file = Path(input_path)
    out_file = Path(output_path)
    kb_file = Path(docs_path)
    if not kb_file.is_absolute():
        kb_file = ROOT / kb_file

    raw_tickets = load_tickets(in_file)
    docs = json.loads(kb_file.read_text(encoding="utf-8"))

    pipeline = SupportPipeline(docs=docs, use_llm=use_llm)
    results: List[Dict[str, Any]] = []
    for raw in raw_tickets:
        rec = pipeline.process_ticket(raw)
        expected_intent = (raw.get("labels") or {}).get("intent")
        if expected_intent:
            rec["expected_intent"] = expected_intent
        results.append(rec)

    summary = summarize_results(results, len(pipeline.decision_logs))
    payload = {
        "summary": summary,
        "results": results,
        "decision_logs": pipeline.decision_logs,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CloudServe support evaluation harness.")
    parser.add_argument("--input", required=True, help="Path to input tickets JSON file.")
    parser.add_argument("--output", required=True, help="Path to output JSON report file.")
    parser.add_argument("--docs", default="data/documentation.json", help="Path to documentation.json.")
    parser.add_argument("--use-llm", action="store_true", help="Enable live OpenRouter calls during classification.")
    args = parser.parse_args()
    summary = run_evaluation(args.input, args.output, args.docs, use_llm=args.use_llm)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()