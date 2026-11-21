from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import SupportPipeline, normalize_ticket
from evaluation.run_eval import run_evaluation


def test_four_channel_normalization():
    for idx, ch in enumerate(["email", "chat", "docs_comment", "forum"]):
        t = normalize_ticket(
            {
                "ticket_id": f"T-{idx}",
                "channel": ch,
                "subject": "Should be blank on chat" if ch == "chat" else "Subject line",
                "body": "My deployment keeps rolling back after health check.",
                "customer_tier": "business",
                "customer_region": "europe",
                "language_fluency": "fluent",
            }
        )
        assert t.channel == ch
        assert t.subject == "" if ch == "chat" else "Subject line"


def test_mandatory_escalation_and_decision_log_parity(tmp_path: Path):
    docs_path = ROOT / "data" / "documentation.json"
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    pipe = SupportPipeline(docs=docs, use_llm=False)

    sec_out = pipe.process_ticket(
        {
            "ticket_id": "SEC-1",
            "channel": "email",
            "subject": "Possible account compromise",
            "body": "A former employee still has access and we see suspicious activity in audit log.",
            "customer_tier": "enterprise",
            "customer_region": "europe",
            "language_fluency": "fluent",
        }
    )
    assert sec_out["route"] == "escalate"
    assert sec_out["resolved_first_contact"] is False
    assert sec_out["handoff_summary"] is not None
    assert len(pipe.decision_logs) == 1

    sample_in = tmp_path / "sample.json"
    sample_out = tmp_path / "report.json"
    sample_in.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "CHK-1",
                    "channel": "chat",
                    "subject": "",
                    "body": "How do I revert to an earlier revision after a bad release?",
                    "customer_tier": "standard",
                    "customer_region": "asia_pacific",
                    "language_fluency": "non_fluent",
                }
            ]
        ),
        encoding="utf-8",
    )
    summary = run_evaluation(str(sample_in), str(sample_out), str(docs_path), use_llm=False)
    assert summary["total_tickets"] == 1
    assert summary["decision_log_reconciled"] is True