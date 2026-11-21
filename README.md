# CloudServe Support System — Automated Triage, Grounded Response & Escalation Pipeline

**Author:** Deepak Vijayalu Krishnamurthy  
**Submission Deadline:** 22 November 2026, 11:59 PM IST  

---

## Overview

The **CloudServe Support System** is an auditable, governed customer support automation pipeline designed for CloudServe Solutions. Rather than exposing an ungrounded conversational chatbot, the system processes inbound support tickets across four channels (`email`, `chat`, `docs_comment`, `forum`) through a seven-stage workflow:

1. **Four-Channel Ingestion & Normalization (`src/pipeline.py`):** Standardizes ticket payloads across channels while preserving `customer_tier`, `customer_region`, and `language_fluency` metadata.
2. **Intent & Urgency Classification (`PR-01`):** Classifies tickets across 22 intent classes and 3 urgency levels (`high`, `medium`, `low`) with confidence scores via `OpenRouterClient` (with bounded retries, rate-limit pacing, response caching, and deterministic local fallback).
3. **Section-Aware Local Knowledge Base Retrieval (`LocalKBRetriever`):** Performs zero-compiler TF-IDF cosine similarity retrieval over section-aware chunks of CloudServe's 29 curated articles in `data/documentation.json`, tagging every chunk with its source `doc_id`.
4. **Policy & Confidence Routing:** Enforces mandatory escalation to Tier 2 for `must_not_auto_respond` intents and high-risk security, compliance, GDPR, data-residency, or billing-sensitive cases, while routing high-confidence routine tickets to grounded auto-response.
5. **Grounded Response & Tier 2 Handoff Generation (`PR-02`):** Generates responses with bracketed `[DOC-ID]` citations and explicit AI authorship disclosure, or produces structured Tier 2 handoff summaries when escalating.
6. **Pre- & Post-Response Guardrails:** Checks for prompt-injection attempts, PII/secret leakage, and citation grounding before finalizing outputs.
7. **1:1 Auditable Decision Logging:** Appends a structured decision record (`decision_id`, `timestamp`, `ticket_id`, `stage`, `prediction`, `sources_used`, `threshold_applied`, `action_taken`, `reason`, `guardrail_results`, `prompt_version`, `requirement_ids`) reconciling 1-to-1 with every processed ticket.

---

## Repository Structure

```text
cloudserve-support-system/
├── .github/
│   └── workflows/           # Continuous integration checks
├── data/
│   ├── documentation.json   # 29 curated Knowledge Base articles
│   ├── development_tickets.json
│   └── validation_tickets.json
├── docs/                    # Stage requirement specifications (PRD v1.0 / v2.0)
├── evaluation/
│   └── run_eval.py          # Standalone CLI evaluation & fairness audit harness
├── prompts/                 # Versioned prompt templates (PR-01, PR-02, PR-03)
├── src/
│   └── pipeline.py          # Core SupportPipeline, LocalKBRetriever & OpenRouterClient
├── tests/
│   └── test_pipeline.py     # Unit tests for normalization, routing, and logging
├── .env.example             # Environment variable template (no real secrets)
├── .gitignore               # Excludes .env, .venv/, storage/, and caches
├── README.md                # Setup and execution guide
└── requirements.txt         # Pinned Python dependencies
```

---

## Setup Instructions (Clean Checkout)

### 1. Prerequisites
* **Python 3.11** (tested on Windows with Python `3.11.9`; avoids C++ build-tool compilation requirements on Windows).

### 2. Create & Activate a Virtual Environment

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional for Live LLM Calls)
Copy `.env.example` to `.env` and add your OpenRouter API key if running with live LLM calls enabled:

```cmd
copy .env.example .env
```

> **Note:** Never commit `.env` or API keys to version control. When live API calls are disabled or rate-limited (`429`), the pipeline automatically uses its deterministic fallback mode without crashing.

---

## Running Tests & Evaluation

### 1. Run Unit Tests (`pytest`)
```bash
python -m pytest tests/test_pipeline.py -v
```

### 2. Run Unattended Validation Evaluation
`evaluation/run_eval.py` accepts `--input` and `--output` paths from the command line:

```bash
python evaluation/run_eval.py --input data/validation_tickets.json --output storage/validation_report.json
```

---

## Validation Summary (`80` Held-Out Validation Tickets)

* **Decision Log Reconciliation:** `80 / 80` (`100%`, zero crashes)
* **First-Contact Resolution (FCR):** `58.75%` (up from `42.0%` brief baseline / `43.8%` dev baseline)
* **Escalation Rate:** `41.25%`
* **Intent Accuracy (Labelled):** `87.50%`
* **Mean Latency (Deterministic Mode):** `0.54 ms` per ticket (`0.52 ms` median, `0.66 ms` p95)
* **Segment Fairness Note:** Disaggregated reporting across `customer_tier`, `language_fluency`, and `customer_region` highlights a lower FCR (`28.57%`) and higher escalation rate (`71.43%`) on `latin_america` (`n = 7`), flagged for targeted terminology expansion and human-in-the-loop oversight.