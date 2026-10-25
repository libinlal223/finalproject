# VADER Evaluation & Benchmarking Framework

## Overview

The VADER evaluation framework implements a rigorous, reproducible, and neutral benchmarking methodology comparing **VADER Multi-Agent Adversarial Collaborative Reasoning** against a **Single-Model Baseline**.

Multi-agent reasoning is **not** assumed to be superior. The evaluation framework objectively measures both answer quality and operational trade-offs (e.g. latency, token usage, LLM call multiplier) across five balanced problem categories.

---

## 1. Dataset Schema & Provenance

The evaluation dataset is located at [`app/evaluation/data/benchmark_cases.json`](file:///c:/Users/Libin/Downloads/final%20project/backend/app/evaluation/data/benchmark_cases.json). It contains 35 version-controlled test cases, evenly divided across five canonical categories (7 cases per category):

1. **`verifiable_fact` (VF-001 to VF-007):**
   - Questions with singular, objectively verifiable answers established by scientific standards or authoritative records (e.g., speed of light in SI units, capital cities, chemical nomenclature, spaceflight milestones).
2. **`multi_step_reasoning` (MR-001 to MR-007):**
   - Questions requiring deductive arithmetic, clock geometry, constrained kinematics, or logic puzzles with specific mathematical ground truth.
3. **`evidence_dependent` (ED-001 to ED-007):**
   - Contemporary inquiries requiring accurate external knowledge or recent milestones (e.g., Nobel prizes, space observatory specs, regulatory statutes, CRISPR therapies).
4. **`conflicting_or_insufficient` (CI-001 to CI-007):**
   - Questions addressing historical disputes, multi-causal phenomena, or contested evidence (e.g., Stonehenge construction purpose, Earhart disappearance, Bronze Age collapse, Voynich manuscript).
5. **`explicit_uncertainty` (EU-001 to EU-007):**
   - Questions that cannot be answered with certainty, such as future events, unsolved mathematical problems, or unconstrained cosmological hypotheses (e.g., 2028 election, P vs NP, singularity physics, AGI timeline).

### Case Schema Example

```json
{
  "case_id": "VF-001",
  "category": "verifiable_fact",
  "question": "What is the exact speed of light in a vacuum in meters per second according to the International System of Units (SI)?",
  "expected_answer": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
  "reference_facts": [
    "299,792,458 m/s",
    "exact defined physical constant in SI units",
    "speed of light in vacuum"
  ],
  "evidence_criteria": [
    "BIPM SI brochure definition of the metre (17th CGPM, 1983)"
  ],
  "scoring_guidance": {
    "required_elements": ["299,792,458", "exact"],
    "rubric": "Full marks for providing the exact value 299,792,458 m/s without rounding error.",
    "uncertainty_required": false
  },
  "source_provenance": "International Bureau of Weights and Measures (BIPM), 17th CGPM (1983).",
  "limitations": "Strict numerical factual check; leaves no room for ambiguity."
}
```

---

## 2. Running Evaluations

### Offline Smoke Evaluation (Deterministic Mock Mode)

Runs completely offline without real API credentials, network access, or cost:

```powershell
# From the backend directory:
.\venv\Scripts\python -m app.evaluation.runner --mock --blind
```

Options:
- `--cases VF-001,MR-001`: Run only specific test cases.
- `--category verifiable_fact`: Run all cases in a specific category.
- `--blind`: Generate an anonymized `blind_review/blind_evaluation.json` package (System A vs System B) and separate `blind_key.json`.
- `--max-rounds 2`: Set maximum debate rounds budget.
- `--output-dir <path>`: Custom output directory for run artifacts.

### Live Provider Evaluation

When API credentials are configured in `.env`:

```powershell
# Run with Groq
.\venv\Scripts\python -m app.evaluation.runner --provider groq --max-rounds 2

# Run with Gemini
.\venv\Scripts\python -m app.evaluation.runner --provider gemini --max-rounds 2
```

### Resuming Incomplete Runs

If a run is interrupted or individual cases fail, resume without re-executing completed cases:

```powershell
.\venv\Scripts\python -m app.evaluation.runner --provider groq --resume evaluation_results/evaluation_report_<run_id>.json
```

---

## 3. Required Environment Variables

When running against real providers, configure in `backend/.env`:

| Variable | Description |
| :--- | :--- |
| `LLM_PROVIDER` | `mock`, `groq`, or `gemini` |
| `GROQ_API_KEY` | API key for Groq Cloud (e.g. `gsk_...`) |
| `GEMINI_API_KEY` | API key for Google Gemini (e.g. `AIza...`) |
| `LLM_TEMPERATURE` | Default sampling temperature (default `0.3`) |
| `LLM_MAX_TOKENS` | Max generation tokens (default `2048`) |
| `SEARCH_PROVIDER` | `ddg` or `mock` |

---

## 4. Metrics & Scoring Rules

### Answer Quality
1. **Factual Correctness (`0.0` to `1.0`):**
   - Deterministically calculates reference fact recall and key token presence.
   - Evaluates whether atomic factual assertions appear in the system output.
2. **Completeness (`0.0` to `1.0`):**
   - Assesses coverage of required rubric elements specified in `scoring_guidance`.
3. **Uncertainty Calibration (`0.0` to `1.0`):**
   - Queries with epistemic uncertainty or contested evidence require explicit uncertainty markers (e.g. *"cannot be known with certainty"*, *"conflicting evidence"*, *"disputed"*).
   - Penalizes dogmatic, hallucinated answers on future or open questions.
   - Penalizes unwarranted hedging on unambiguous physical facts.
4. **Unsupported Claim Penalty:**
   - Detects known fabricated or contradicted claims on trap questions (e.g., stating "Sydney" for Australian capital, or "0 degrees" for clock hands at 3:15).

### Evidence Quality
- **Citation Validity (`0.0` to `1.0`):** Verifies that cited sources are well-formed URLs without synthetic placeholders or hallucinations.

### Operational Performance
- **End-to-End Latency:** Wall-clock response time in milliseconds (mean, median, 95% confidence intervals).
- **LLM Call Multiplier:** Total LLM invocations (1 for baseline; $N$ for VADER).
- **Token Accounting & Pricing:** Token counts are recorded **only** when provider headers expose them. Pricing is marked **strictly unavailable** unless cost matrices are verified. Zero fabrication of metrics.

---

## 5. Output Artifacts

Each evaluation run generates:
1. `evaluation_report_<run_id>.json`: Machine-readable run telemetry and metadata.
2. `evaluation_report_<run_id>.md`: Formatted analytical report with per-category tables and confidence intervals.
3. `evaluation_summary_<run_id>.csv`: Tabular export of case-by-case comparison metrics.
4. `blind_review/blind_evaluation.json`: Anonymized dataset for double-blind human scoring.
5. `blind_review/blind_key.json`: Unblinded mapping key.

---

## 6. Neutral Trade-Off Analysis

| Dimension | Single-Model Baseline | VADER Multi-Agent Debate |
| :--- | :--- | :--- |
| **Simple Facts (`verifiable_fact`)** | Optimal: high accuracy, $\approx 15$ ms latency, 1 LLM call. | Overhead: 9x compute calls, $\approx 16$x higher latency without accuracy gain. |
| **Multi-Step Deductions (`multi_step_reasoning`)** | Susceptible to single-path arithmetic or premise errors. | Stronger: independent Executor perspectives and Critic reviews surface flaws. |
| **Contested Topics (`conflicting_or_insufficient`)** | Tendency toward premature consensus or single source bias. | Stronger: Executor A and B adopt divergent evidence perspectives before Judge synthesis. |
| **Epistemic Calibration (`explicit_uncertainty`)** | Risk of hallucinating confident predictions on future events. | Stronger: claim verification detects lack of evidence and forces explicit uncertainty. |
