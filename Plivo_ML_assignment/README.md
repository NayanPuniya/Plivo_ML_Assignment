# Ultra-fast STT Post-Processor (Interview Assignment)

**Goal:** Build a ≤30 ms (p95) text post-processor that fixes **emails**, **numbers (Indian style)**, **Indian names**, and **punctuation** on noisy speech transcripts.  
This is **Option 2** from the brief. You will implement a two-stage system: **rules (candidate generation) + learned ranker (DistilBERT masked-LM in ONNX, quantized)**.

⏱️ **Timebox:** 2 hours.  
📖 **Open book:** You may consult the internet and docs.

---

## What’s provided

- **Data**
  - `data/noisy_transcripts.jsonl` — ~80 synthetic noisy lines (id, text)
  - `data/gold.jsonl` — gold references (id, text)
  - `data/names_lexicon.txt` — small Indian names lexicon
  - `data/misspell_map.json` — some common misspellings

- **Starter code**
  - `src/rules.py` — Stage 1: rule-based candidate generation (emails, numbers, currency, names)
  - `src/ranker_onnx.py` — Stage 2: DistilBERT pseudo-likelihood ranker (ONNX + PyTorch fallback)
  - `src/export_onnx.py` — Export and **quantize** DistilBERT to ONNX INT8
  - `src/metrics.py` — WER/CER + Punctuation F1 + Email/Number/Name metrics
  - `src/utils.py` — helpers
  - `run_pipeline.py` — processes `noisy_transcripts.jsonl` to `out/corrected.jsonl`
  - `evaluate.py` — prints metrics comparing predictions to gold
  - `measure_latency.py` — measures p50/p95 latency for the post-processor
  - `score.sh` — one-command pipeline: export → run → eval → latency

- **Environment**
  - `requirements.txt` — CPU-friendly stack (transformers, torch, onnxruntime, jiwer, rapidfuzz)

---

## Your tasks (choose A or B)

### A) **Rules + ONNX-quantized DistilBERT ranker (recommended)**
1. At the repository root, create a folder named `models`.
2. Add more entries to `gold.jsonl` and `noise_transcripts.jsonl` - both file should have 80 samples; a few starting sample records are already included for reference.
3. Install deps and export the model:
   ```bash
   pip install -r requirements.txt
   bash score.sh  # does: export+quantize → run → eval → latency
   ```
4. Improve the **rules** in `src/rules.py` (email/number/name/punctuation heuristics) and/or **ranking** in `src/ranker_onnx.py`.
5. Show **metric gains** on at least 2 of: WER/CER, EmailAcc, NumberAcc, NameF1, PunctuationF1.
6. Show **latency**: p95 ≤ 30 ms (CPU) via `python measure_latency.py` (you can tweak max length, batch masking, etc.).

### B) **(Optional)** Replace ranker with your own tiny tagger or alternative ranker, **but keep ≤30 ms**. You may still use the provided metrics and latency harness.

---

## Metrics

We report:
- **WER**, **CER** (via `jiwer`)
- **PunctuationF1** (for `[.,?]`)
- **EmailAcc** (exact list match)
- **NumberAcc** (normalized numeric string equality, Indian grouping tolerant)
- **NameF1** (lexicon-based fuzzy match)

Run:
```bash
python evaluate.py --pred out/corrected.jsonl --gold data/gold.jsonl --names data/names_lexicon.txt
```

---

## Latency

Measure per-utterance latency (includes rules + ranker + minimal punctuation heuristic):
```bash
python measure_latency.py --onnx models/distilbert-base-uncased.int8.onnx --runs 100 --warmup 10
```
Target: **p95 ≤ 30 ms** on CPU with `max_length=64`. You can:
- keep utterances short (truncate sensibly),
- quantize to INT8 (already in `score.sh`),
- ensure batch-masked scoring (implemented) and avoid large candidate sets.

---

## Tips & next steps (what we look for)

- Target **emails/numbers/names** first—these drive the biggest quality wins in commerce/haggle contexts.
- Keep your candidate set small (≤5) but diverse.
- Treat punctuation minimally (end-of-utterance `.`/`?`) unless you have time.
- Explain **why** your changes help; add a tiny ablation (e.g., with/without a rule, different α for any weighting you introduce).

---

## Deliverables

- A short Loom (≤5 min) walking through: approach, key changes, metrics, latency, and next steps.
- Committed code (or a Colab) that we can run:
  ```bash
  bash score.sh
  ```

Good luck!
