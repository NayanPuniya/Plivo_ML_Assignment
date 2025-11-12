# Candidate Brief — Ultra-fast STT Post-Processor

**Objective:** In 2 hours, improve the accuracy of a post-processor for noisy transcripts using **rules + a learned ranker** while keeping **p95 ≤ 30 ms** latency.

## Steps
1. `pip install -r requirements.txt`
2. `bash score.sh` to export/quantize the ranker, run the pipeline, evaluate metrics, and measure latency.
3. Improve either or both of:
   - `src/rules.py` (emails, numbers, currency, names, basic punctuation)
   - `src/ranker_onnx.py` (scoring strategy, token caps, batching)
4. Re-run `bash score.sh` and capture metric gains.
5. Record a ≤5 min Loom summarizing your changes, results, and what you would ship next.

## What we grade
- **Accuracy:** Improvements in at least **two** metrics among WER/CER, EmailAcc, NumberAcc, NameF1, PunctuationF1.
- **Latency:** p95 ≤ 30 ms on `measure_latency.py` (you can reduce max length, cap candidates, optimize masking).
- **ML rigor:** Clear rationale and (tiny) ablation.
- **Code quality:** Clean diffs, no brittle hacks, reasonable edge-case handling.
- **Communication:** Your Loom is crisp and concrete.

Good luck!
