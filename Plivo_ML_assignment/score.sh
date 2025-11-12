#!/usr/bin/env bash
set -e

# 1) Export and quantize DistilBERT to ONNX (first run only)
python -m src.export_onnx --model distilbert-base-uncased --max_length 64 --out models/distilbert-base-uncased.onnx --quant_out models/distilbert-base-uncased.int8.onnx

# 2) Run pipeline
python run_pipeline.py --onnx models/distilbert-base-uncased.int8.onnx

# 3) Evaluate metrics
python evaluate.py --pred out/corrected.jsonl --gold data/gold.jsonl --names data/names_lexicon.txt

# 4) Measure latency
python measure_latency.py --onnx models/distilbert-base-uncased.int8.onnx --runs 100 --warmup 10
