import argparse, json, random, time
from src.postprocess_pipeline import PostProcessor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/noisy_transcripts.jsonl")
    ap.add_argument("--names", default="data/names_lexicon.txt")
    ap.add_argument("--onnx", default="models/distilbert-base-uncased.int8.onnx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input, 'r', encoding='utf-8')]
    texts = [r["text"] for r in rows][:50]
    pp = PostProcessor(args.names, onnx_model_path=args.onnx, device=args.device, max_length=64)

    # Warmup
    for _ in range(args.warmup):
        _ = pp.process_one(texts[0])

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        _ = pp.process_one(texts[i % len(texts)])
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

    times_sorted = sorted(times)
    p50 = times_sorted[int(0.5*len(times))-1]
    p95 = times_sorted[int(0.95*len(times))-1]

    print(f"p50_ms={p50:.2f} p95_ms={p95:.2f} (runs={args.runs})")

if __name__ == "__main__":
    main()
