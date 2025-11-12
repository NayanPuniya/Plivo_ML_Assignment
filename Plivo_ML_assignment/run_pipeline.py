import argparse, os
from src.postprocess_pipeline import run_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/noisy_transcripts.jsonl")
    ap.add_argument("--output", default="out/corrected.jsonl")
    ap.add_argument("--names", default="data/names_lexicon.txt")
    ap.add_argument("--onnx", default="models/distilbert-base-uncased.int8.onnx")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_file(args.input, args.output, args.names, onnx_model_path=args.onnx, device=args.device)

if __name__ == "__main__":
    main()
