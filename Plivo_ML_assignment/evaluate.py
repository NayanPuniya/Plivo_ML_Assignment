import argparse
from src.metrics import eval_corpus

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="out/corrected.jsonl")
    ap.add_argument("--gold", default="data/gold.jsonl")
    ap.add_argument("--names", default="data/names_lexicon.txt")
    args = ap.parse_args()
    m = eval_corpus(args.pred, args.gold, args.names)
    for k,v in m.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
