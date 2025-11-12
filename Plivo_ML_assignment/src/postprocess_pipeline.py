import json, time
from typing import Dict, List
from .rules import generate_candidates
from .ranker_onnx import PseudoLikelihoodRanker

class PostProcessor:
    def __init__(self, names_lex_path: str, onnx_model_path: str = None, device: str = "cpu", max_length: int = 64):
        self.names_lex = [x.strip() for x in open(names_lex_path, 'r', encoding='utf-8').read().splitlines() if x.strip()]
        self.ranker = PseudoLikelihoodRanker(onnx_path=onnx_model_path, device=device, max_length=max_length)

    def process_one(self, text: str) -> str:
        cands = generate_candidates(text, self.names_lex)
        best = self.ranker.choose_best(cands)
        # Simple punctuation heuristic: ensure trailing period for statements; add '?' if leading "can/shall/will/could"
        lower = best.lower().strip()
        if lower.endswith(('?', '.', ',')) is False:
            if lower.split()[0] in ('can','shall','will','could','would','is','are','do','does','did','should','hey','hello'):
                best = best.rstrip() + '?'
            else:
                best = best.rstrip() + '.'
        return best

def run_file(input_path: str, output_path: str, names_lex_path: str, onnx_model_path: str = None, device: str = "cpu", max_length: int = 64):
    pp = PostProcessor(names_lex_path, onnx_model_path=onnx_model_path, device=device, max_length=max_length)
    rows = [json.loads(line) for line in open(input_path, 'r', encoding='utf-8')]
    out = []
    for r in rows:
        pred = pp.process_one(r["text"])
        out.append({"id": r["id"], "text": pred})
    with open(output_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
