from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import numpy as np
from jiwer import wer, cer
from rapidfuzz import fuzz
from .utils import punctuation_f1, extract_emails, extract_numbers, normalize_number_str

@dataclass
class MetricResult:
    wer: float
    cer: float
    punct_f1: float
    email_acc: float
    number_acc: float
    name_f1: float

def _names_from_text(s: str, names_lex: List[str]) -> List[str]:
    s_tokens = [t.strip(".,?").lower() for t in s.split()]
    found = []
    for n in names_lex:
        ln = n.lower()
        # fuzzy token match for single words; direct in-string check for multi-word
        if " " in ln:
            if ln in s.lower():
                found.append(ln)
        else:
            # pick best token match >= 90
            for t in s_tokens:
                if fuzz.ratio(ln, t) >= 90:
                    found.append(ln)
                    break
    return sorted(list(set(found)))

def compute_entity_metrics(pred: str, gold: str, names_lex: List[str]) -> Dict[str, float]:
    # Emails exact list equality
    pe, ge = extract_emails(pred), extract_emails(gold)
    email_acc = 1.0 if pe == ge else 0.0

    # Numbers: compare normalized list equality
    pn, gn = extract_numbers(pred), extract_numbers(gold)
    pn = [normalize_number_str(x) for x in pn]
    gn = [normalize_number_str(x) for x in gn]
    number_acc = 1.0 if pn == gn else 0.0

    # Names: F1 over lexicon-based fuzzy matches
    pnms = set(_names_from_text(pred, names_lex))
    gnms = set(_names_from_text(gold, names_lex))
    tp = len(pnms & gnms)
    prec = tp / (len(pnms) + 1e-9)
    rec = tp / (len(gnms) + 1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9) if (prec+rec) > 0 else 0.0
    return {
        "email_acc": email_acc,
        "number_acc": number_acc,
        "name_f1": f1
    }

def eval_corpus(pred_path: str, gold_path: str, names_lex_path: str) -> Dict[str,float]:
    names_lex = [x.strip() for x in open(names_lex_path, 'r', encoding='utf-8').read().splitlines() if x.strip()]
    preds = [json.loads(line) for line in open(pred_path, 'r', encoding='utf-8')]
    golds = [json.loads(line) for line in open(gold_path, 'r', encoding='utf-8')]
    assert len(preds) == len(golds), "Pred and gold must have same length/order"
    WERs, CERs, PUNCS, EAs, NAs, NMF1 = [], [], [], [], [], []
    for p, g in zip(preds, golds):
        pt, gt = p["text"], g["text"]
        WERs.append(wer(gt, pt))
        CERs.append(cer(gt, pt))
        PUNCS.append(punctuation_f1(pt, gt)["f1"])
        ent = compute_entity_metrics(pt, gt, names_lex)
        EAs.append(ent["email_acc"]); NAs.append(ent["number_acc"]); NMF1.append(ent["name_f1"])
    return {
        "WER": float(np.mean(WERs)),
        "CER": float(np.mean(CERs)),
        "PunctuationF1": float(np.mean(PUNCS)),
        "EmailAcc": float(np.mean(EAs)),
        "NumberAcc": float(np.mean(NAs)),
        "NameF1": float(np.mean(NMF1))
    }
