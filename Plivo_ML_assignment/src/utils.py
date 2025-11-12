import re
import math

PUNCS = ['.', ',', '?']

def strip_punc(s: str) -> str:
    return re.sub(r'[\.,\?]+', '', s)

def punctuation_f1(pred: str, gold: str):
    def seq(x): return [c for c in x if c in PUNCS]
    p, g = seq(pred), seq(gold)
    tp = sum(min(p.count(c), g.count(c)) for c in PUNCS)
    fp = len(p) - tp
    fn = len(g) - tp
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return {'precision':prec,'recall':rec,'f1':f1}

def normalize_number_str(s: str) -> str:
    s = s.replace('₹','').replace(',','').strip().lower()
    return s

EMAIL_RE = re.compile(r'[\w\.\-+]+@[\w\.-]+\.[a-z]{2,}')

def extract_emails(s: str):
    return [x.lower() for x in EMAIL_RE.findall(s.lower())]

NUM_RE = re.compile(r'[₹$]?\s?[0-9][0-9,\.]*')

def extract_numbers(s: str):
    return [normalize_number_str(x) for x in NUM_RE.findall(s)]

def equal_lists(a, b):
    return a == b

def safe_div(a,b): 
    return a / (b + 1e-9)

def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))
