# src/rules.py
import re
from typing import List, Optional, Tuple
from rapidfuzz import process, fuzz

# ---------- knobs (tune these) ----------
MAX_CANDIDATES = 5
NAME_MATCH_THRESHOLD = 90     # rapidfuzz ratio (0-100)
MIN_NAME_TOKEN_LEN = 3        # ignore very short tokens for name matching
MIN_SPELLED_SEQ = 3           # collapse spelled-letter runs of length >= this
# ----------------------------------------

# Precompiled regexes (speed)
_RE_AT = re.compile(r'\b\(?(at|@)\)?\b', flags=re.IGNORECASE)
_RE_DOT = re.compile(r'\b(dot|period|point|full stop)\b', flags=re.IGNORECASE)
_RE_UNDERSCORE = re.compile(r'\b(underscore|under score)\b', flags=re.IGNORECASE)
_RE_DASH = re.compile(r'\b(dash|hyphen|minus)\b', flags=re.IGNORECASE)
_RE_SPACES_AROUND = re.compile(r'\s*([@._-])\s*')
_RE_RUPEES = re.compile(r'\brupees\s+', flags=re.IGNORECASE)
_RE_RUPEE_WITH_DIGITS = re.compile(r'₹\s*[0-9][0-9,\.]*')
_RE_DIGITS_ONLY = re.compile(r'^\d+$')
_RE_EMAIL_LIKE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', flags=re.IGNORECASE)

# Number word map
NUM_WORD = {
    'zero':'0','oh':'0','o':'0',
    'one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9'
}

# Pre-lowercased names lexicon cache helper
_cached_lexicons = {}

def prepare_names_lexicon(names_lex: List[str]) -> List[str]:
    """
    Lowercase and deduplicate lexicon once for faster matching.
    """
    key = id(names_lex)
    if key in _cached_lexicons:
        return _cached_lexicons[key]
    lower = list({n.strip() for n in names_lex if n and n.strip()})
    lower = [n for n in lower]
    _cached_lexicons[key] = lower
    return lower

# ---------- email utils ----------
def collapse_spelled_letters(s: str) -> str:
    """
    Collapse runs of single-letter tokens of length >= MIN_SPELLED_SEQ into a
    contiguous token. Example: 'g m a i l' -> 'gmail'
    """
    tokens = s.split()
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        # find run of single-letter tokens
        j = i
        while j < n and len(tokens[j]) == 1:
            j += 1
        run_len = j - i
        if run_len >= MIN_SPELLED_SEQ:
            out.append(''.join(tokens[i:j]))
            i = j
            continue
        # otherwise, keep current token
        out.append(tokens[i])
        i += 1
    return ' '.join(out)

def normalize_email_tokens(s: str) -> str:
    """
    Convert spoken email tokens to typical punctuation:
    'john dot doe at gmail dot com' -> 'john.doe@gmail.com'
    """
    s2 = collapse_spelled_letters(s)
    # replace spoken tokens
    s2 = _RE_AT.sub('@', s2)
    s2 = _RE_DOT.sub('.', s2)
    s2 = _RE_UNDERSCORE.sub('_', s2)
    s2 = _RE_DASH.sub('-', s2)
    # remove spaces around punctuation introduced
    s2 = _RE_SPACES_AROUND.sub(r'\1', s2)
    return s2

def looks_like_email(s: str) -> bool:
    return bool(_RE_EMAIL_LIKE.search(s))

# ---------- number utils ----------
def words_to_digits(seq: List[str]) -> str:
    """
    Convert a short sequence of word tokens (like ['double','nine','one']) into
    digits: '991' or '' if nothing convertible.
    """
    out = []
    i = 0
    while i < len(seq):
        tok = seq[i].lower()
        if tok in ('double','triple') and i + 1 < len(seq):
            times = 2 if tok == 'double' else 3
            nxt = seq[i+1].lower()
            if nxt in NUM_WORD:
                out.append(NUM_WORD[nxt] * times)
                i += 2
                continue
        if tok in NUM_WORD:
            out.append(NUM_WORD[tok])
            i += 1
            continue
        # not a numeric token: stop trying further for this window
        break
    return ''.join(out)

def normalize_numbers_spoken(s: str) -> str:
    """
    Scan through tokens and greedily convert short runs of spoken digits into numeric strings.
    This version only advances by the actual consumed tokens, avoiding skipping unrelated text.
    """
    tokens = s.split()
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        # greedy attempt: examine up to 8 tokens (short) to try detect numeric run
        window = tokens[i:i+8]
        wd = words_to_digits(window)
        if len(wd) >= 2:
            # determine how many tokens were consumed by words_to_digits
            # recompute consumption:
            consumed = 0
            # iterate window tokens and count how many token words contributed
            j = 0
            while j < len(window):
                tok = window[j].lower()
                if tok in ('double','triple'):
                    # requires next token to be NUM_WORD
                    if j+1 < len(window) and window[j+1].lower() in NUM_WORD:
                        consumed += 2
                        j += 2
                        continue
                    else:
                        break
                if tok in NUM_WORD:
                    consumed += 1
                    j += 1
                    continue
                break
            if consumed == 0:
                # safety fallback
                consumed = 1
            out.append(wd)
            i += consumed
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

def looks_like_number(s: str) -> bool:
    """
    fast check: true if token contains digits-only after stripping punctuation
    """
    t = re.sub(r'[^\d]', '', s)
    return bool(t) and len(t) >= 1

# ---------- currency utils ----------
def indian_grouping_str(num: str) -> str:
    """
    Group digits in Indian system: last 3 digits, then groups of 2.
    Works on a plain digit string.
    """
    n = num
    if len(n) <= 3:
        return n
    last3 = n[-3:]
    rest = n[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ','.join(parts + [last3])

def normalize_currency(s: str) -> str:
    """
    Replace 'rupees 12345' or '₹ 12345' with ₹12,345 or ₹1,23,456 etc.
    """
    # replace 'rupees' with ₹ first
    s = _RE_RUPEES.sub('₹', s)
    # replace any ₹ + digits-like sequences using a callback
    def repl(m):
        raw = re.sub(r'[^\d]', '', m.group(0))
        if not raw:
            return m.group(0)
        return '₹' + indian_grouping_str(raw)
    s = _RE_RUPEE_WITH_DIGITS.sub(repl, s)
    return s

# ---------- name correction ----------
def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = NAME_MATCH_THRESHOLD) -> str:
    """
    Fuzzy-correct tokens that look like names. Only attempt for tokens
    of MIN_NAME_TOKEN_LEN or more to avoid false matches on small words.
    Uses rapidfuzz (fast C-based).
    """
    if not names_lex:
        return s
    # prepare lexicon once per list
    prepared = prepare_names_lexicon(names_lex)
    tokens = s.split()
    out = []
    for t in tokens:
        if len(t) < MIN_NAME_TOKEN_LEN:
            out.append(t)
            continue
        best = process.extractOne(t, prepared, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            out.append(best[0])
        else:
            out.append(t)
    return ' '.join(out)

# ---------- punctuation helper ----------
def ensure_sentence_final_punct(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] in '.?!':
        return text
    return text + '.'

# ---------- candidate generation ----------
def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    """
    Produce a small set (<= MAX_CANDIDATES) of candidate corrections for the ranker.
    Order is short->long by length to keep deterministic and to favor compact corrections.
    """
    cands = []
    seen = set()

    # keep original
    def add_candidate(s: str):
        s = s.strip()
        if not s or s in seen:
            return
        seen.add(s)
        cands.append(s)

    text = text.strip()

    # Candidate A: full pipeline (emails, numbers, currency, names, punctuation)
    a = normalize_email_tokens(text)
    a = normalize_numbers_spoken(a)
    a = normalize_currency(a)
    a = correct_names_with_lexicon(a, names_lex)
    a = ensure_sentence_final_punct(a)
    add_candidate(a)

    # Candidate B: email-focused (useful if ASR garbles punctuation)
    b = normalize_email_tokens(text)
    b = correct_names_with_lexicon(b, names_lex)
    add_candidate(b)

    # Candidate C: numbers+currency focused (preserve original names)
    c = normalize_numbers_spoken(text)
    c = normalize_currency(c)
    add_candidate(c)

    # Candidate D: only names corrected
    d = correct_names_with_lexicon(text, names_lex)
    d = ensure_sentence_final_punct(d)
    add_candidate(d)

    # Candidate E: original (maybe punctuation only)
    e = ensure_sentence_final_punct(text)
    add_candidate(e)

    # deduplicate & cap
    out = sorted(cands, key=lambda x: len(x))[:MAX_CANDIDATES]
    return out

# small helpers exported for Stage2 short-circuiting
def extract_email_if_obvious(s: str) -> Optional[str]:
    """
    If we see a clear email after normalization, return it (fast).
    """
    norm = normalize_email_tokens(s)
    m = _RE_EMAIL_LIKE.search(norm)
    return m.group(0) if m else None
