# src/ranker_onnx.py
from typing import List, Tuple, Optional
import numpy as np
import math
import random
import time

# optional imports
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None

# Small helpers (import from src.rules when available for tighter integration)
import re
_EMAIL_LIKE_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', flags=re.IGNORECASE)
_DIGITS_ONLY_RE = re.compile(r'^\s*\d+\s*$')

def looks_like_email(s: str) -> bool:
    return bool(_EMAIL_LIKE_RE.search(s))

def looks_like_number(s: str) -> bool:
    t = re.sub(r'[^\d]', '', s)
    return bool(t) and len(t) >= 1

class PseudoLikelihoodRanker:
    """
    Fast pseudo-likelihood ranker using ONNX runtime (preferred) or torch fallback.
    Improvements:
      - short-circuit for obvious emails/numbers
      - candidate cap
      - mask sampling + batching (limit masked positions)
      - ONNX session options tuned for low per-call overhead
    """
    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 onnx_path: Optional[str] = None,
                 device: str = "cpu",
                 max_length: int = 64,
                 max_candidates: int = 6,
                 max_masked_positions: int = 32,
                 seed: int = 1234):
        self.max_length = max_length
        self.model_name = model_name
        self.onnx = None
        self.torch_model = None
        self.device = device
        self.tokenizer = None

        # performance knobs
        self.MAX_CANDIDATES = max_candidates
        self.MAX_MASKED_POSITIONS = max_masked_positions  # sample at most this many positions to score
        self.SEED = seed

        # init
        if onnx_path and ort is not None:
            self._init_onnx(onnx_path)
        elif AutoTokenizer is not None and AutoModelForMaskedLM is not None:
            self._init_torch()
        else:
            raise RuntimeError("Neither onnxruntime nor transformers/torch are available. Install requirements or pass onnx_path.")

    def _init_onnx(self, onnx_path: str):
        # tokenizer still from transformers - this is lightweight
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # enable mem pattern / IPC? keep defaults for portability
        providers = ['CPUExecutionProvider']
        self.onnx = ort.InferenceSession(onnx_path, sess_opts, providers=providers)

    def _init_torch(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.torch_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.torch_model.eval()
        if self.device != "cpu":
            try:
                self.torch_model.to(self.device)
            except Exception:
                pass

    # ---- internal scoring helpers ----
    def _tokenize_np(self, text: str):
        """
        Tokenize and return numpy arrays for ONNX runtime.
        """
        toks = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        return toks["input_ids"].astype(np.int64), toks["attention_mask"].astype(np.int64)

    def _sample_mask_positions(self, attn_mask: np.ndarray) -> List[int]:
        """
        Return sampled token positions to mask (skip special tokens).
        attn_mask: shape (1, L) numpy
        """
        L = int(attn_mask[0].sum())
        if L <= 3:
            return []
        positions = list(range(1, L - 1))  # skip first and last special tokens
        if len(positions) <= self.MAX_MASKED_POSITIONS:
            return positions
        # deterministic sampling for reproducibility
        rnd = random.Random(self.SEED)
        chosen = rnd.sample(positions, self.MAX_MASKED_POSITIONS)
        chosen.sort()
        return chosen

    def _compute_logprob_batch_onnx(self, input_ids: np.ndarray, attn: np.ndarray, mask_positions: List[int]) -> float:
        """
        Vectorized ONNX scoring:
          - input_ids: shape (1, L)
          - mask_positions: list of positions to mask; we'll build a batch of size B=len(mask_positions)
          - returns sum(logprob(original_token)) over masked positions
        """
        if not mask_positions:
            return 0.0
        mask_id = int(self.tokenizer.mask_token_id)
        seq = input_ids[0]  # (L,)
        B = len(mask_positions)
        L = seq.shape[0]
        # build batch (B, L) where each row masks one position
        batch = np.repeat(seq[None, :], B, axis=0).astype(np.int64)
        for i, pos in enumerate(mask_positions):
            batch[i, pos] = mask_id
        batch_attn = np.repeat(attn, B, axis=0).astype(np.int64)
        ort_inputs = {
            "input_ids": batch,
            "attention_mask": batch_attn
        }
        # run once, returns logits [B, L, V]
        logits = self.onnx.run(None, ort_inputs)[0]  # numpy array
        # gather logits at masked positions and compute log-softmax
        rows = np.arange(B)
        cols = np.array(mask_positions, dtype=np.int64)
        # shape [B, V]
        logits_pos = logits[rows, cols, :]
        # numeric stable log-softmax per row
        m = logits_pos.max(axis=1, keepdims=True)
        log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum(axis=1, keepdims=True))
        # original token ids at those positions
        orig_tokens = seq[cols]
        # pick logprob for original token per row
        picked = log_probs[np.arange(B), orig_tokens]
        total = float(picked.sum())
        return total

    def _score_with_onnx(self, text: str) -> float:
        input_ids, attn = self._tokenize_np(text)
        mask_positions = self._sample_mask_positions(attn)
        # if no mask positions -> short, return 0
        if not mask_positions:
            return 0.0
        # batch compute
        return self._compute_logprob_batch_onnx(input_ids, attn, mask_positions)

    def _score_with_torch(self, text: str) -> float:
        import torch
        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]
        L = int(attn[0].sum())
        if L <= 2:
            return 0.0
        positions = list(range(1, L - 1))
        # sample positions similar to ONNX path
        if len(positions) > self.MAX_MASKED_POSITIONS:
            rnd = random.Random(self.SEED)
            positions = sorted(rnd.sample(positions, self.MAX_MASKED_POSITIONS))
        B = len(positions)
        seq = input_ids[0]
        batch = seq.unsqueeze(0).repeat(B, 1)
        for i, pos in enumerate(positions):
            batch[i, pos] = self.tokenizer.mask_token_id
        batch_attn = attn.repeat(B, 1)
        with torch.no_grad():
            out = self.torch_model(input_ids=batch, attention_mask=batch_attn).logits  # [B, L, V]
            rows = torch.arange(B, device=out.device)
            cols = torch.tensor(positions, device=out.device)
            logits_pos = out[rows, cols, :]
            log_probs = logits_pos.log_softmax(dim=-1)
            orig = seq.unsqueeze(0).repeat(B, 1)
            token_ids = orig[rows, cols]
            picked = log_probs[torch.arange(B), token_ids]
            return float(picked.sum().item())

    # ---- public API ----
    def score(self, sentences: List[str]) -> List[float]:
        """
        Return list of scores for the list of sentences (candidates).
        Uses ONNX if initialized, otherwise torch.
        """
        if self.onnx is not None:
            return [self._score_with_onnx(s) for s in sentences]
        else:
            return [self._score_with_torch(s) for s in sentences]

    def choose_best(self, candidates: List[str], original: Optional[str] = None) -> str:
        """
        Main entrypoint: pick best candidate.
        - caps candidates to MAX_CANDIDATES
        - short-circuits for emails/numbers
        - caches tokenization per candidate
        """
        if not candidates:
            return original or ""

        # cap candidates
        cands = candidates[:self.MAX_CANDIDATES]

        # cheap short-circuit: if any candidate is obvious email/number -> pick first such
        for c in cands:
            if looks_like_email(c) or looks_like_number(c):
                return c

        # if only one candidate left
        if len(cands) == 1:
            return cands[0]

        # Score all candidates; keep a small cache of tokenized forms for speed (if needed later)
        scores = []
        for c in cands:
            try:
                s = self._score_with_onnx(c) if self.onnx is not None else self._score_with_torch(c)
            except Exception:
                # fallback: extremely robust; give a very low score if model fails
                s = -1e9
            scores.append(s)

        best_idx = int(np.argmax(scores))
        return cands[best_idx]
