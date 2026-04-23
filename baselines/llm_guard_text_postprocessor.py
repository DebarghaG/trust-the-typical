"""
Filename: llm_guard_text_postprocessor.py
Llm Guard Postprocessor for Text Safety Classification
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

import numpy as np

# --- LLM-Guard imports ---
from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

from .evaluation_utils import evaluate_binary_classifier


# ---------- shared aggregation ----------
def _aggregate(
    results_valid: dict, results_score: dict, mode: str = "max", weights: dict | None = None
) -> tuple[str, float, float]:
    """
    Combine per-scanner results to a single (label, confidence, risk) triple.
    - label: "unsafe" iff any scanner invalidates.
    - confidence: 1 - aggregated_risk  (higher = safer).
    """
    is_safe = all(results_valid.values()) if results_valid else True
    if not results_score:
        # No scores returned (unlikely) → default very safe if valid, else low
        return ("safe" if is_safe else "unsafe", 1.0 if is_safe else 0.1, 0.0)

    risks = list(results_score.values())
    if mode == "max":
        risk = max(risks)
    elif mode == "mean":
        risk = float(sum(risks)) / len(risks)
    else:
        raise ValueError("mode must be 'max' | 'mean' | 'weighted'")

    confidence = max(0.0, min(1.0, 1.0 - float(risk)))
    return ("safe" if is_safe else "unsafe", confidence, float(risk))


# ---------- base class (mirrors your llama class shape) ----------
class _LLMGuardInputPostprocessor:
    """
    Generic LLM-Guard postprocessor for *input/prompt* scanning.
    Keeps the same API as your LlamaGuardTextPostprocessor.
    """

    def __init__(
        self,
        scanners: list,
        batch_size: int = 8,
        embedding_dir: str = "./embeddings_llm_guard",
        fail_fast: bool = False,
        agg_mode: str = "max",
        agg_weights: dict | None = None,
    ):
        """
        Args:
          scanners: list of instantiated LLM-Guard input scanners.
          fail_fast: stop at first violation (faster, less telemetry).
          agg_mode: 'max' | 'mean' | 'weighted' for risk aggregation.
          agg_weights: dict of weights if agg_mode='weighted'.
        """
        self.scanners = scanners
        self.batch_size = batch_size
        self.embedding_dir = embedding_dir
        self.fail_fast = fail_fast
        self.agg_mode = agg_mode
        self.agg_weights = agg_weights

        self.setup_flag = False
        os.makedirs(self.embedding_dir, exist_ok=True)

    # --- compatibility with your class ---
    def setup(self, id_texts: Sequence[str] | None = None, random_state: int = 42) -> None:
        if self.setup_flag:
            print("LLM-Guard postprocessor already set up.")
            return
        print("Initializing LLM-Guard input scanners...")
        _ = [type(s).__name__ for s in self.scanners]  # touch instances
        self.setup_flag = True

    def _classify_text(self, text: str) -> tuple[str, float]:
        # returns (label, confidence) just like your llama _classify_text
        _, valid, score = scan_prompt(self.scanners, str(text), fail_fast=self.fail_fast)
        label, confidence, _ = _aggregate(
            valid, score, mode=self.agg_mode, weights=self.agg_weights
        )
        return label, confidence

    def _hash_key(self, items: Sequence[str], salt: str) -> str:
        h = hashlib.sha1()
        for s in items:
            h.update(str(s).encode("utf-8"))
            h.update(b"\n")
        h.update(salt.encode("utf-8"))
        # also include scanner names + agg settings for cache safety
        h.update(("|".join(type(s).__name__ for s in self.scanners)).encode("utf-8"))
        h.update(
            f"|{self.fail_fast}|{self.agg_mode}|{sorted(self.agg_weights.items()) if self.agg_weights else 'None'}".encode()
        )
        return h.hexdigest()[:12]

    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Scores in [0,1] where higher = safer (same as your llama class).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")
        key = self._hash_key(texts, cache_name)
        cache_path = os.path.join(self.embedding_dir, f"{cache_name}_{key}.npy")

        if os.path.exists(cache_path):
            return np.load(cache_path)

        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for t in batch:
                try:
                    _, conf = self._classify_text(t)
                    scores.append(conf)
                except Exception as e:
                    print(f"[LLM-Guard] error on sample: {e}")
                    scores.append(0.1)  # conservative fallback

        arr = np.asarray(scores, dtype=np.float32)
        np.save(cache_path, arr)
        return arr

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        scores = self.postprocess(texts)
        return np.where(scores > 0.5, 1, -1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.postprocess(texts)

    def evaluate(self, id_texts: Sequence[str], ood_texts: Sequence[str]) -> dict[str, float]:
        """
        Same metrics block you use (AUROC, FPR@95, AUPRC, best F1)
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before evaluate()")

        id_scores = self.postprocess(id_texts, cache_name="eval_safe_llmg_in")
        ood_scores = self.postprocess(ood_texts, cache_name="eval_unsafe_llmg_in")

        return evaluate_binary_classifier(id_scores, ood_scores)


# ---------- factory: Jailbreak / Prompt-Injection detector ----------
def create_llm_guard_jailbreak_detector(
    batch_size: int = 8,
    embedding_dir: str = "./embeddings_llm_guard",
    fail_fast: bool = False,
    prompt_injection_threshold: float = 0.5,
    token_limit: int = 2048,
    agg_mode: str = "max",
    agg_weights: dict | None = None,
) -> _LLMGuardInputPostprocessor:
    scanners = [
        TokenLimit(limit=token_limit, encoding_name="cl100k_base"),
        PromptInjection(threshold=prompt_injection_threshold),
    ]
    return _LLMGuardInputPostprocessor(
        scanners=scanners,
        batch_size=batch_size,
        embedding_dir=embedding_dir,
        fail_fast=fail_fast,
        agg_mode=agg_mode,
        agg_weights=agg_weights,
    )


# ---------- factory: Toxicity detector ----------
def create_llm_guard_toxicity_detector(
    batch_size: int = 8,
    embedding_dir: str = "./embeddings_llm_guard",
    fail_fast: bool = False,
    toxicity_threshold: float = 0.5,
    lang: str = "en",  # kept for signature compatibility with your CLI
    token_limit: int = 2048,
    agg_mode: str = "max",
    agg_weights: dict | None = None,
) -> _LLMGuardInputPostprocessor:
    scanners = [
        TokenLimit(limit=token_limit, encoding_name="cl100k_base"),
        Toxicity(threshold=toxicity_threshold),
    ]
    return _LLMGuardInputPostprocessor(
        scanners=scanners,
        batch_size=batch_size,
        embedding_dir=embedding_dir,
        fail_fast=fail_fast,
        agg_mode=agg_mode,
        agg_weights=agg_weights,
    )
