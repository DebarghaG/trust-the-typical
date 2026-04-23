"""
Filename: openai_omni_moderation_postprocessor.py
OpenAI Omni Moderation Postprocessor with Robust Caching and Logging

This implementation uses the OpenAI Omni Moderation API for content safety classification.
It is a full-featured class with methods for scoring, detailed predictions, evaluation,
and a persistent raw-file cache for all API interactions.
"""

from __future__ import annotations

import os
import time
import typing
import random
from collections.abc import Sequence
from typing import Any
from collections import deque

import numpy as np
from openai import APIError, OpenAI
from tqdm import tqdm

from utils.api_logger_cache import APILoggerCache

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class OpenaiOmniModerationPostprocessor:
    """
    OpenAI Omni Moderation Postprocessor with a persistent, raw-file cache.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "omni-moderation-latest",
        batch_size: int = 32,
        embedding_dir: str = "./embeddings_openai_omni",
        threshold: float = 0.5,
        rpm_limit: int = 1000,
        tpm_limit: int = 50_000,
        token_estimator: str = "words",  # or "chars"
        verbose_rate_limit: bool = True,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.threshold = threshold
        self.embedding_dir = embedding_dir

        self.cache = APILoggerCache(cache_dir_name="openai_omni_cache")

        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

        self.category_names = [
            "sexual",
            "sexual/minors",
            "harassment",
            "harassment/threatening",
            "hate",
            "hate/threatening",
            "illicit",
            "illicit/violent",
            "self-harm",
            "self-harm/intent",
            "self-harm/instructions",
            "violence",
            "violence/graphic",
        ]
        self.setup_flag = False
        os.makedirs(self.embedding_dir, exist_ok=True)

        # --- Rate limiting (non-intrusive) ---
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.token_estimator = token_estimator
        self.verbose_rate_limit = verbose_rate_limit
        self._request_timestamps = deque()  
        self._token_records = deque()       

    # ---------------- Rate Limiting Helpers -----------------
    def _prune_windows(self, now: float) -> None:
        """Prune old entries outside the 60 second window."""
        while self._request_timestamps and now - self._request_timestamps[0] >= 60.0:
            self._request_timestamps.popleft()
        while self._token_records and now - self._token_records[0][0] >= 60.0:
            self._token_records.popleft()

    def _current_token_usage(self) -> int:
        return sum(t for _, t in self._token_records)

    def _estimate_tokens(self, texts: list[str]) -> int:
        """Lightweight token estimate.

        words: count words (1 word ~= 1 token approx; conservative)
        chars: 4 chars ~= 1 token.
        """
        if not texts:
            return 0
        if self.token_estimator == "chars":
            total_chars = sum(len(t) for t in texts)
            return max(1, total_chars // 4)
        total_words = sum(len(t.split()) for t in texts)
        return max(1, int(total_words))

    def _respect_rate_limits(self, tokens_needed: int) -> None:
        if (self.rpm_limit <= 0) and (self.tpm_limit <= 0):
            return

        now = time.time()
        self._prune_windows(now)

        if self.tpm_limit > 0 and tokens_needed > self.tpm_limit:
            if self.verbose_rate_limit:
                print(
                    f"[RateLimit Warning] Single batch estimated tokens ({tokens_needed}) exceeds TPM limit ({self.tpm_limit}). Consider reducing batch_size."
                )

        while True:
            now = time.time()
            self._prune_windows(now)

            req_count = len(self._request_timestamps)
            token_usage = self._current_token_usage()

            over_rpm = self.rpm_limit > 0 and req_count >= self.rpm_limit
            over_tpm = self.tpm_limit > 0 and (token_usage + tokens_needed) > self.tpm_limit

            if not over_rpm and not over_tpm:
                break

            sleep_rpm = (
                (60.0 - (now - self._request_timestamps[0])) if over_rpm and self._request_timestamps else 0.0
            )
            sleep_tpm = (
                (60.0 - (now - self._token_records[0][0])) if over_tpm and self._token_records else 0.0
            )
            sleep_for = max(sleep_rpm, sleep_tpm, 0.05)
            if self.verbose_rate_limit:
                reasons = []
                if over_rpm:
                    reasons.append(f"RPM {req_count}/{self.rpm_limit}")
                if over_tpm:
                    reasons.append(f"TPM {token_usage + tokens_needed}/{self.tpm_limit}")
                print(f"[RateLimit] Sleeping {sleep_for:.2f}s due to {' & '.join(reasons)}")
            time.sleep(sleep_for)

        now = time.time()
        self._request_timestamps.append(now)
        self._token_records.append((now, tokens_needed))

    # ---------------- Adaptive Batch Handling -----------------
    def _estimate_tokens_per_text(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        if self.token_estimator == "chars":
            return [max(1, len(t)//4) for t in texts]
        return [max(1, len(t.split())) for t in texts]

    def _split_batch_if_needed(self, texts: list[str]) -> list[list[str]]:
        """Split a batch if its estimated tokens are likely to breach TPM.

        Strategy: keep chunks under 80% of TPM (or full list if limits disabled).
        This reduces long sleeps and 429s when one batch is too large.
        """
        if self.tpm_limit <= 0 or not texts:
            return [texts]
        per_text_tokens = self._estimate_tokens_per_text(texts)
        total_tokens = sum(per_text_tokens)
        soft_cap = int(self.tpm_limit * 0.8)
        if total_tokens <= soft_cap:
            return [texts]
        # Greedy split
        batches: list[list[str]] = []
        current: list[str] = []
        tok_acc = 0
        for text, tks in zip(texts, per_text_tokens):
            if tok_acc + tks > soft_cap and current:
                batches.append(current)
                current = [text]
                tok_acc = tks
            else:
                current.append(text)
                tok_acc += tks
        if current:
            batches.append(current)
        if self.verbose_rate_limit and len(batches) > 1:
            print(f"[BatchSplit] Split large batch of {len(texts)} texts into {len(batches)} sub-batches to respect TPM.")
        return batches

    def _fetch_from_api_and_log(self, texts_to_fetch: list[str]) -> list[Any]:
        """
        Handles the actual API call and logs everything using the cache utility.
        """
        if not texts_to_fetch:
            return []

        results_map: dict[str, Any] = {}
        sub_batches = self._split_batch_if_needed(texts_to_fetch)

        for sub in sub_batches:
            pending = [t for t in sub if t not in results_map]
            if not pending:
                continue

            est_tokens = self._estimate_tokens(pending)
            self._respect_rate_limits(est_tokens)

            request_payload = {"input": pending, "model": self.model_id}
            attempt = 0
            max_attempts = 5
            backoff_base = 1.0
            while attempt < max_attempts:
                start_time = time.time()
                try:
                    response = self.client.moderations.create(**request_payload)
                    duration = time.time() - start_time
                    got = len(response.results)
                    if got != len(pending) and self.verbose_rate_limit:
                        print(
                            f"[Warning] API returned {got} results for {len(pending)} inputs (possible partial)."
                        )
                    for text, res in zip(pending, response.results):
                        single_input = {"input": text, "model": self.model_id}
                        single_output = {"results": [res.model_dump()]}
                        self.cache.log_and_save(
                            text,
                            self.model_id,
                            single_input,
                            single_output,
                            duration / max(1, len(pending)),
                            True,
                        )
                        results_map[text] = res
                    if len(results_map) - sum(t in results_map for t in pending) < len(pending):
                        missing = [t for t in pending if t not in results_map]
                        if missing and self.verbose_rate_limit:
                            print(f"[Partial] {len(missing)} inputs missing; will retry them.")
                        pending = missing
                        if not pending:
                            break
                        time.sleep(0.5)
                        request_payload = {"input": pending, "model": self.model_id}
                        attempt += 1
                        continue
                    break  # success
                except APIError as e:
                    is_429 = '429' in str(e) or 'Too Many Requests' in str(e)
                    attempt += 1
                    if is_429 and attempt < max_attempts:
                        sleep_time = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                        if self.verbose_rate_limit:
                            print(f"[429 Retry] Attempt {attempt}/{max_attempts} sleeping {sleep_time:.2f}s before retry.")
                        time.sleep(sleep_time)
                        continue
                    duration = time.time() - start_time
                    print(f"OpenAI API error processing batch: {e}. Logging failure.")
                    for text in pending:
                        if text not in results_map:
                            self.cache.log_and_save(
                                text,
                                self.model_id,
                                {"input": text, "model": self.model_id},
                                {"error": str(e)},
                                duration,
                                False,
                            )
                            results_map[text] = None
                    break

        for text in texts_to_fetch:
            if text not in results_map:
                results_map[text] = None

        return [results_map[t] for t in texts_to_fetch]

    def _classify_batch(self, texts: list[str]) -> tuple[list[dict], list[float]]:
        """
        Classifies a batch, using the cache first and then calling the API.
        """
        all_scores_map = {}
        max_probs_map = {}
        texts_to_fetch = []

        for text in texts:
            cached_output = self.cache.check_and_load(text, self.model_id)
            if cached_output:
                res = cached_output["results"][0]
                category_scores = res["category_scores"]
                all_scores_map[text] = category_scores
                max_probs_map[text] = max(category_scores.values()) if category_scores else 0.0
            else:
                texts_to_fetch.append(text)

        if texts_to_fetch:
            print(f"  (Cache miss: querying API for {len(texts_to_fetch)} of {len(texts)} texts)")
            api_results = self._fetch_from_api_and_log(texts_to_fetch)

            if api_results:
                if len(api_results) != len(texts_to_fetch):
                    print(
                        f"[Warning] Mismatch: received {len(api_results)} results for {len(texts_to_fetch)} inputs. Truncating to min length."
                    )
                for text, res in zip(texts_to_fetch, api_results):
                    try:
                        category_scores = res.category_scores.dict()
                    except Exception:
                        category_scores = getattr(res, 'category_scores', {})
                        if hasattr(category_scores, 'dict'):
                            category_scores = category_scores.dict()
                    all_scores_map[text] = category_scores
                    max_probs_map[text] = max(category_scores.values()) if category_scores else 0.0
            else:
                for text in texts_to_fetch:
                    all_scores_map[text] = {cat: 0.0 for cat in self.category_names}
                    max_probs_map[text] = 1.0

        final_all_scores = [all_scores_map[text] for text in texts]
        final_max_probs = [max_probs_map[text] for text in texts]

        return final_all_scores, final_max_probs

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        if self.setup_flag:
            print("OpenAI Omni Moderation processor already set up.")
            return
        print("\nInitializing OpenAI Omni Moderation processor...")
        try:
            self._fetch_from_api_and_log(["test connection"])
            print("OpenAI API connection and logging successful.")
        except Exception as e:
            raise RuntimeError(f"OpenAI API connection/logging failed: {e}") from e
        self.setup_flag = True
        print("OpenAI Omni Moderation processor setup completed.")

    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        cache_path = os.path.join(self.embedding_dir, f"{cache_name}_scores.npy")
        if os.path.exists(cache_path):
            cached_scores = np.load(cache_path)
            if len(cached_scores) == len(texts):
                print(f"Loading cached scores from {cache_path}")
                return cached_scores
            print("Cached scores size mismatch, recomputing...")

        print(f"Classifying {len(texts)} texts with OpenAI Omni Moderation API...")
        all_safety_scores = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing with OpenAI API"):
            batch_texts = list(texts[i : i + self.batch_size])
            _, max_probs = self._classify_batch(batch_texts)
            batch_safety_scores = (1.0 - np.array(max_probs)).tolist()
            all_safety_scores.extend(batch_safety_scores)

        scores_array = np.array(all_safety_scores)
        np.save(cache_path, scores_array)
        print(f"Cached {len(scores_array)} scores to {cache_path}")
        return scores_array

    def get_detailed_predictions(self, texts: Sequence[str]) -> dict:
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before get_detailed_predictions()")

        print(f"Getting detailed predictions for {len(texts)} texts...")
        all_detailed_results = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Getting detailed predictions"):
            batch_texts = list(texts[i : i + self.batch_size])
            all_category_scores, max_probs = self._classify_batch(batch_texts)

            for j, text in enumerate(batch_texts):
                text_scores = all_category_scores[j]
                max_prob = max_probs[j]
                overall_unsafe = max_prob > self.threshold

                category_details = {}
                for idx, cat_name in enumerate(self.category_names):
                    prob = text_scores.get(cat_name, 0.0)
                    category_details[cat_name] = {
                        "probability": float(prob),
                        "predicted": bool(prob > self.threshold),
                        "index": idx,
                    }

                result = {
                    "text": text,
                    "overall_unsafe": bool(overall_unsafe),
                    "max_probability": float(max_prob),
                    "safety_score": float(1.0 - max_prob),
                    "categories": category_details,
                    "threshold": self.threshold,
                }
                all_detailed_results.append(result)

        return {
            "results": all_detailed_results,
            "summary": {
                "total_texts": len(texts),
                "unsafe_count": sum(1 for r in all_detailed_results if r["overall_unsafe"]),
                "safe_count": sum(1 for r in all_detailed_results if not r["overall_unsafe"]),
                "threshold": self.threshold,
                "category_names": self.category_names,
            },
        }

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        scores = self.postprocess(texts)
        return np.where(scores > 0.5, 1, -1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.postprocess(texts)

    def evaluate(self, id_texts: Sequence[str], ood_texts: Sequence[str]) -> dict[str, float]:
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before evaluate()")

        print(f"Evaluating on {len(id_texts)} safe and {len(ood_texts)} unsafe texts...")
        id_scores = self.postprocess(id_texts, cache_name="eval_safe")
        ood_scores = self.postprocess(ood_texts, cache_name="eval_unsafe")

        print_score_statistics(id_scores, ood_scores)
        return evaluate_binary_classifier(id_scores, ood_scores)


def create_openai_omni_moderation_detector(
    api_key: str | None = None, **kwargs: typing.Any
) -> OpenaiOmniModerationPostprocessor:
    return OpenaiOmniModerationPostprocessor(api_key=api_key, **kwargs)
