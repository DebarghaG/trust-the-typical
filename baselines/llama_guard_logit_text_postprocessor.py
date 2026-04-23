"""
Filename: llama_guard_logit_text_postprocessor.py
Llama Guard 3-1B Logit-Based Postprocessor for Text Safety Classification

This implementation uses Meta's Llama Guard 3-1B model's actual output logits
for the "safe" token to provide better uncertainty quantification, rather than
using hardcoded confidence scores based on text parsing.
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class LlamaGuardLogitTextPostprocessor:
    """
    Llama Guard 3-1B Logit-Based Postprocessor for text safety classification.

    Uses Meta's Llama Guard 3-1B model to classify content as safe or unsafe
    based on the MLCommons taxonomy of 13 hazard categories. This version
    extracts the actual logits for the "safe" token to provide genuine
    model uncertainty rather than hardcoded confidence scores.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Guard-3-1B",
        batch_size: int = 32,
        device: str | None = None,
        embedding_dir: str = "./embeddings_llama_guard_logit",
        max_new_tokens: int = 20,
        temperature: float = 0.1,
    ):
        """
        Initialize Llama Guard Logit-Based Text Postprocessor.

        Args:
            model_id: HuggingFace model ID for Llama Guard.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store cache files.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Temperature for generation.
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.embedding_dir = embedding_dir

        # Model components
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None

        # Cache for token IDs
        self.safe_token_id: int | None = None
        self.unsafe_token_id: int | None = None

        self.setup_flag = False
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _init_model(self) -> None:
        """Initialize the Llama Guard model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            print(f"Initializing Llama Guard model {self.model_id} on {self.device}...")

            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

            # Model kwargs
            model_kwargs = {}
            if self.device.startswith("cuda"):
                model_kwargs["torch_dtype"] = torch.bfloat16

            # First try to load from local cache
            try:
                print("Attempting to load from local cache...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    **model_kwargs,
                )
                print("Successfully loaded from local cache")

            except Exception as cache_error:
                # If not in cache, try downloading with retries
                print(f"Cache load failed: {cache_error}")
                print("Attempting to download model (this may take a while)...")

                import time

                max_retries = 3
                retry_delay = 60  # seconds

                for attempt in range(max_retries):
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_id,
                            trust_remote_code=True,
                            cache_dir=cache_dir,
                            local_files_only=False,
                        )

                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            device_map="auto",
                            trust_remote_code=True,
                            cache_dir=cache_dir,
                            local_files_only=False,
                            **model_kwargs,
                        )
                        print("Successfully downloaded and loaded model")
                        break

                    except Exception as e:
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            if attempt < max_retries - 1:
                                print(
                                    f"Rate limit hit (429 error). Waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries}..."
                                )
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                print(f"Failed after {max_retries} attempts. Error: {e}")
                                raise
                        else:
                            print(f"Download failed with error: {e}")
                            raise

            assert self.tokenizer is not None
            # Note: Token IDs will be extracted from actual model outputs
            # to handle output layer pruning correctly (only ~20 tokens available)
            print("Llama Guard Logit postprocessor model loaded.")

    def _classify_text(self, text: str) -> tuple[str, float]:
        """
        Classify a single text using Llama Guard with logit-based scoring.

        Args:
            text: Input text to classify.

        Returns:
            Tuple of (classification, confidence_score).
            Classification is either "safe" or "unsafe".
            Confidence score is the probability of the "safe" token (0-1).
        """
        if self.model is None or self.tokenizer is None:
            self._init_model()

        assert self.model is not None
        assert self.tokenizer is not None

        # Format conversation for Llama Guard
        conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(
            self.model.device
        )

        prompt_len = input_ids.shape[1]

        # Generate response with scores to get logits
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=0,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode the generated text to determine classification
        generated_tokens = output.sequences[:, prompt_len:]
        response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Parse the response to determine safety
        response_lower = response.lower().strip()

        if response_lower.startswith("safe"):
            classification = "safe"
        elif response_lower.startswith("unsafe"):
            classification = "unsafe"
        else:
            # Unknown response format, assume unsafe to be conservative
            classification = "unsafe"

        # Find position of "safe" or "unsafe" token in the generated sequence
        # We need to find which position in output.scores contains the decision
        safe_token_pos = None
        unsafe_token_pos = None

        # Get token IDs for "safe" and "unsafe" by encoding them
        safe_tokens = self.tokenizer.encode("safe", add_special_tokens=False)
        unsafe_tokens = self.tokenizer.encode("unsafe", add_special_tokens=False)

        # Debug: Print first few generated tokens (only once)
        if not hasattr(self, "_debug_printed"):
            first_tokens = generated_tokens[0][: min(5, len(generated_tokens[0]))]
            token_strs = [self.tokenizer.decode([tid.item()]) for tid in first_tokens]
            print(f"Debug: First 5 generated tokens: {token_strs}")
            print(f"Debug: 'safe' encodes to token IDs: {safe_tokens}")
            print(f"Debug: 'unsafe' encodes to token IDs: {unsafe_tokens}")
            self._debug_printed = True

        # Search for "safe" or "unsafe" tokens in the generated sequence
        for pos, token_id in enumerate(generated_tokens[0]):
            tid = token_id.item()
            if tid in safe_tokens:
                safe_token_pos = pos
                break
            elif tid in unsafe_tokens:
                unsafe_token_pos = pos
                break

        # Determine which position to extract logits from
        decision_pos = safe_token_pos if safe_token_pos is not None else unsafe_token_pos

        if decision_pos is None:
            print(
                "Warning: Could not find 'safe' or 'unsafe' token in generated sequence. "
                "Using classification-based fallback score."
            )
            # Use classification to determine a reasonable fallback
            if classification == "safe":
                return classification, 0.8
            else:
                return classification, 0.2

        # Extract logits from the position containing the decision token
        # output.scores is a tuple of tensors, one for each generated position
        decision_logits = output.scores[decision_pos][0]  # shape: (vocab_size,)

        # Due to output layer pruning, only ~20 tokens have valid logits
        # Get the actually generated token at this position
        generated_token_id = generated_tokens[0][decision_pos].item()

        # Get logits for all possible "safe" and "unsafe" token IDs
        safe_logit_values = [
            decision_logits[tid].item() for tid in safe_tokens if tid < len(decision_logits)
        ]
        unsafe_logit_values = [
            decision_logits[tid].item() for tid in unsafe_tokens if tid < len(decision_logits)
        ]

        # Use max logit for each (in case multiple sub-tokens)
        safe_logit = max(safe_logit_values) if safe_logit_values else None
        unsafe_logit = max(unsafe_logit_values) if unsafe_logit_values else None

        # With output layer pruning, typically only the generated token has a valid logit
        # The other token will be -inf. We need to handle this case.
        safe_is_valid = safe_logit is not None and np.isfinite(safe_logit)
        unsafe_is_valid = unsafe_logit is not None and np.isfinite(unsafe_logit)

        # Case 1: Both logits are valid (ideal case, rare with pruning)
        if safe_is_valid and unsafe_is_valid:
            safe_prob = torch.softmax(
                torch.tensor([safe_logit, unsafe_logit], dtype=torch.float32), dim=0
            )[0].item()

            # Success message (only once)
            if not hasattr(self, "_success_printed"):
                print(
                    f"Success: Both 'safe' and 'unsafe' logits are valid (ideal case). "
                    f"Classification: {classification}, Confidence: {safe_prob:.4f}"
                )
                self._success_printed = True

            return classification, safe_prob

        # Case 2: Only one logit is valid (common with output layer pruning)
        # Use softmax over all valid logits at this position to get confidence
        if safe_is_valid or unsafe_is_valid:
            # Get all finite logits at this position
            valid_logits = []
            valid_token_ids = []

            for tid in range(len(decision_logits)):
                logit_val = decision_logits[tid].item()
                if np.isfinite(logit_val):
                    valid_logits.append(logit_val)
                    valid_token_ids.append(tid)

            if len(valid_logits) == 0:
                print(
                    "Warning: No valid logits found at decision position. "
                    "Using classification-based fallback score."
                )
                if classification == "safe":
                    return classification, 0.8
                else:
                    return classification, 0.2

            # Convert to tensors for softmax
            valid_logits_tensor = torch.tensor(valid_logits, dtype=torch.float32)
            probs = torch.softmax(valid_logits_tensor, dim=0)

            # Find the probability of the generated token
            try:
                generated_idx = valid_token_ids.index(generated_token_id)
                generated_prob = probs[generated_idx].item()
            except ValueError:
                print(
                    f"Warning: Generated token {generated_token_id} not in valid logits. "
                    f"Using classification-based fallback score."
                )
                if classification == "safe":
                    return classification, 0.8
                else:
                    return classification, 0.2

            # The confidence score depends on what was generated
            # If "safe" was generated, generated_prob is our safe confidence
            # If "unsafe" was generated, 1 - generated_prob would be safe confidence
            if classification == "safe":
                confidence = generated_prob
            else:  # unsafe
                confidence = 1.0 - generated_prob

            # Success message (only once)
            if not hasattr(self, "_success_printed"):
                print(
                    f"Success: Logit extraction working correctly. "
                    f"Found {len(valid_logits)} valid logits at decision position. "
                    f"Classification: {classification}, Confidence: {confidence:.4f}"
                )
                self._success_printed = True

            return classification, confidence

        # Case 3: Neither logit is valid (should not happen)
        print(
            f"Warning: Could not extract valid logits for 'safe' or 'unsafe'. "
            f"safe_logit={safe_logit}, unsafe_logit={unsafe_logit}. "
            f"Using classification-based fallback score."
        )
        if classification == "safe":
            return classification, 0.8
        else:
            return classification, 0.2

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the Llama Guard postprocessor.

        Args:
            id_texts: In-distribution training texts (not used for Llama Guard).
            random_state: Random seed (for compatibility).
        """
        if self.setup_flag:
            print("Llama Guard Logit postprocessor already set up.")
            return

        print("\nInitializing Llama Guard model with logit extraction...")
        self._init_model()

        print("Llama Guard Logit postprocessor setup completed.")
        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get safety scores using Llama Guard logits.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            Safety confidence scores (higher = more safe/in-distribution).
            Based on actual model logit probabilities for the "safe" token.
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        # Check for cached results
        cache_path = os.path.join(self.embedding_dir, f"{cache_name}_scores.npy")

        if os.path.exists(cache_path):
            print(f"Loading cached scores from {cache_path}")
            cached_scores = np.load(cache_path)
            if len(cached_scores) == len(texts):
                return cached_scores
            else:
                print("Cached scores size mismatch, recomputing...")

        print(f"Classifying {len(texts)} texts with Llama Guard (logit-based)...")

        scores = []

        # Process texts in batches for memory efficiency
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_scores = []

            for text in batch_texts:
                try:
                    classification, confidence = self._classify_text(text)
                    batch_scores.append(confidence)
                except Exception as e:
                    print(f"Error processing text: {e}")
                    # Default to unsafe (low confidence) on error
                    batch_scores.append(0.1)

            scores.extend(batch_scores)

            # Clear GPU cache periodically
            if self.device.startswith("cuda") and (i + self.batch_size) % 64 == 0:
                torch.cuda.empty_cache()

        scores_array = np.array(scores)

        # Validate and clean NaN values before caching
        nan_mask = ~np.isfinite(scores_array)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            print(
                f"Warning: Found {nan_count} NaN/inf values in scores. "
                "Replacing with default confidence (0.5)."
            )
            scores_array[nan_mask] = 0.5

        # Cache the results
        np.save(cache_path, scores_array)
        print(f"Cached {len(scores_array)} scores to {cache_path}")

        return scores_array

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """
        Predict safety status.

        Args:
            texts: Input texts.

        Returns:
            Binary predictions (1 for safe, -1 for unsafe).
        """
        scores = self.postprocess(texts)
        # Use 0.5 as threshold (scores > 0.5 are considered safe)
        return np.where(scores > 0.5, 1, -1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return probability scores.

        Args:
            texts: Input texts.

        Returns:
            Probability scores (already normalized between 0 and 1).
        """
        return self.postprocess(texts)

    def evaluate(self, id_texts: Sequence[str], ood_texts: Sequence[str]) -> dict[str, float]:
        """
        Evaluate the Llama Guard Logit postprocessor.

        Args:
            id_texts: In-distribution (safe) texts.
            ood_texts: OOD (unsafe) texts.

        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before evaluate()")

        print(f"Evaluating on {len(id_texts)} safe and {len(ood_texts)} unsafe texts...")

        # Get scores
        id_scores = self.postprocess(id_texts, cache_name="eval_safe")
        ood_scores = self.postprocess(ood_texts, cache_name="eval_unsafe")

        print_score_statistics(id_scores, ood_scores)
        return evaluate_binary_classifier(id_scores, ood_scores)


# Convenience function matching other postprocessor styles
def create_llama_guard_logit_detector(
    model_id: str = "meta-llama/Llama-Guard-3-1B", **kwargs: typing.Any
) -> LlamaGuardLogitTextPostprocessor:
    """
    Create Llama Guard logit-based text detector.

    Args:
        model_id: Llama Guard model ID to use.
        **kwargs: Additional arguments for LlamaGuardLogitTextPostprocessor.

    Returns:
        Configured Llama Guard Logit postprocessor.
    """
    return LlamaGuardLogitTextPostprocessor(model_id=model_id, **kwargs)
