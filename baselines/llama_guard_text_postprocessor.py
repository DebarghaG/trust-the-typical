"""
Filename: llama_guard_text_postprocessor.py
Llama Guard 3-1B Postprocessor for Text Safety Classification

This implementation uses Meta's Llama Guard 3-1B model for content safety classification.
Llama Guard 3-1B is fine-tuned for detecting unsafe content across 13 hazard categories
based on the MLCommons taxonomy.
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class LlamaGuardTextPostprocessor:
    """
    Llama Guard 3-1B Postprocessor for text safety classification.

    Uses Meta's Llama Guard 3-1B model to classify content as safe or unsafe
    based on the MLCommons taxonomy of 13 hazard categories.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Guard-3-1B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_llama_guard",
        max_new_tokens: int = 20,
        temperature: float = 0.1,
    ):
        """
        Initialize Llama Guard Text Postprocessor.

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

    def _classify_text(self, text: str) -> tuple[str, float]:
        """
        Classify a single text using Llama Guard.

        Args:
            text: Input text to classify.

        Returns:
            Tuple of (classification, confidence_score).
            Classification is either "safe" or "unsafe".
            Confidence score is between 0 and 1 (higher = more confident).
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

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=0,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        generated_tokens = output[:, prompt_len:]
        response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Parse the response to determine safety
        response_lower = response.lower().strip()

        if response_lower.startswith("safe"):
            classification = "safe"
            # Safe content gets high confidence score
            confidence = 0.9
        elif response_lower.startswith("unsafe"):
            classification = "unsafe"
            # Unsafe content gets low confidence score (for OOD detection)
            confidence = 0.1

            # Extract specific violation categories if present
            if "s" in response_lower and any(c.isdigit() for c in response_lower):
                # More specific unsafe classification detected
                confidence = 0.05
        else:
            # Unknown response format, assume unsafe to be conservative
            classification = "unsafe"
            confidence = 0.2

        return classification, confidence

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the Llama Guard postprocessor.

        Args:
            id_texts: In-distribution training texts (not used for Llama Guard).
            random_state: Random seed (for compatibility).
        """
        if self.setup_flag:
            print("Llama Guard postprocessor already set up.")
            return

        print("\nInitializing Llama Guard model...")
        self._init_model()

        print("Llama Guard postprocessor setup completed.")
        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get safety scores using Llama Guard.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            Safety confidence scores (higher = more safe/in-distribution).
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

        print(f"Classifying {len(texts)} texts with Llama Guard...")

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
        Evaluate the Llama Guard postprocessor.

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
def create_llama_guard_detector(
    model_id: str = "meta-llama/Llama-Guard-3-1B", **kwargs: typing.Any
) -> LlamaGuardTextPostprocessor:
    """
    Create Llama Guard text detector.

    Args:
        model_id: Llama Guard model ID to use.
        **kwargs: Additional arguments for LlamaGuardTextPostprocessor.

    Returns:
        Configured Llama Guard postprocessor.
    """
    return LlamaGuardTextPostprocessor(model_id=model_id, **kwargs)
