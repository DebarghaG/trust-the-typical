"""
Filename: duoguard_text_postprocessor.py
DuoGuard-0.5B Postprocessor for Text Safety Classification

This implementation uses DuoGuard-0.5B model for content safety classification.
DuoGuard-0.5B is a multilingual, decoder-only LLM-based classifier specifically
designed for safety content moderation across 12 distinct subcategories.
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class DuoGuardTextPostprocessor:
    """
    DuoGuard-0.5B Postprocessor for text safety classification.

    Uses DuoGuard-0.5B model to classify content as safe or unsafe
    based on 12 distinct safety subcategories.
    """

    def __init__(
        self,
        model_id: str = "DuoGuard/DuoGuard-0.5B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_duoguard",
        threshold: float = 0.5,
        max_length: int = 512,
    ):
        """
        Initialize DuoGuard Text Postprocessor.

        Args:
            model_id: HuggingFace model ID for DuoGuard.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store cache files.
            threshold: Threshold for binary safe/unsafe classification.
            max_length: Maximum input sequence length.
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.threshold = threshold
        self.max_length = max_length

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
        self.model: AutoModelForSequenceClassification | None = None
        self.tokenizer: AutoTokenizer | None = None

        # DuoGuard category names
        self.category_names = [
            "Violent crimes",
            "Non-violent crimes",
            "Sex-related crimes",
            "Child sexual exploitation",
            "Specialized advice",
            "Privacy",
            "Intellectual property",
            "Indiscriminate weapons",
            "Hate",
            "Suicide and self-harm",
            "Sexual content",
            "Jailbreak prompts",
        ]

        self.setup_flag = False
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _init_model(self) -> None:
        """Initialize the DuoGuard model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            print(f"Initializing DuoGuard model {self.model_id} on {self.device}...")

            # Set HuggingFace to use cached models to avoid rate limits
            os.environ["HF_HUB_OFFLINE"] = "1"  # Use offline mode if model is cached
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

            try:
                # Initialize tokenizer from Qwen 2.5 base model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-0.5B", cache_dir=cache_dir, local_files_only=True
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token

                # Initialize DuoGuard model with appropriate dtype for device
                # Use float16 instead of bfloat16 to avoid numpy conversion issues
                model_kwargs = {}
                if self.device.startswith("cuda"):
                    model_kwargs["torch_dtype"] = torch.float16
                    print("Using float16 precision for better numpy compatibility")

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id, cache_dir=cache_dir, local_files_only=True, **model_kwargs
                ).to(self.device)

                print(f"DuoGuard model loaded successfully on {self.device}")

            except Exception as e:
                print(f"Model initialization failed: {e}")
                raise

    def _classify_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Classify a batch of texts using DuoGuard.

        Args:
            texts: List of input texts to classify.

        Returns:
            Tuple of (probabilities, max_probabilities).
            probabilities: Shape (batch_size, 12) - probability for each category
            max_probabilities: Shape (batch_size,) - max probability across categories
        """
        if self.model is None or self.tokenizer is None:
            self._init_model()

        assert self.model is not None
        assert self.tokenizer is not None

        # Tokenize the batch
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            # DuoGuard outputs a 12-dimensional vector (one probability per subcategory)
            logits = outputs.logits  # shape: (batch_size, 12)
            probabilities = torch.sigmoid(logits)  # element-wise sigmoid

        # Convert to numpy - handle dtype compatibility
        if probabilities.dtype in [torch.bfloat16, torch.float16]:
            # Convert to float32 for better numpy compatibility
            prob_array = probabilities.float().cpu().numpy()  # shape: (batch_size, 12)
        else:
            prob_array = probabilities.cpu().numpy()  # shape: (batch_size, 12)

        # Get maximum probability across all categories for each sample
        max_probs = np.max(prob_array, axis=1)  # shape: (batch_size,)

        return prob_array, max_probs

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the DuoGuard postprocessor.

        Args:
            id_texts: In-distribution training texts (not used for DuoGuard).
            random_state: Random seed (for compatibility).
        """
        if self.setup_flag:
            print("DuoGuard postprocessor already set up.")
            return

        print("\nInitializing DuoGuard model...")
        self._init_model()

        print("DuoGuard postprocessor setup completed.")
        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get safety scores using DuoGuard.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            Safety confidence scores (higher = more safe/in-distribution).
            For DuoGuard, we return 1 - max_probability (safe when low risk).
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

        print(f"Classifying {len(texts)} texts with DuoGuard...")

        all_scores = []

        # Process texts in batches for memory efficiency
        for i in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[i : i + self.batch_size])

            try:
                _, max_probs = self._classify_batch(batch_texts)

                # Convert to safety scores (1 - risk_probability)
                # Higher scores mean safer (more in-distribution)
                batch_scores = 1.0 - max_probs
                all_scores.extend(batch_scores.tolist())

            except Exception as e:
                print(f"Error processing batch {i//self.batch_size}: {e}")
                # Default to unsafe (low confidence) on error
                batch_scores = [0.1] * len(batch_texts)
                all_scores.extend(batch_scores)

            # Clear GPU cache periodically
            if self.device.startswith("cuda") and (i + self.batch_size) % 64 == 0:
                torch.cuda.empty_cache()

        scores_array = np.array(all_scores)

        # Cache the results
        np.save(cache_path, scores_array)
        print(f"Cached {len(scores_array)} scores to {cache_path}")

        return scores_array

    def get_detailed_predictions(self, texts: Sequence[str]) -> dict:
        """
        Get detailed multi-label predictions for each category.

        Args:
            texts: Input texts.

        Returns:
            Dictionary with detailed predictions and probabilities.
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before get_detailed_predictions()")

        print(f"Getting detailed predictions for {len(texts)} texts...")

        all_detailed_results = []

        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[i : i + self.batch_size])

            try:
                prob_array, max_probs = self._classify_batch(batch_texts)

                # Process each text in the batch
                for j, text in enumerate(batch_texts):
                    text_probs = prob_array[j]  # probabilities for this text
                    max_prob = max_probs[j]

                    # Multi-label predictions (one for each category)
                    predicted_labels = (text_probs > self.threshold).astype(int)

                    # Overall binary classification
                    overall_unsafe = max_prob > self.threshold

                    # Category details
                    category_details = {}
                    for cat_idx, (cat_name, prob, label) in enumerate(
                        zip(self.category_names, text_probs, predicted_labels, strict=False)
                    ):
                        category_details[cat_name] = {
                            "probability": float(prob),
                            "predicted": bool(label),
                            "index": cat_idx,
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

            except Exception as e:
                print(f"Error processing batch {i//self.batch_size}: {e}")
                # Add error results for this batch
                for text in batch_texts:
                    error_result = {
                        "text": text,
                        "overall_unsafe": True,
                        "max_probability": 1.0,
                        "safety_score": 0.0,
                        "categories": {
                            cat: {"probability": 0.0, "predicted": False, "index": i}
                            for i, cat in enumerate(self.category_names)
                        },
                        "threshold": self.threshold,
                        "error": str(e),
                    }
                    all_detailed_results.append(error_result)

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
            Probability scores (safety scores between 0 and 1).
        """
        return self.postprocess(texts)

    def evaluate(self, id_texts: Sequence[str], ood_texts: Sequence[str]) -> dict[str, float]:
        """
        Evaluate the DuoGuard postprocessor.

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
def create_duoguard_detector(
    model_id: str = "DuoGuard/DuoGuard-0.5B", **kwargs: typing.Any
) -> DuoGuardTextPostprocessor:
    """
    Create DuoGuard text detector.

    Args:
        model_id: DuoGuard model ID to use.
        **kwargs: Additional arguments for DuoGuardTextPostprocessor.

    Returns:
        Configured DuoGuard postprocessor.
    """
    return DuoGuardTextPostprocessor(model_id=model_id, **kwargs)
