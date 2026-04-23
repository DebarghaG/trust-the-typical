"""
Filename: mahalanobis_text_postprocessor.py
Mahalanobis Distance OOD Detection for Text Embeddings

This module implements Mahalanobis distance-based OOD detection for text:
"A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks"
(Lee et al., NeurIPS 2018)

Mahalanobis approach:
1. Computes mean vector μ from ID training embeddings
2. Computes covariance matrix Σ from ID training embeddings
3. For test samples, computes Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
4. Lower distance = more in-distribution (higher score after negation)
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.covariance import EmpiricalCovariance

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class MahalanobisTextPostprocessor:
    """
    Mahalanobis Postprocessor for text embeddings.

    Uses Mahalanobis distance for OOD detection:
    - Computes mean and covariance from ID training embeddings
    - Calculates Mahalanobis distance for test samples
    - Lower distance indicates more in-distribution
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_mahalanobis",
    ):
        """
        Initialize Mahalanobis Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.embedding_dir = embedding_dir

        # Embedding model
        self.embedding_model: SentenceTransformer | None = None

        # Mahalanobis components
        self.mean: np.ndarray | None = None  # Mean vector
        self.precision: np.ndarray | None = None  # Precision matrix (inverse of covariance)

        self.setup_flag = False
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _init_embedding_model(self) -> None:
        """Initialize the text embedding model."""
        if self.embedding_model is None:
            print(f"Initializing embedding model {self.embedding_model_name} on {self.device}...")

            os.environ["HF_HUB_OFFLINE"] = "1"
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

            try:
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device,
                    trust_remote_code=True,
                    cache_folder=cache_dir,
                    model_kwargs=(
                        {"torch_dtype": torch.float16} if self.device.startswith("cuda") else {}
                    ),
                )
            except Exception as e:
                print(f"FP16 initialization failed: {e}, falling back to FP32")
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device,
                    trust_remote_code=True,
                    cache_folder=cache_dir,
                )

    def _extract_embeddings(self, texts: Sequence[str], name: str = "tmp") -> torch.Tensor:
        """Extract embeddings from texts using the embedding model."""
        if self.embedding_model is None:
            self._init_embedding_model()
        assert self.embedding_model is not None

        cache_path = os.path.join(self.embedding_dir, f"{name}_embeddings.pt")

        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            embeddings = torch.load(cache_path, map_location=self.device)
            if embeddings.size(0) == len(texts):
                return embeddings
            else:
                print("Cached embeddings size mismatch, recomputing...")

        print(f"Extracting embeddings for {len(texts)} texts...")

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        embeddings = self.embedding_model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=False,  # Use raw embeddings for Mahalanobis
            device=self.device,
        )

        torch.save(embeddings, cache_path)
        print(f"Cached embeddings shape {embeddings.shape} to {cache_path}")

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return embeddings.to(self.device)

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the Mahalanobis postprocessor.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed (for reproducibility).
        """
        if self.setup_flag:
            print("Mahalanobis postprocessor already set up.")
            return

        print("\n Setup: Computing mean and covariance from ID training data...")

        # Extract ID features
        id_features = self._extract_embeddings(id_texts, name="id_train_mahalanobis")
        id_features_np = id_features.cpu().numpy().astype(np.float32)

        # Compute mean
        self.mean = np.mean(id_features_np, axis=0)
        print(f"Mean vector shape: {self.mean.shape}")

        # Compute covariance and precision matrix
        # Center the data
        centered_features = id_features_np - self.mean

        # Use sklearn's EmpiricalCovariance for robust estimation
        cov_estimator = EmpiricalCovariance(assume_centered=True)
        cov_estimator.fit(centered_features)

        # Get precision matrix (inverse of covariance)
        self.precision = cov_estimator.precision_.astype(np.float32)
        print(f"Precision matrix shape: {self.precision.shape}")

        print(f"Fitted on {len(id_features_np)} ID samples")
        print(f"Feature dimension: {id_features_np.shape[1]}")

        self.setup_flag = True
        print("Mahalanobis setup completed.")

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores using Mahalanobis distance.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            Mahalanobis scores (higher = more in-distribution).
            Note: We negate distances so higher score means more ID.
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        assert self.mean is not None
        assert self.precision is not None

        features = self._extract_embeddings(texts, name=cache_name)
        features_np = features.cpu().numpy().astype(np.float32)

        # Center features
        centered = features_np - self.mean

        # Compute Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
        # This is equivalent to computing for each sample:
        # distance = centered @ precision @ centered.T (diagonal elements)
        mahal_dist = np.sum(centered @ self.precision * centered, axis=1)

        # Negate so that higher score = more ID (lower distance)
        scores = -mahal_dist

        return scores

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """
        Predict OOD status.

        Args:
            texts: Input texts.

        Returns:
            Binary predictions (1 for in-distribution, -1 for OOD).
        """
        scores = self.postprocess(texts)
        threshold = float(np.median(scores))
        return np.where(scores > threshold, 1, -1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return probability scores.

        Args:
            texts: Input texts.

        Returns:
            Normalized probability scores.
        """
        scores = self.postprocess(texts)

        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5

        return normalized_scores

    def evaluate(self, id_texts: Sequence[str], ood_texts: Sequence[str]) -> dict[str, float]:
        """
        Evaluate the Mahalanobis postprocessor.

        Args:
            id_texts: In-distribution texts.
            ood_texts: OOD texts.

        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before evaluate()")

        print(f"Evaluating on {len(id_texts)} ID and {len(ood_texts)} OOD texts...")

        id_scores = self.postprocess(id_texts, cache_name="eval_id")
        ood_scores = self.postprocess(ood_texts, cache_name="eval_ood")

        print_score_statistics(id_scores, ood_scores)
        return evaluate_binary_classifier(id_scores, ood_scores)

    def set_hyperparam(self, hyperparam: list) -> None:
        """Set hyperparameters (following reference API)."""
        # Mahalanobis has no hyperparameters to tune
        pass

    def get_hyperparam(self) -> list:
        """Get hyperparameters (following reference API)."""
        return []


def create_mahalanobis_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> MahalanobisTextPostprocessor:
    """
    Create Mahalanobis text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for MahalanobisTextPostprocessor.

    Returns:
        Configured Mahalanobis postprocessor.
    """
    return MahalanobisTextPostprocessor(embedding_model=embedding_model, **kwargs)
