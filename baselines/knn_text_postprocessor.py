"""
Filename: knn_text_postprocessor.py
KNN (k-Nearest Neighbors) OOD Detection for Text Embeddings

This module implements KNN-based OOD detection for text:
"Out-of-Distribution Detection with Deep Nearest Neighbors" (Sun et al., ICML 2022)

KNN approach:
1. Fits k-NN model on ID training embeddings
2. For test samples, finds k nearest neighbors in embedding space
3. Uses distance to nearest neighbors as OOD score
4. Lower distance = more in-distribution (higher score after negation)
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class KNNTextPostprocessor:
    """
    KNN Postprocessor for text embeddings.

    Uses k-nearest neighbors distance for OOD detection:
    - Fits k-NN model on ID training embeddings
    - Computes distance to nearest neighbors for test samples
    - Lower distance indicates more in-distribution
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_knn",
        n_neighbors: int = 1,  # Number of nearest neighbors (k)
        metric: str = "euclidean",  # Distance metric
    ):
        """
        Initialize KNN Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            n_neighbors: Number of nearest neighbors to consider (k).
            metric: Distance metric ('euclidean', 'cosine', 'manhattan', etc.).
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.metric = metric

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

        # KNN model
        self.knn_model: NearestNeighbors | None = None

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
            normalize_embeddings=True,
            device=self.device,
        )

        torch.save(embeddings, cache_path)
        print(f"Cached embeddings shape {embeddings.shape} to {cache_path}")

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return embeddings.to(self.device)

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the KNN postprocessor.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed (for reproducibility).
        """
        if self.setup_flag:
            print("KNN postprocessor already set up.")
            return

        print("\n Setup: Fitting k-NN model on ID training data...")

        # Extract ID features
        id_features = self._extract_embeddings(id_texts, name="id_train_knn")
        id_features_np = id_features.cpu().numpy()

        # Fit k-NN model
        print(f"Fitting k-NN with k={self.n_neighbors}, metric={self.metric}")
        self.knn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm="auto",
            n_jobs=-1,
        )
        self.knn_model.fit(id_features_np)

        print(f"k-NN model fitted on {len(id_features_np)} ID samples")
        print(f"Feature dimension: {id_features_np.shape[1]}")

        self.setup_flag = True
        print("KNN setup completed.")

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores using k-NN.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            k-NN scores (higher = more in-distribution).
            Note: We negate distances so higher score means more ID.
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        assert self.knn_model is not None

        features = self._extract_embeddings(texts, name=cache_name)
        features_np = features.cpu().numpy()

        distances, indices = self.knn_model.kneighbors(
            features_np, n_neighbors=self.n_neighbors, return_distance=True
        )

        if self.n_neighbors == 1:
            avg_distances = distances.ravel()
        else:
            avg_distances = np.mean(distances, axis=1)

        # Negate distances so that higher score = more ID
        scores = -avg_distances

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
        Evaluate the KNN postprocessor.

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
        self.n_neighbors = hyperparam[0]
        if self.knn_model is not None:
            print(f"Warning: Changing k to {self.n_neighbors} requires re-running setup()")

    def get_hyperparam(self) -> int:
        """Get hyperparameters (following reference API)."""
        return self.n_neighbors


def create_knn_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> KNNTextPostprocessor:
    """
    Create k-NN text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for KNNTextPostprocessor.

    Returns:
        Configured k-NN postprocessor.
    """
    return KNNTextPostprocessor(embedding_model=embedding_model, **kwargs)
