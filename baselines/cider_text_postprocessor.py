"""
Filename: cider_text_postprocessor.py
CIDER Postprocessor for Text Embeddings (without FAISS)

This module implements CIDER for text OOD detection without FAISS dependency:
"CIDER: Exploiting Hyperdimensional Embeddings for Out-of-Distribution Detection"

CIDER approach:
1. Stores training embeddings as reference set
2. For test samples, finds K nearest neighbors using cosine distance
3. Uses distance to Kth nearest neighbor as OOD score
4. Implements efficient nearest neighbor search without FAISS
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class CIDERTextPostprocessor:
    """
    CIDER Postprocessor for text embeddings without FAISS.

    Uses K-nearest neighbor distance for OOD detection:
    - Stores training embeddings as reference
    - Computes distance to Kth nearest neighbor for test samples
    - Higher distance = more OOD
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_cider",
        K: int = 5,  # Number of nearest neighbors
    ):
        """
        Initialize CIDER Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            K: Number of nearest neighbors for distance computation.
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.K = K

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

        # CIDER components (following reference pattern)
        self.activation_log: np.ndarray | None = None  # Training embeddings

        self.setup_flag = False
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _init_embedding_model(self) -> None:
        """Initialize the text embedding model."""
        if self.embedding_model is None:
            print(f"Initializing embedding model {self.embedding_model_name} on {self.device}...")

            # Set HuggingFace to use cached models to avoid rate limits
            os.environ["HF_HUB_OFFLINE"] = "1"  # Use offline mode if model is cached
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

            # Use smaller precision and enable memory optimizations
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

        # Check for cached embeddings
        cache_path = os.path.join(self.embedding_dir, f"{name}_embeddings.pt")

        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            embeddings = torch.load(cache_path, map_location=self.device)
            if embeddings.size(0) == len(texts):
                return embeddings
            else:
                print("Cached embeddings size mismatch, recomputing...")

        print(f"Extracting embeddings for {len(texts)} texts...")

        # Clear GPU cache before encoding
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        embeddings = self.embedding_model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,  # CIDER works better with normalized embeddings
            device=self.device,
        )

        # Cache embeddings
        torch.save(embeddings, cache_path)
        print(f"Cached embeddings shape {embeddings.shape} to {cache_path}")

        # Clear cache again after encoding
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return embeddings.to(self.device)

    def _compute_distances(self, query_embeddings: np.ndarray, k: int | None = None) -> np.ndarray:
        """
        Compute distances to k nearest neighbors without FAISS.

        Args:
            query_embeddings: Query embeddings [n_queries, embedding_dim].
            k: Number of nearest neighbors (defaults to self.K).

        Returns:
            Distances to kth nearest neighbor [n_queries].
        """
        if k is None:
            k = self.K

        if self.activation_log is None:
            raise RuntimeError("Must call setup() before computing distances")

        print(f"Computing {k}-NN distances for {len(query_embeddings)} queries...")

        # Use cosine distance (1 - cosine similarity) for normalized embeddings
        # Cosine similarity = dot product for normalized vectors
        similarities = query_embeddings @ self.activation_log.T  # [n_queries, n_training]
        distances = 1 - similarities  # Convert to distances

        # Find k smallest distances (k nearest neighbors)
        # Use numpy's partition for efficiency (faster than full sort)
        kth_distances = np.partition(distances, k - 1, axis=1)[:, k - 1]  # k-1 because 0-indexed

        return kth_distances

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the CIDER postprocessor following the exact reference pattern.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed (for compatibility).
        """
        if self.setup_flag:
            print("CIDER postprocessor already set up.")
            return

        print("\n Setup: Extracting training features...")

        # Extract features from training texts (following reference pattern)
        activation_log = []

        # Extract all features from training texts
        features = self._extract_embeddings(id_texts, name="id_train")
        activation_log.append(features.cpu().numpy())

        # Concatenate all features (following reference pattern exactly)
        self.activation_log = np.concatenate(activation_log, axis=0).astype(np.float32)
        print(f"Training feature shape: {self.activation_log.shape}")

        # No need for FAISS index - we'll compute distances directly
        print(f"CIDER setup completed with {len(self.activation_log)} training samples")

        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores following CIDER pattern.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            CIDER scores (higher = more in-distribution, i.e., lower distance).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        # Extract features
        features = self._extract_embeddings(texts, name=cache_name)
        features_np = features.cpu().numpy().astype(np.float32)

        # Following CIDER reference pattern exactly:
        # Compute distance to Kth nearest neighbor
        kth_distances = self._compute_distances(features_np, k=self.K)

        # Return negative distance (so higher = more in-distribution)
        cider_scores = -kth_distances

        return cider_scores

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

        # Normalize to [0, 1]
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5

        return normalized_scores

    def evaluate(self, id_texts: Sequence[str], ood_texts: Sequence[str]) -> dict[str, float]:
        """
        Evaluate the CIDER postprocessor.

        Args:
            id_texts: In-distribution texts.
            ood_texts: OOD texts.

        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before evaluate()")

        print(f"Evaluating on {len(id_texts)} ID and {len(ood_texts)} OOD texts...")

        # Get scores
        id_scores = self.postprocess(id_texts, cache_name="eval_id")
        ood_scores = self.postprocess(ood_texts, cache_name="eval_ood")

        print_score_statistics(id_scores, ood_scores)
        return evaluate_binary_classifier(id_scores, ood_scores)

    def set_hyperparam(self, hyperparam: list) -> None:
        """Set hyperparameters (following reference API)."""
        self.K = hyperparam[0]

    def get_hyperparam(self) -> int:
        """Get hyperparameters (following reference API)."""
        return self.K


# Convenience function matching other postprocessors
def create_cider_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> CIDERTextPostprocessor:
    """
    Create CIDER text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for CIDERTextPostprocessor.

    Returns:
        Configured CIDER postprocessor.
    """
    return CIDERTextPostprocessor(embedding_model=embedding_model, **kwargs)
