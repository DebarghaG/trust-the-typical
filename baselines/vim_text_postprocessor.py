"""
Filename: vim_text_postprocessor.py
VIM (Virtual-logit Matching) Postprocessor for Text Embeddings

This module implements VIM for text OOD detection, adapting the approach from:
"ViM: Out-Of-Distribution with Virtual-logit Matching" (Wang et al.)

VIM approach:
1. Extracts features from pre-trained text encoder
2. Computes principal subspace using PCA on training features
3. Projects features to null space (orthogonal complement)
4. Uses norm in null space + energy score for OOD detection
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sentence_transformers import SentenceTransformer
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class VIMTextPostprocessor:
    """
    VIM Postprocessor for text embeddings.

    Uses Virtual-logit Matching approach adapted for text OOD detection:
    - Computes principal subspace from training embeddings
    - Projects test embeddings to null space
    - Combines null space norm with energy score
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_vim",
        dim: int = 512,  # Principal subspace dimension
    ):
        """
        Initialize VIM Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            dim: Dimension of principal subspace to keep.
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.dim = dim

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

        # VIM components (following reference pattern)
        self.w: np.ndarray | None = None  # Linear classifier weights
        self.b: np.ndarray | None = None  # Linear classifier bias
        self.u: np.ndarray | None = None  # Center point for features
        self.NS: np.ndarray | None = None  # Null space basis (principal complement)
        self.alpha: float | None = None  # Scaling factor

        # Binary classifier for creating w, b
        self.binary_classifier: LogisticRegression | None = None

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
            normalize_embeddings=True,
            device=self.device,
        )

        # Cache embeddings
        torch.save(embeddings, cache_path)
        print(f"Cached embeddings shape {embeddings.shape} to {cache_path}")

        # Clear cache again after encoding
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return embeddings.to(self.device)

    def _load_background_data(self) -> list[str]:
        """Load background data to create binary classification task."""
        print("Loading background data for VIM binary classifier...")

        # Dummy background data for demonstration
        background_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Climate change represents one of the most significant challenges facing humanity today.",
            "The human brain contains approximately 86 billion neurons interconnected in complex networks.",
            "Photosynthesis is the biological process by which plants convert sunlight into chemical energy.",
            "The Internet has revolutionized global communication and information sharing worldwide.",
            "Scientific research drives innovation and technological advancement in modern society.",
            "Economic markets fluctuate based on supply, demand, and investor sentiment patterns.",
            "Education plays a crucial role in personal development and societal progress overall.",
            "Art and creativity provide essential outlets for human expression and cultural identity.",
        ] * 50  # Repeat to get more samples

        return background_texts

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the VIM postprocessor following the exact reference pattern.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed.
        """
        if self.setup_flag:
            print("VIM postprocessor already set up.")
            return

        print("\n Extracting id training feature")

        # Extract ID features
        feature_id_train = self._extract_embeddings(id_texts, name="id_train")
        feature_id_train_np = feature_id_train.cpu().numpy().astype(np.float32)

        # Create binary classification task (ID vs background)
        print("Setting up binary classifier for VIM...")
        background_texts = self._load_background_data()
        background_features = self._extract_embeddings(background_texts, name="background")
        background_features_np = background_features.cpu().numpy().astype(np.float32)

        # Train binary classifier to get w, b (like the FC layer in reference)
        X_train = np.vstack([feature_id_train_np, background_features_np])
        y_train = np.hstack(
            [np.ones(len(feature_id_train_np)), np.zeros(len(background_features_np))]
        )

        self.binary_classifier = LogisticRegression(
            random_state=random_state, max_iter=1000, fit_intercept=True
        )
        self.binary_classifier.fit(X_train, y_train)

        # Extract w, b from trained classifier (following reference pattern)
        self.w = self.binary_classifier.coef_.astype(np.float32)  # [1, feature_dim]
        self.b = self.binary_classifier.intercept_.astype(np.float32)  # [1]

        print(f"Binary classifier weights shape: {self.w.shape}")
        print(f"Binary classifier bias shape: {self.b.shape}")

        # Compute logits for ID training data
        logit_id_train = feature_id_train_np @ self.w.T + self.b  # [n_samples, 1]

        # Following VIM reference pattern exactly:

        # 1. Compute u (center point): u = -pinv(w) * b
        self.u = -np.matmul(pinv(self.w), self.b)  # [feature_dim]
        print(f"Center point u shape: {self.u.shape}")

        # 2. Fit covariance on centered features
        ec = EmpiricalCovariance(assume_centered=True)
        centered_features = feature_id_train_np - self.u
        ec.fit(centered_features)

        # 3. Compute null space (NS) - eigenvectors corresponding to smallest eigenvalues
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        # Sort by eigenvalues (descending), take the complement of top dims
        sorted_indices = np.argsort(eig_vals * -1)  # Descending order
        self.NS = np.ascontiguousarray(
            (eigen_vectors.T[sorted_indices[self.dim :]]).T  # Take bottom eigenvectors
        )
        print(f"Null space basis shape: {self.NS.shape}")

        # 4. Compute scaling factor alpha
        vlogit_id_train = norm(np.matmul(feature_id_train_np - self.u, self.NS), axis=-1)
        self.alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f"alpha = {self.alpha:.4f}")

        self.setup_flag = True
        print("VIM postprocessor setup completed.")

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores following VIM pattern.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            VIM scores (higher = more in-distribution).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        assert self.w is not None
        assert self.b is not None
        assert self.u is not None
        assert self.NS is not None
        assert self.alpha is not None

        # Extract features
        feature_ood = self._extract_embeddings(texts, name=cache_name)
        feature_ood_np = feature_ood.cpu().numpy().astype(np.float32)

        # Following VIM reference pattern exactly:

        # 1. Compute logits using trained classifier
        logit_ood = feature_ood_np @ self.w.T + self.b  # [n_samples, 1]

        # 2. Compute energy score
        energy_ood = logsumexp(logit_ood, axis=-1)  # [n_samples]

        # 3. Compute virtual logit (norm in null space)
        vlogit_ood = (
            norm(np.matmul(feature_ood_np - self.u, self.NS), axis=-1) * self.alpha
        )  # [n_samples]

        # 4. VIM score = energy - virtual_logit (following reference)
        score_ood = energy_ood - vlogit_ood

        return score_ood

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
        Evaluate the VIM postprocessor.

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
        self.dim = hyperparam[0]

    def get_hyperparam(self) -> int:
        """Get hyperparameters (following reference API)."""
        return self.dim


# Convenience function matching other postprocessors
def create_vim_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> VIMTextPostprocessor:
    """
    Create VIM text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for VIMTextPostprocessor.

    Returns:
        Configured VIM postprocessor.
    """
    return VIMTextPostprocessor(embedding_model=embedding_model, **kwargs)
