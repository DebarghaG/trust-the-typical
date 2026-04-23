"""
Filename: fdbd_text_postprocessor.py
fDBD (feature Distance-Based Detection) Postprocessor for Text Embeddings

This module implements fDBD for text OOD detection:
"Feature Distance-Based Detection of Out-of-Distribution Examples"

fDBD approach:
1. Trains a binary classifier on ID vs background features
2. Computes denominator matrix from classifier weights
3. Uses logit differences normalized by feature distance/norm
4. Score = sum(|logit_i - max_logit| / denominator) / feature_regularizer
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class fDBDTextPostprocessor:
    """
    fDBD Postprocessor for text embeddings.

    Uses feature distance-based detection adapted for text OOD:
    - Trains binary classifier to get decision boundaries
    - Computes denominator matrix from classifier weights
    - Normalizes logit differences by feature distance or norm
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_fdbd",
        distance_as_normalizer: bool = True,  # Use distance to mean vs. feature norm
    ):
        """
        Initialize fDBD Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            distance_as_normalizer: If True, use distance to mean; else use feature norm.
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.distance_as_normalizer = distance_as_normalizer

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

        # fDBD components (following reference pattern)
        self.train_mean: torch.Tensor | None = None  # Training feature mean
        self.denominator_matrix: torch.Tensor | None = None  # Denominator matrix
        self.num_classes: int = 2  # Binary classification (ID vs OOD)
        self.activation_log: np.ndarray | None = None  # Training features

        # Binary classifier components
        self.binary_classifier: LogisticRegression | None = None
        self.w: np.ndarray | None = None  # Classifier weights
        self.b: np.ndarray | None = None  # Classifier bias

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
            normalize_embeddings=False,  # fDBD works with raw embeddings
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
        print("Loading background data for fDBD binary classifier...")

        # Dummy background data for demonstration
        background_texts = [
            "The quick brown fox jumps over the lazy dog and runs through the forest.",
            "Machine learning algorithms have revolutionized data analysis and artificial intelligence applications.",
            "Climate change represents one of the most significant environmental challenges of the 21st century.",
            "The human brain's neural networks process information through billions of interconnected synapses.",
            "Photosynthesis converts solar energy into chemical energy, sustaining most life on Earth.",
            "Global economic systems are influenced by trade policies, market dynamics, and technological innovations.",
            "Education systems worldwide are adapting to digital transformation and remote learning technologies.",
            "Space exploration has led to numerous scientific discoveries and technological breakthroughs.",
            "Renewable energy sources like solar and wind power are becoming increasingly cost-effective.",
            "Cultural diversity enriches societies through varied perspectives, traditions, and creative expressions.",
        ] * 50  # Repeat to get more samples

        return background_texts

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the fDBD postprocessor following the exact reference pattern.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed.
        """
        if self.setup_flag:
            print("fDBD postprocessor already set up.")
            return

        print("\n Setup: Collecting training features...")

        # Extract features from training texts (following reference pattern)
        activation_log = []

        # Extract all features from training texts
        features = self._extract_embeddings(id_texts, name="id_train")
        activation_log.append(features.cpu().numpy())

        # Concatenate all features (following reference pattern exactly)
        activation_log_concat = np.concatenate(activation_log, axis=0)
        self.activation_log = activation_log_concat

        # Compute training mean
        self.train_mean = torch.from_numpy(np.mean(activation_log_concat, axis=0)).to(self.device)

        print(f"Training features shape: {self.activation_log.shape}")
        print(f"Training mean shape: {self.train_mean.shape}")

        # Create binary classification task (ID vs background) to get classifier weights
        print("Setting up binary classifier for fDBD...")
        background_texts = self._load_background_data()
        background_features = self._extract_embeddings(background_texts, name="background")
        background_features_np = background_features.cpu().numpy()

        # Train binary classifier
        X_train = np.vstack([self.activation_log, background_features_np])
        y_train = np.hstack(
            [np.ones(len(self.activation_log)), np.zeros(len(background_features_np))]
        )

        self.binary_classifier = LogisticRegression(
            random_state=random_state, max_iter=1000, fit_intercept=True
        )
        self.binary_classifier.fit(X_train, y_train)

        # Extract weights and bias (following reference pattern)
        self.w = self.binary_classifier.coef_  # [1, feature_dim]
        self.b = self.binary_classifier.intercept_  # [1]

        print(f"Binary classifier weights shape: {self.w.shape}")
        print(f"Binary classifier bias shape: {self.b.shape}")

        # Compute denominator matrix following reference pattern
        # For binary case, we create a 2x2 matrix
        self.num_classes = 2

        denominator_matrix = np.zeros((self.num_classes, self.num_classes))

        # Create weight matrix for binary case
        w_full = np.vstack([self.w, -self.w])  # [2, feature_dim] - pos and neg class

        for p in range(self.num_classes):
            w_p = w_full - w_full[p, :]  # Subtract class p weights
            denominator = np.linalg.norm(w_p, axis=1)
            denominator[p] = 1  # Avoid division by zero
            denominator_matrix[p, :] = denominator

        self.denominator_matrix = torch.tensor(denominator_matrix, dtype=torch.float32).to(
            self.device
        )

        print(f"Denominator matrix shape: {self.denominator_matrix.shape}")
        print("fDBD setup completed.")

        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores following fDBD pattern.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            fDBD scores (lower = more OOD).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        assert self.w is not None
        assert self.b is not None
        assert self.denominator_matrix is not None
        assert self.train_mean is not None

        # Extract features
        features = self._extract_embeddings(texts, name=cache_name)

        # Compute logits using binary classifier
        features_np = features.cpu().numpy().astype(np.float32)
        logits = features_np @ self.w.T + self.b  # [n_samples, 1]

        # Convert to binary classification logits
        # Create logits for both classes: [logit_pos, logit_neg]
        output = torch.from_numpy(np.hstack([logits, -logits])).to(self.device)  # [n_samples, 2]

        # Following fDBD reference pattern exactly:
        values, nn_idx = output.max(1)  # Max logit and predicted class

        # Compute absolute differences from max logit
        logits_sub = torch.abs(output - values.repeat(self.num_classes, 1).T)

        # Normalize by denominator matrix
        denominator_terms = logits_sub / self.denominator_matrix[nn_idx]
        numerator = torch.sum(denominator_terms, axis=1)

        # Choose normalizer based on configuration
        if self.distance_as_normalizer:
            # Use distance to training mean
            normalizer = torch.norm(features - self.train_mean, dim=1)
        else:
            # Use feature norm
            normalizer = torch.norm(features, dim=1)

        # Avoid division by zero
        normalizer = torch.clamp(normalizer, min=1e-8)

        # fDBD score = numerator / normalizer
        score = numerator / normalizer

        # Convert to numpy and flip sign (lower score = more OOD)
        fdbd_scores = -score.cpu().numpy()  # Negative so higher = more ID

        return fdbd_scores

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
        Evaluate the fDBD postprocessor.

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
        self.distance_as_normalizer = hyperparam[0]

    def get_hyperparam(self) -> bool:
        """Get hyperparameters (following reference API)."""
        return self.distance_as_normalizer


# Convenience function matching other postprocessors
def create_fdbd_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> fDBDTextPostprocessor:
    """
    Create fDBD text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for fDBDTextPostprocessor.

    Returns:
        Configured fDBD postprocessor.
    """
    return fDBDTextPostprocessor(embedding_model=embedding_model, **kwargs)
