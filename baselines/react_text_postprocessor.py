"""
Filename: react_text_postprocessor.py
ReAct (Rectified Activation) Postprocessor for Text Embeddings

This module implements ReAct for text OOD detection:
"ReAct: Out-of-distribution Detection With Rectified Activations"

ReAct approach:
1. Computes threshold from training activation percentile
2. Clips/rectifies activations above threshold to threshold value
3. Uses energy score from rectified activations for OOD detection
4. Simple but effective activation truncation method
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


class ReActTextPostprocessor:
    """
    ReAct Postprocessor for text embeddings.

    Uses rectified activation thresholding for OOD detection:
    - Computes activation threshold from training data percentile
    - Clips activations above threshold during inference
    - Uses energy score from rectified activations
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_react",
        percentile: float = 90.0,  # Percentile for activation threshold
    ):
        """
        Initialize ReAct Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            percentile: Percentile of training activations for threshold.
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.percentile = percentile

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

        # ReAct components (following reference pattern)
        self.activation_log: np.ndarray | None = None  # Training activations
        self.threshold: float | None = None  # Activation threshold

        # Binary classifier for creating logits (since we don't have forward_threshold)
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
            normalize_embeddings=False,  # ReAct works with raw activations
            device=self.device,
        )

        # Cache embeddings
        torch.save(embeddings, cache_path)
        print(f"Cached embeddings shape {embeddings.shape} to {cache_path}")

        # Clear cache again after encoding
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return embeddings.to(self.device)

    def _apply_threshold(self, features: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Apply ReAct thresholding to features.

        Args:
            features: Input features [batch_size, feature_dim].
            threshold: Activation threshold value.

        Returns:
            Thresholded features with activations clipped at threshold.
        """
        # ReAct: clip activations above threshold to threshold value
        return torch.clamp(features, max=threshold)

    def _load_background_data(self) -> list[str]:
        """Load background data to create binary classification for energy scores."""
        print("Loading background data for ReAct binary classifier...")

        # Dummy background data for demonstration
        background_texts = [
            "The weather today is sunny with clear blue skies and gentle winds.",
            "Technology advances have transformed communication methods in recent decades significantly.",
            "Scientific research continues to reveal new insights about natural phenomena and processes.",
            "Economic indicators suggest varying trends in global market performance this quarter.",
            "Educational institutions are adapting teaching methods to meet modern student needs.",
            "Environmental conservation efforts require collaborative action from communities worldwide.",
            "Cultural exchange programs promote understanding between different societies and traditions.",
            "Healthcare innovations are improving treatment outcomes for patients across diverse conditions.",
            "Transportation systems are evolving with sustainable and efficient mobility solutions.",
            "Agricultural practices are being modernized to enhance food security globally.",
        ] * 50  # Repeat to get more samples

        return background_texts

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the ReAct postprocessor following the exact reference pattern.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed.
        """
        if self.setup_flag:
            print("ReAct postprocessor already set up.")
            return

        print("\n Setup: Collecting activation statistics...")

        # Following reference pattern: collect activations from validation data
        # Since we don't have separate validation, we'll use a split of training data
        val_texts = list(id_texts)[::2]  # Use every other sample as "validation"

        activation_log = []

        # Extract features from validation texts
        features = self._extract_embeddings(val_texts, name="id_val_react")
        activation_log.append(features.cpu().numpy())

        # Concatenate all activations (following reference pattern exactly)
        self.activation_log = np.concatenate(activation_log, axis=0)

        print(f"Activation log shape: {self.activation_log.shape}")

        # Compute threshold at specified percentile (following reference pattern)
        self.threshold = np.percentile(self.activation_log.flatten(), self.percentile)
        print(
            f"Threshold at percentile {self.percentile:.1f} over id data is: {self.threshold:.6f}"
        )

        # Train binary classifier for energy score computation
        print("Training binary classifier for energy scores...")
        background_texts = self._load_background_data()
        background_features = self._extract_embeddings(background_texts, name="background")

        # Use full training data for classifier training
        id_features = self._extract_embeddings(id_texts, name="id_train_react")

        # Train binary classifier
        X_train = torch.vstack([id_features, background_features]).cpu().numpy()
        y_train = np.hstack([np.ones(len(id_features)), np.zeros(len(background_features))])

        self.binary_classifier = LogisticRegression(
            random_state=random_state, max_iter=1000, fit_intercept=True
        )
        self.binary_classifier.fit(X_train, y_train)

        self.setup_flag = True
        print("ReAct setup completed.")

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores following ReAct pattern.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            ReAct energy scores (higher = more in-distribution).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        assert self.threshold is not None
        assert self.binary_classifier is not None

        # Extract features
        features = self._extract_embeddings(texts, name=cache_name)

        # Following ReAct reference pattern exactly:
        # Apply threshold to features (rectified activations)
        features_thresholded = self._apply_threshold(features, self.threshold)

        # Get logits using binary classifier on thresholded features
        features_np = features_thresholded.cpu().numpy().astype(np.float32)
        id_logits = self.binary_classifier.decision_function(features_np)
        # Convert to binary logits [background_logit, id_logit]
        binary_logits = np.column_stack([-id_logits, id_logits])

        # Convert to torch for logsumexp computation
        output = torch.from_numpy(binary_logits).to(self.device)

        # Compute energy score (logsumexp)
        energyconf = torch.logsumexp(output, dim=1)

        return energyconf.cpu().numpy()

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
        Evaluate the ReAct postprocessor.

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
        self.percentile = hyperparam[0]
        if self.activation_log is not None:
            self.threshold = np.percentile(self.activation_log.flatten(), self.percentile)
            print(
                f"Threshold at percentile {self.percentile:.1f} over id data is: {self.threshold:.6f}"
            )

    def get_hyperparam(self) -> float:
        """Get hyperparameters (following reference API)."""
        return self.percentile


# Convenience function matching other postprocessors
def create_react_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> ReActTextPostprocessor:
    """
    Create ReAct text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for ReActTextPostprocessor.

    Returns:
        Configured ReAct postprocessor.
    """
    return ReActTextPostprocessor(embedding_model=embedding_model, **kwargs)
