"""
Filename: adascale_text_postprocessor.py
AdaScale (Adaptive Scaling) Postprocessor for Text Embeddings

This module implements AdaScale for text OOD detection:
"AdaScale: Towards Real-time Video Object Segmentation"

AdaScale approach:
1. Computes gradient perturbations on input embeddings
2. Analyzes feature shifts between original and perturbed embeddings
3. Uses top-k features and adaptive percentile thresholding
4. Combines correction terms with shift magnitudes for OOD scoring
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class AdaScaleTextPostprocessor:
    """
    AdaScale Postprocessor for text embeddings.

    Uses adaptive scaling with gradient perturbations for OOD detection:
    - Computes gradients on text embeddings
    - Applies perturbations and measures feature shifts
    - Uses top-k feature selection with adaptive thresholding
    - Combines correction terms for final OOD scoring
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_adascale",
        percentile: tuple[float, float] = (90.0, 99.0),  # Min/max percentile range
        k1: float = 50.0,  # Top-k1 percentage for correction term
        k2: float = 50.0,  # Top-k2 percentage for shift term
        lmbda: float = 1.0,  # Weight for shift term
        o: float = 0.1,  # Perturbation strength
        num_samples: int | None = None,  # Subsample for efficiency
    ):
        """
        Initialize AdaScale Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            percentile: Min/max percentile range for adaptive thresholding.
            k1: Top-k1 percentage for correction term calculation.
            k2: Top-k2 percentage for shift term calculation.
            lmbda: Lambda weight for combining shift terms.
            o: Perturbation strength for gradient-based perturbations.
            num_samples: Number of samples to use for setup (None = all).
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.percentile = percentile
        self.k1 = k1
        self.k2 = k2
        self.lmbda = lmbda
        self.o = o
        self.num_samples = num_samples

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

        # AdaScale components
        self.feature_log: torch.Tensor | None = None  # Training features
        self.feature_perturbed_log: torch.Tensor | None = None  # Perturbed features
        self.feature_shift_log: torch.Tensor | None = None  # Feature shifts
        self.feature_dim: int | None = None  # Feature dimension
        self.ecdf: ECDF | None = None  # Empirical CDF for percentile computation
        self.min_percentile: float = 0.0
        self.max_percentile: float = 0.0
        self.k1_: int = 0  # Actual k1 value
        self.k2_: int = 0  # Actual k2 value

        # Binary classifier for creating decision boundaries
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
            normalize_embeddings=False,  # AdaScale works with raw embeddings
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
        """Load background data for creating binary classification task."""
        print("Loading background data for AdaScale binary classifier...")

        # Dummy background data for demonstration
        background_texts = [
            "Technology continues to advance rapidly in various sectors of modern society.",
            "Environmental sustainability requires global cooperation and innovative solutions.",
            "Education systems are evolving to incorporate digital learning methodologies.",
            "Healthcare research leads to breakthroughs in treatment and prevention methods.",
            "Economic policies influence market dynamics and international trade relationships.",
            "Cultural exchanges promote understanding between diverse communities worldwide.",
            "Scientific discoveries expand our knowledge of natural phenomena and processes.",
            "Transportation networks are being modernized for efficiency and sustainability.",
            "Communication technologies enable instant global connectivity and collaboration.",
            "Agricultural innovations address food security challenges in developing regions.",
        ] * 50  # Repeat to get more samples

        return background_texts

    def _perturb_embeddings(
        self, embeddings: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gradient-based perturbations to embeddings.

        Args:
            embeddings: Original embeddings [batch_size, feature_dim].
            gradients: Gradient information [batch_size, feature_dim].

        Returns:
            Perturbed embeddings.
        """
        batch_size, feature_dim = embeddings.shape
        n_features = int(feature_dim * self.o)  # Number of features to perturb

        # Find features with smallest absolute gradients (most stable)
        abs_grad = torch.abs(gradients)
        _, topk_indices = torch.topk(abs_grad, n_features, dim=1, largest=False)

        # Create perturbation mask
        mask = torch.zeros_like(abs_grad, dtype=torch.float32)
        mask.scatter_(1, topk_indices, 1.0)

        # Apply perturbation: add gradient direction with mask
        embeddings_perturbed = embeddings + torch.sign(gradients) * mask * 0.1

        return embeddings_perturbed

    def _compute_gradients(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute pseudo-gradients for embeddings using classifier weights.

        Since we can't backpropagate through sklearn classifier directly,
        we approximate gradients using the classifier weights.

        Args:
            embeddings: Input embeddings.
            labels: Predicted class labels.

        Returns:
            Pseudo-gradients based on classifier weights.
        """
        # Get classifier weights and bias, ensure correct dtype
        assert self.binary_classifier is not None
        w = (
            torch.from_numpy(self.binary_classifier.coef_[0]).to(self.device).float()
        )  # [feature_dim]
        b = torch.from_numpy(self.binary_classifier.intercept_).to(self.device).float()  # [1]

        # Ensure embeddings are float32 for consistency
        embeddings = embeddings.float()

        # Create a loss that we can differentiate
        embeddings_grad = embeddings.detach().clone().float().requires_grad_(True)
        logits_grad = torch.matmul(embeddings_grad, w) + b

        # Use sigmoid loss for binary classification
        loss = torch.sum(torch.log(1 + torch.exp(-logits_grad * labels.float())))

        loss.backward()
        gradients = embeddings_grad.grad.data

        return gradients

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the AdaScale postprocessor.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed.
        """
        if self.setup_flag:
            print("AdaScale postprocessor already set up.")
            return

        print("\n Setup: Collecting AdaScale statistics...")

        # Setup hyperparameters
        self.min_percentile, self.max_percentile = self.percentile

        # Extract training features
        print("Extracting training embeddings...")
        id_features = self._extract_embeddings(id_texts, name="id_train_adascale")
        self.feature_dim = id_features.shape[1]

        # Convert percentages to actual k values
        self.k1_ = int(self.feature_dim * self.k1 / 100)
        self.k2_ = int(self.feature_dim * self.k2 / 100)

        print(f"Feature dimension: {self.feature_dim}")
        print(f"k1: {self.k1_}, k2: {self.k2_}")

        # Train binary classifier for gradient computation
        print("Training binary classifier for gradient computation...")
        background_texts = self._load_background_data()
        background_features = self._extract_embeddings(background_texts, name="background_adascale")

        # Train binary classifier
        X_train = torch.vstack([id_features, background_features]).cpu().numpy()
        y_train = np.hstack([np.ones(len(id_features)), np.zeros(len(background_features))])

        self.binary_classifier = LogisticRegression(
            random_state=random_state, max_iter=1000, fit_intercept=True
        )
        self.binary_classifier.fit(X_train, y_train)

        # Subsample if specified
        if self.num_samples is not None and self.num_samples < len(id_features):
            indices = torch.randperm(len(id_features))[: self.num_samples]
            id_features = id_features[indices]
            print(f"Subsampled to {self.num_samples} features for efficiency")

        # Compute perturbations and feature shifts
        print("Computing gradients and perturbations...")
        feature_log = []
        feature_perturbed_log = []
        feature_shift_log = []

        # Process in batches to avoid memory issues
        batch_size = min(self.batch_size, len(id_features))
        for i in tqdm(range(0, len(id_features), batch_size), desc="Processing batches"):
            batch_features = id_features[i : i + batch_size]

            # Get predicted labels
            batch_np = batch_features.cpu().numpy()
            predicted_labels = self.binary_classifier.predict(batch_np)
            predicted_labels = torch.from_numpy(predicted_labels).to(self.device)

            # Enable gradients for this batch
            with torch.enable_grad():
                # Compute gradients
                gradients = self._compute_gradients(batch_features, predicted_labels)

            # Apply perturbations
            perturbed_features = self._perturb_embeddings(batch_features, gradients)

            # Compute feature shifts
            feature_shift = torch.abs(batch_features - perturbed_features)

            # Store results
            feature_log.append(batch_features.cpu())
            feature_perturbed_log.append(perturbed_features.cpu())
            feature_shift_log.append(feature_shift.cpu())

        # Concatenate all results
        self.feature_log = torch.cat(feature_log, dim=0)
        self.feature_perturbed_log = torch.cat(feature_perturbed_log, dim=0)
        self.feature_shift_log = torch.cat(feature_shift_log, dim=0)

        # Compute ECDF for percentile computation
        print("Computing empirical CDF...")
        topk_indices = torch.topk(self.feature_log, k=self.k1_, dim=1)[1]
        topk_feature_perturbed = torch.gather(
            torch.relu(self.feature_perturbed_log), 1, topk_indices
        )
        topk_indices = torch.topk(self.feature_log, k=self.k2_, dim=1)[1]
        topk_feature_shift_log = torch.gather(self.feature_shift_log, 1, topk_indices)
        sum_log = topk_feature_perturbed.sum(dim=1) + self.lmbda * topk_feature_shift_log.sum(dim=1)
        self.ecdf = ECDF(sum_log.numpy())

        print(f"Setup completed with {len(self.feature_log)} samples")
        self.setup_flag = True

    @torch.no_grad()
    def get_percentile(
        self, feature: torch.Tensor, feature_perturbed: torch.Tensor, feature_shift: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive percentile based on feature analysis.

        Args:
            feature: Original features.
            feature_perturbed: Perturbed features.
            feature_shift: Feature shift magnitudes.

        Returns:
            Adaptive percentile values.
        """
        # Get top-k features for correction term
        topk_indices = torch.topk(feature, dim=1, k=self.k1_)[1]
        topk_feature_perturbed = torch.gather(
            torch.relu(feature_perturbed), 1, topk_indices
        )  # Correction term C_o

        # Get top-k features for shift term
        topk_indices = torch.topk(feature, dim=1, k=self.k2_)[1]
        topk_feature_shift = torch.gather(feature_shift, 1, topk_indices)  # Q

        # Combine terms
        topk_norm = topk_feature_perturbed.sum(dim=1) + self.lmbda * topk_feature_shift.sum(
            dim=1
        )  # Q'

        # Compute adaptive percentile using ECDF
        assert self.ecdf is not None
        percent = 1 - self.ecdf(topk_norm.cpu().numpy())
        percentile = self.min_percentile + percent * (self.max_percentile - self.min_percentile)

        return torch.from_numpy(percentile).to(self.device)

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores using AdaScale.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            AdaScale confidence scores (higher = more in-distribution).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        # Extract features
        features = self._extract_embeddings(texts, name=cache_name)

        # Get predicted labels
        features_np = features.cpu().numpy()
        assert self.binary_classifier is not None
        predicted_labels = self.binary_classifier.predict(features_np)
        predicted_labels = torch.from_numpy(predicted_labels).to(self.device)

        # Compute gradients with enable_grad context
        with torch.enable_grad():
            gradients = self._compute_gradients(features, predicted_labels)

        # Apply perturbations
        features_perturbed = self._perturb_embeddings(features, gradients)

        # Compute feature shifts
        feature_shift = torch.abs(features - features_perturbed)

        # Get adaptive percentiles
        percentile = self.get_percentile(features, features_perturbed, feature_shift)

        # Compute confidence scores using logits and percentile scaling
        logits = self.binary_classifier.decision_function(features_np)
        logits_tensor = torch.from_numpy(logits).to(self.device)

        # Scale logits by percentile (higher percentile = more confident)
        scaled_logits = logits_tensor * (percentile / 100.0)

        # Use log-sum-exp for final confidence
        # Create binary logits for log-sum-exp
        binary_logits = torch.stack([-scaled_logits, scaled_logits], dim=1)
        conf = torch.logsumexp(binary_logits, dim=1)

        return conf.cpu().numpy()

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
        Evaluate the AdaScale postprocessor.

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
        if len(hyperparam) >= 5:
            self.percentile = hyperparam[0] if isinstance(hyperparam[0], tuple) else (90.0, 99.0)
            self.k1 = hyperparam[1]
            self.k2 = hyperparam[2]
            self.lmbda = hyperparam[3]
            self.o = hyperparam[4]

            # Update derived parameters if already setup
            if self.setup_flag and self.feature_dim is not None:
                self.min_percentile, self.max_percentile = self.percentile
                self.k1_ = int(self.feature_dim * self.k1 / 100)
                self.k2_ = int(self.feature_dim * self.k2 / 100)

    def get_hyperparam(self) -> list:
        """Get hyperparameters (following reference API)."""
        return [self.percentile, self.k1, self.k2, self.lmbda, self.o]


# Convenience function matching other postprocessors
def create_adascale_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: Any
) -> AdaScaleTextPostprocessor:
    """
    Create AdaScale text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for AdaScaleTextPostprocessor.

    Returns:
        Configured AdaScale postprocessor.
    """
    return AdaScaleTextPostprocessor(embedding_model=embedding_model, **kwargs)
