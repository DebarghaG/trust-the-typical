"""
Filename: gmm_text_postprocessor.py
GMM (Gaussian Mixture Model) Postprocessor for Text Embeddings

This module implements GMM-based OOD detection for text:
"Deep Ensembles vs. Monte Carlo Dropout for Distributed Uncertainty Estimation"

GMM approach:
1. Fits Gaussian Mixture Model on training text embeddings
2. Computes likelihood of test embeddings under fitted GMM
3. Uses GMM likelihood as OOD confidence score
4. Supports multiple components and dimensionality reduction
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


def tensor2list(tensor: torch.Tensor) -> list:
    """Convert tensor to list."""
    return tensor.detach().cpu().numpy().tolist()


def process_feature_type(features: torch.Tensor, feature_type: str = "penultimate") -> torch.Tensor:
    """Process features based on type."""
    # For text embeddings, we typically use the raw embeddings
    if feature_type in ["penultimate", "raw", "embedding"]:
        return features
    elif feature_type == "norm":
        return torch.norm(features, dim=1, keepdim=True)
    elif feature_type == "normalized":
        return torch.nn.functional.normalize(features, dim=1)
    else:
        return features


def reduce_feature_dim(
    features: np.ndarray,
    labels: np.ndarray,
    reduce_method: str = "none",
    target_dim: int = 50,
) -> np.ndarray:
    """
    Reduce feature dimensionality.

    Args:
        features: Feature array [n_samples, feature_dim].
        labels: Label array [n_samples].
        reduce_method: Dimensionality reduction method.
        target_dim: Target dimension.

    Returns:
        Transform matrix for dimensionality reduction.
    """
    if reduce_method == "none" or reduce_method is None:
        return np.eye(features.shape[1])
    elif reduce_method == "pca":
        pca = PCA(n_components=min(target_dim, features.shape[1], features.shape[0]))
        pca.fit(features)
        return pca.components_.T
    elif reduce_method == "lda":
        # Use LDA if we have labels and multiple classes
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            lda = LinearDiscriminantAnalysis(n_components=min(target_dim, len(unique_labels) - 1))
            lda.fit(features, labels)
            # Pad with identity if needed
            transform_matrix = lda.scalings_
            if transform_matrix.shape[1] < features.shape[1]:
                padding = np.eye(features.shape[1], features.shape[1] - transform_matrix.shape[1])
                transform_matrix = np.hstack([transform_matrix, padding])
            return transform_matrix
        else:
            return np.eye(features.shape[1])
    else:
        return np.eye(features.shape[1])


class GMMTextPostprocessor:
    """
    GMM Postprocessor for text embeddings.

    Uses Gaussian Mixture Models for OOD detection:
    - Fits GMM on training embeddings
    - Computes likelihood for test embeddings
    - Supports dimensionality reduction and multiple components
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_gmm",
        num_clusters: int = 8,  # Number of GMM components
        feature_type: str = "penultimate",  # Feature processing type
        reduce_dim_method: str = "none",  # Dimensionality reduction
        target_dim: int = 50,  # Target dimension for reduction
        covariance_type: str = "tied",  # GMM covariance type
        use_sklearn_gmm: bool = True,  # Use sklearn GMM (True) or custom implementation (False)
    ):
        """
        Initialize GMM Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            num_clusters: Number of GMM components.
            feature_type: Feature processing type.
            reduce_dim_method: Dimensionality reduction method.
            target_dim: Target dimension after reduction.
            covariance_type: GMM covariance type ('tied', 'full', 'diag', 'spherical').
            use_sklearn_gmm: Use sklearn's GMM (True) or custom torch implementation (False).
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.feature_type = feature_type
        self.reduce_dim_method = reduce_dim_method
        self.target_dim = target_dim
        self.covariance_type = covariance_type
        self.use_sklearn_gmm = use_sklearn_gmm

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

        # GMM components (for custom torch implementation)
        self.feature_mean: torch.Tensor | None = None  # GMM means
        self.feature_prec: torch.Tensor | None = None  # GMM precision matrices
        self.component_weight: torch.Tensor | None = None  # GMM component weights
        self.transform_matrix: torch.Tensor | None = None  # Dimensionality reduction matrix

        # Sklearn GMM (for library implementation)
        self.sklearn_gmm: GaussianMixture | None = None
        self.sklearn_reducer: Any | None = None  # PCA/LDA reducer

        # Binary classifier for creating labels
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
            normalize_embeddings=False,  # GMM works with raw embeddings
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
        """Load background data for creating pseudo-labels."""
        print("Loading background data for GMM label creation...")

        # Dummy background data for demonstration
        background_texts = [
            "Technology advances continue to reshape modern society and business practices.",
            "Environmental conservation requires coordinated global efforts and sustainable policies.",
            "Educational systems worldwide are adapting to digital learning methodologies rapidly.",
            "Healthcare innovations are improving patient outcomes through advanced medical techniques.",
            "Economic markets fluctuate based on various geopolitical and social factors.",
            "Cultural diversity enriches communities through varied perspectives and traditions.",
            "Scientific research drives discoveries that expand our understanding of nature.",
            "Transportation infrastructure development focuses on efficiency and environmental impact.",
            "Communication technologies enable instant global connectivity and information sharing.",
            "Agricultural innovations help address food security challenges in developing regions.",
        ] * 50  # Repeat to get more samples

        return background_texts

    @torch.no_grad()
    def get_GMM_stat(self, id_texts: Sequence[str]) -> tuple:
        """
        Compute GMM statistics from training data.

        Args:
            id_texts: In-distribution training texts.

        Returns:
            Tuple of GMM parameters (means, precisions, weights, transform_matrix).
        """
        print("Compute GMM Stats [Collecting]")

        # Extract features and create pseudo-labels for dimensionality reduction
        feature_all = []
        label_list = []

        # Extract ID features
        id_features = self._extract_embeddings(id_texts, name="id_train_gmm")
        id_features_processed = process_feature_type(id_features, self.feature_type)
        feature_all.extend(tensor2list(id_features_processed))
        label_list.extend([1] * len(id_texts))  # ID label = 1

        # Extract background features for better GMM fitting
        background_texts = self._load_background_data()
        bg_features = self._extract_embeddings(background_texts, name="background_gmm")
        bg_features_processed = process_feature_type(bg_features, self.feature_type)
        feature_all.extend(tensor2list(bg_features_processed))
        label_list.extend([0] * len(background_texts))  # Background label = 0

        feature_all = np.array(feature_all)
        label_list = np.array(label_list)

        print("Compute GMM Stats [Estimating]")

        # Reduce feature dimensionality
        transform_matrix = reduce_feature_dim(
            feature_all, label_list, self.reduce_dim_method, self.target_dim
        )
        feature_reduced = np.dot(feature_all, transform_matrix)

        # Fit GMM on reduced features (use only ID data for GMM fitting)
        id_features_reduced = feature_reduced[label_list == 1]  # Only ID data

        print(
            f"Fitting GMM with {self.num_clusters} components on {len(id_features_reduced)} ID samples"
        )
        print(f"Feature dimension after reduction: {id_features_reduced.shape[1]}")

        gm = GaussianMixture(
            n_components=self.num_clusters,
            random_state=42,
            covariance_type=self.covariance_type,
            max_iter=100,
            tol=1e-3,
        )
        gm.fit(id_features_reduced)

        # Extract GMM parameters
        feature_mean = torch.tensor(gm.means_, dtype=torch.float32).to(self.device)

        # Handle precision matrices based on covariance type
        if self.covariance_type == "tied":
            feature_prec = torch.tensor(np.linalg.inv(gm.covariances_), dtype=torch.float32).to(
                self.device
            )
        elif self.covariance_type == "full":
            feature_prec = torch.tensor(np.linalg.inv(gm.covariances_), dtype=torch.float32).to(
                self.device
            )
        elif self.covariance_type == "diag":
            feature_prec = torch.tensor(1.0 / gm.covariances_, dtype=torch.float32).to(self.device)
        else:  # spherical
            feature_prec = torch.tensor(1.0 / gm.covariances_, dtype=torch.float32).to(self.device)

        component_weight = torch.tensor(gm.weights_, dtype=torch.float32).to(self.device)
        transform_matrix_tensor = torch.tensor(transform_matrix, dtype=torch.float32).to(
            self.device
        )

        print(f"GMM means shape: {feature_mean.shape}")
        print(f"GMM precision shape: {feature_prec.shape}")
        print(f"GMM weights shape: {component_weight.shape}")
        print(f"Transform matrix shape: {transform_matrix_tensor.shape}")

        return feature_mean, feature_prec, component_weight, transform_matrix_tensor

    @torch.no_grad()
    def setup_sklearn_GMM(self, id_texts: Sequence[str]) -> None:
        """
        Setup sklearn GMM (simpler approach).

        Args:
            id_texts: In-distribution training texts.
        """
        print("Setup sklearn GMM [Collecting]")

        # Extract ID features
        id_features = self._extract_embeddings(id_texts, name="id_train_gmm_sklearn")
        id_features_processed = process_feature_type(id_features, self.feature_type)
        features_np = id_features_processed.cpu().numpy()

        print("Setup sklearn GMM [Reducing Dimensions]")

        # Apply dimensionality reduction if specified
        if self.reduce_dim_method == "none" or self.reduce_dim_method is None:
            features_reduced = features_np
            self.sklearn_reducer = None
        elif self.reduce_dim_method == "pca":
            from sklearn.decomposition import PCA

            self.sklearn_reducer = PCA(
                n_components=min(self.target_dim, features_np.shape[1], features_np.shape[0])
            )
            features_reduced = self.sklearn_reducer.fit_transform(features_np)
        elif self.reduce_dim_method == "lda":
            # For LDA we need labels, create dummy binary labels
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            # Use background data to create binary classification task
            background_texts = self._load_background_data()[: len(id_texts)]  # Match size
            bg_features = self._extract_embeddings(background_texts, name="background_gmm_sklearn")
            bg_features_processed = process_feature_type(bg_features, self.feature_type)
            bg_features_np = bg_features_processed.cpu().numpy()

            # Combine ID and background for LDA training
            all_features = np.vstack([features_np, bg_features_np])
            all_labels = np.hstack([np.ones(len(features_np)), np.zeros(len(bg_features_np))])

            self.sklearn_reducer = LinearDiscriminantAnalysis(
                n_components=min(self.target_dim, 1)
            )  # Binary task = 1 component max
            self.sklearn_reducer.fit(all_features, all_labels)
            features_reduced = self.sklearn_reducer.transform(
                features_np
            )  # Only transform ID data for GMM
        else:
            features_reduced = features_np
            self.sklearn_reducer = None

        print(f"Feature dimension after reduction: {features_reduced.shape[1]}")

        print("Setup sklearn GMM [Fitting]")

        # Fit sklearn GMM on reduced features
        self.sklearn_gmm = GaussianMixture(
            n_components=self.num_clusters,
            covariance_type=self.covariance_type,
            random_state=42,
            max_iter=100,
            tol=1e-3,
        )
        self.sklearn_gmm.fit(features_reduced)

        print(f"Sklearn GMM fitted with {self.num_clusters} components")
        print(f"GMM converged: {self.sklearn_gmm.converged_}")
        print(f"Log likelihood: {self.sklearn_gmm.lower_bound_:.2f}")

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the GMM postprocessor.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed.
        """
        if self.setup_flag:
            print("GMM postprocessor already set up.")
            return

        if self.use_sklearn_gmm:
            print("\n Setup: Using sklearn GMM (default)...")
            self.setup_sklearn_GMM(id_texts)
        else:
            print("\n Setup: Using custom torch GMM...")
            # Compute GMM statistics using custom implementation
            self.feature_mean, self.feature_prec, self.component_weight, self.transform_matrix = (
                self.get_GMM_stat(id_texts)
            )

        self.setup_flag = True
        print("GMM setup completed.")

    @torch.no_grad()
    def compute_GMM_score(
        self, features: torch.Tensor, return_pred: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GMM likelihood score.

        Args:
            features: Input features [batch_size, feature_dim].
            return_pred: Whether to return prediction.

        Returns:
            GMM likelihood scores (and predictions if requested).
        """
        assert self.transform_matrix is not None
        assert self.feature_mean is not None
        assert self.feature_prec is not None
        assert self.component_weight is not None
        # Process features and ensure consistent dtype
        feature_processed = process_feature_type(features, self.feature_type).float()

        # Apply dimensionality reduction
        feature_reduced = torch.mm(feature_processed, self.transform_matrix)

        # Compute GMM likelihood for each component
        prob_list = []

        for cluster_idx in range(len(self.feature_mean)):
            # Compute Mahalanobis distance
            diff = feature_reduced - self.feature_mean[cluster_idx]  # [batch_size, reduced_dim]

            if self.covariance_type == "tied":
                # Single precision matrix for all components
                mahal_dist = -0.5 * torch.sum(diff * torch.mm(diff, self.feature_prec), dim=1)
            elif self.covariance_type == "full":
                # Different precision matrix for each component
                mahal_dist = -0.5 * torch.sum(
                    diff * torch.mm(diff, self.feature_prec[cluster_idx]), dim=1
                )
            elif self.covariance_type == "diag":
                # Diagonal covariance
                mahal_dist = -0.5 * torch.sum(diff * diff * self.feature_prec[cluster_idx], dim=1)
            else:  # spherical
                # Spherical covariance
                mahal_dist = -0.5 * torch.sum(diff * diff, dim=1) * self.feature_prec[cluster_idx]

            prob_gau = torch.exp(mahal_dist)
            prob_list.append(prob_gau.view(-1, 1))

        # Combine probabilities from all components
        prob_matrix = torch.cat(prob_list, dim=1)  # [batch_size, n_components]
        prob = torch.mm(prob_matrix, self.component_weight.view(-1, 1))  # [batch_size, 1]

        if return_pred:
            # Create dummy predictions (since we don't have a classifier)
            pred = torch.zeros(features.size(0), dtype=torch.long, device=self.device)
            return pred, prob.squeeze()
        else:
            return prob.squeeze()

    @torch.no_grad()
    def compute_sklearn_GMM_score(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute GMM likelihood score using sklearn.

        Args:
            features: Input features [batch_size, feature_dim].

        Returns:
            GMM log likelihood scores.
        """
        assert self.sklearn_gmm is not None
        # Process features
        feature_processed = process_feature_type(features, self.feature_type)
        features_np = feature_processed.cpu().numpy().astype(np.float32)

        # Apply dimensionality reduction if available
        if self.sklearn_reducer is not None:
            features_reduced = self.sklearn_reducer.transform(features_np)
        else:
            features_reduced = features_np

        # Compute log likelihood using sklearn GMM
        log_likelihood = self.sklearn_gmm.score_samples(features_reduced)

        return log_likelihood

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores using GMM.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            GMM likelihood scores (higher = more in-distribution).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        # Extract features
        features = self._extract_embeddings(texts, name=cache_name)

        if self.use_sklearn_gmm:
            # Use sklearn GMM (simpler and more stable)
            log_scores = self.compute_sklearn_GMM_score(features)
        else:
            # Use custom torch GMM implementation
            scores = self.compute_GMM_score(features, return_pred=False)
            # Take log for numerical stability
            log_scores = torch.log(scores + 1e-45).cpu().numpy()

        return log_scores

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
        Evaluate the GMM postprocessor.

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
        if len(hyperparam) >= 1:
            self.num_clusters = hyperparam[0]

    def get_hyperparam(self) -> list:
        """Get hyperparameters (following reference API)."""
        return [self.num_clusters]


# Convenience function matching other postprocessors
def create_gmm_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    use_sklearn_gmm: bool = True,
    **kwargs: typing.Any,
) -> GMMTextPostprocessor:
    """
    Create GMM text detector.

    Args:
        embedding_model: Text embedding model to use.
        use_sklearn_gmm: Use sklearn GMM (True) or custom torch implementation (False).
        **kwargs: Additional arguments for GMMTextPostprocessor.

    Returns:
        Configured GMM postprocessor.
    """
    return GMMTextPostprocessor(
        embedding_model=embedding_model, use_sklearn_gmm=use_sklearn_gmm, **kwargs
    )
