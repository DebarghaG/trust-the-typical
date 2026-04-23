"""
Filename: nnguide_text_postprocessor.py
NNGuide (Nearest Neighbor Guide) Postprocessor for Text Embeddings (without FAISS)

This module implements NNGuide for text OOD detection without FAISS dependency:
"Nearest Neighbor Guided OOD Detection"

NNGuide approach:
1. Creates "guided" bank features: normalized_features * confidence_scores
2. For test samples, finds K nearest neighbors in guided feature space
3. Combines KNN similarity with energy score for final OOD detection
4. Uses efficient numpy operations instead of FAISS
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch
from scipy.special import logsumexp
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


def normalizer(x: np.ndarray) -> np.ndarray:
    """Normalize vectors along last dimension."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def knn_score_no_faiss(
    bank_feas: np.ndarray, query_feas: np.ndarray, k: int = 100, min_score: bool = False
) -> np.ndarray:
    """
    Compute KNN scores without FAISS using numpy operations.

    Args:
        bank_feas: Bank features [n_bank, feature_dim].
        query_feas: Query features [n_query, feature_dim].
        k: Number of nearest neighbors.
        min_score: If True, use min distance; else use mean distance.

    Returns:
        KNN scores [n_query].
    """
    bank_feas = deepcopy(np.array(bank_feas))
    query_feas = deepcopy(np.array(query_feas))

    # Compute inner product (cosine similarity for normalized vectors)
    similarities = query_feas @ bank_feas.T  # [n_query, n_bank]

    # Get top-k similarities (k largest values)
    if k >= bank_feas.shape[0]:
        # If k >= number of bank features, use all
        top_k_sims = similarities
    else:
        # Use partition for efficiency (faster than full sort)
        top_k_indices = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
        batch_indices = np.arange(similarities.shape[0])[:, None]
        top_k_sims = similarities[batch_indices, top_k_indices]

    if min_score:
        scores = np.min(top_k_sims, axis=1)
    else:
        scores = np.mean(top_k_sims, axis=1)

    return scores


class NNGuideTextPostprocessor:
    """
    NNGuide Postprocessor for text embeddings without FAISS.

    Uses nearest neighbor guidance for OOD detection:
    - Creates confidence-weighted bank features
    - Combines KNN similarity with energy scores
    - Efficient numpy implementation without FAISS
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_nnguide",
        K: int = 100,  # Number of nearest neighbors
        alpha: float = 1.0,  # Fraction of training data to use for bank
        min_score: bool = False,  # Use min vs mean for KNN score
    ):
        """
        Initialize NNGuide Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            K: Number of nearest neighbors for similarity computation.
            alpha: Fraction of training data to use for bank features.
            min_score: If True, use min similarity; else use mean similarity.
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.K = K
        self.alpha = alpha
        self.min_score = min_score

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

        # NNGuide components (following reference pattern)
        self.bank_guide: np.ndarray | None = None  # Confidence-weighted bank features

        # Binary classifier for creating confidence scores
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
            normalize_embeddings=False,  # We'll normalize manually following reference
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
        """Load background data to create binary classification for confidence scores."""
        print("Loading background data for NNGuide binary classifier...")

        # Dummy background data for demonstration
        background_texts = [
            "The quick brown fox jumps over the lazy dog while exploring nature trails.",
            "Advanced machine learning algorithms are transforming modern data science applications.",
            "Climate change impacts global weather patterns and environmental sustainability efforts worldwide.",
            "Neuroscience research reveals complex interactions between brain regions and cognitive functions.",
            "Renewable energy technologies like solar panels are becoming increasingly cost-effective solutions.",
            "International trade agreements influence economic policies and market dynamics across nations.",
            "Educational technology platforms enhance remote learning experiences for students globally.",
            "Space exploration missions provide valuable insights into planetary science and astronomy.",
            "Artificial intelligence systems are being integrated into healthcare diagnostic procedures.",
            "Cultural heritage preservation efforts protect historical artifacts and traditional knowledge systems.",
        ] * 50  # Repeat to get more samples

        return background_texts

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the NNGuide postprocessor following the exact reference pattern.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed.
        """
        if self.setup_flag:
            print("NNGuide postprocessor already set up.")
            return

        print("\n Setup: Building guided bank features...")

        # Following reference pattern: use only alpha fraction of training data
        n_samples = int(len(id_texts) * self.alpha)
        selected_texts = list(id_texts)[:n_samples]  # Take first alpha fraction
        print(f"Using {n_samples} samples ({self.alpha:.1%} of {len(id_texts)}) for bank")

        # Extract features from selected training texts
        bank_feas = []
        bank_logits = []

        # Extract features
        features = self._extract_embeddings(selected_texts, name="id_train_bank")
        bank_feas.append(normalizer(features.cpu().numpy()))  # Normalize features

        # Create binary classifier to get confidence scores (logits)
        print("Training binary classifier for confidence scores...")
        background_texts = self._load_background_data()
        background_features = self._extract_embeddings(background_texts, name="background")

        # Train binary classifier
        X_train = torch.vstack([features, background_features]).cpu().numpy()
        y_train = np.hstack([np.ones(len(features)), np.zeros(len(background_features))])

        self.binary_classifier = LogisticRegression(
            random_state=random_state, max_iter=1000, fit_intercept=True
        )
        self.binary_classifier.fit(X_train, y_train)

        # Get logits (confidence scores) for ID training data
        id_logits = self.binary_classifier.decision_function(features.cpu().numpy())
        # Convert single logit to binary logits [background_logit, id_logit]
        binary_logits = np.column_stack([-id_logits, id_logits])
        bank_logits.append(binary_logits)

        # Following reference pattern exactly:
        # Concatenate all features and logits
        bank_feas = np.concatenate(bank_feas, axis=0)
        bank_logits = np.concatenate(bank_logits, axis=0)

        # Compute confidence scores using logsumexp
        bank_confs = logsumexp(bank_logits, axis=-1)

        # Create guided bank features: normalized_features * confidence_scores
        self.bank_guide = bank_feas * bank_confs[:, None]

        print(f"Bank features shape: {bank_feas.shape}")
        print(f"Bank confidence shape: {bank_confs.shape}")
        print(f"Guided bank features shape: {self.bank_guide.shape}")

        self.setup_flag = True
        print("NNGuide setup completed.")

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores following NNGuide pattern.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            NNGuide scores (higher = more in-distribution).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        assert self.binary_classifier is not None

        # Extract features
        features = self._extract_embeddings(texts, name=cache_name)

        # Normalize features (following reference pattern)
        feas_norm = normalizer(features.cpu().numpy())

        # Get energy scores using binary classifier
        features_np = features.cpu().numpy()
        id_logits = self.binary_classifier.decision_function(features_np)
        # Convert to binary logits
        binary_logits = np.column_stack([-id_logits, id_logits])
        energy = logsumexp(binary_logits, axis=-1)

        # Following NNGuide reference pattern exactly:
        # Compute KNN similarity scores
        conf = knn_score_no_faiss(self.bank_guide, feas_norm, k=self.K, min_score=self.min_score)

        # Combine confidence and energy scores
        score = conf * energy

        return score

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
        Evaluate the NNGuide postprocessor.

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
        self.alpha = hyperparam[1]

    def get_hyperparam(self) -> list:
        """Get hyperparameters (following reference API)."""
        return [self.K, self.alpha]


# Convenience function matching other postprocessors
def create_nnguide_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> NNGuideTextPostprocessor:
    """
    Create NNGuide text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for NNGuideTextPostprocessor.

    Returns:
        Configured NNGuide postprocessor.
    """
    return NNGuideTextPostprocessor(embedding_model=embedding_model, **kwargs)
