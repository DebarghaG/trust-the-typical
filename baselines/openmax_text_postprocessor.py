"""
Filename: openmax_text_postprocessor.py
OpenMax Postprocessor for Text Embeddings

This module implements OpenMax for text OOD detection:
"Towards Open Set Deep Networks"

OpenMax approach:
1. Computes Mean Activation Vectors (MAVs) for each class from training data
2. Fits Weibull distributions on distances from MAVs using Extreme Value Theory
3. Re-calibrates softmax scores using Weibull probabilities
4. Adds an "unknown" class for OOD detection
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import scipy.spatial.distance as spd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics

# Try to import libmr, fallback if not available
try:
    import libmr

    LIBMR_AVAILABLE = True
except ImportError:
    print("Warning: libmr not available. Using simplified Weibull approximation.")
    LIBMR_AVAILABLE = False

    # Simple Weibull approximation class
    class SimpleWeibullModel:
        def __init__(self) -> None:
            self.scale = 1.0
            self.shape = 1.0

        def fit_high(self, data: np.ndarray, tail_size: int) -> None:
            """Fit Weibull to tail of data."""
            if len(data) > 0:
                self.scale = float(np.mean(data))
                self.shape = 2.0  # Fixed shape parameter

        def w_score(self, distance: float) -> float:
            """Compute Weibull score (probability of being an outlier)."""
            if self.scale <= 0:
                return 0.0
            # Simple exponential decay approximation
            return max(0.0, min(1.0, 1.0 - np.exp(-distance / self.scale)))


def compute_channel_distances(
    mavs: np.ndarray, features: np.ndarray, eu_weight: float = 0.5
) -> dict[str, np.ndarray]:
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV
        for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append(
            [
                spd.euclidean(mcv, feat[channel]) * eu_weight + spd.cosine(mcv, feat[channel])
                for feat in features
            ]
        )

    return {
        "eucos": np.array(eucos_dists),
        "cosine": np.array(cos_dists),
        "euclidean": np.array(eu_dists),
    }


def fit_weibull(
    means: np.ndarray,
    dists: list,
    categories: list,
    tailsize: int = 20,
    distance_type: str = "eucos",
) -> dict:
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances
                        and save weibull model parameters for re-adjusting
                        softmax scores
    """
    weibull_model: dict[str, dict[str, typing.Any]] = {}
    for mean, dist, category_name in zip(means, dists, categories, strict=False):
        weibull_model[category_name] = {}
        weibull_model[category_name][f"distances_{distance_type}"] = dist[distance_type]
        weibull_model[category_name]["mean_vec"] = mean
        weibull_model[category_name]["weibull_model"] = []
        for channel in range(mean.shape[0]):
            if LIBMR_AVAILABLE:
                mr = libmr.MR()
            else:
                mr = SimpleWeibullModel()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]["weibull_model"].append(mr)

    return weibull_model


def compute_openmax_prob(scores: np.ndarray, scores_u: np.ndarray) -> list:
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u, strict=False):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def query_weibull(category_name: str, weibull_model: dict, distance_type: str = "eucos") -> list:
    return [
        weibull_model[category_name]["mean_vec"],
        weibull_model[category_name][f"distances_{distance_type}"],
        weibull_model[category_name]["weibull_model"],
    ]


def calc_distance(
    query_score: np.ndarray, mcv: np.ndarray, eu_weight: float, distance_type: str = "eucos"
) -> float:
    if distance_type == "eucos":
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + spd.cosine(mcv, query_score)
    elif distance_type == "euclidean":
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == "cosine":
        query_distance = spd.cosine(mcv, query_score)
    else:
        print(
            "distance type not known: enter either of eucos, \
               euclidean or cosine"
        )
    return query_distance


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(
    weibull_model: dict,
    categories: list,
    input_score: np.ndarray,
    eu_weight: float,
    alpha: int = 10,
    distance_type: str = "eucos",
) -> tuple[np.ndarray, np.ndarray]:
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    # Ensure alpha doesn't exceed number of classes
    alpha = min(alpha, nb_classes)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)

    # Only assign weights for the actual number of ranked classes
    if len(ranked_list) > 0:
        omega[ranked_list] = alpha_weights[: len(ranked_list)]

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(
                input_score_channel, mav[channel], eu_weight, distance_type
            )
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


class OpenMaxTextPostprocessor:
    """
    OpenMax Postprocessor for text embeddings.

    Uses Extreme Value Theory and Weibull distributions for OOD detection:
    - Computes Mean Activation Vectors (MAVs) for each class
    - Fits Weibull distributions on feature distances
    - Re-calibrates scores using Weibull probabilities
    - Adds unknown class for OOD detection
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_openmax",
        weibull_alpha: int = 3,  # Number of top classes to modify
        weibull_threshold: float = 0.9,  # Threshold for known vs unknown
        weibull_tail: int = 20,  # Tail size for Weibull fitting
        distance_type: str = "euclidean",  # Distance metric
        eu_weight: float = 0.5,  # Weight for euclidean in eucos distance
    ):
        """
        Initialize OpenMax Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            weibull_alpha: Number of top classes to modify with OpenMax.
            weibull_threshold: Threshold for classifying as known vs unknown.
            weibull_tail: Number of tail samples for Weibull fitting.
            distance_type: Distance metric ('euclidean', 'cosine', 'eucos').
            eu_weight: Weight for euclidean component in eucos distance.
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.weibull_alpha = weibull_alpha
        self.weibull_threshold = weibull_threshold
        self.weibull_tail = weibull_tail
        self.distance_type = distance_type
        self.eu_weight = eu_weight

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

        # OpenMax components
        self.weibull_model: dict | None = None
        self.nc: int = 2  # Number of classes (binary: ID vs Background)
        self.categories: list = [0, 1]  # Class categories

        # Binary classifier for creating class structure
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
            normalize_embeddings=False,  # OpenMax works with raw embeddings
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
        print("Loading background data for OpenMax binary classification...")

        # Dummy background data for demonstration
        background_texts = [
            "Weather patterns change throughout different seasons affecting global climate conditions significantly.",
            "Scientific research methodologies continue advancing with new technological innovations and discoveries.",
            "Economic systems worldwide face challenges from globalization and technological transformation.",
            "Educational approaches adapt to digital learning environments and modern pedagogical methods.",
            "Healthcare developments improve patient outcomes through innovative treatment and prevention strategies.",
            "Cultural traditions preserve historical heritage while adapting to contemporary social changes.",
            "Environmental conservation efforts address sustainability challenges through collaborative global initiatives.",
            "Transportation infrastructure evolves with smart technologies and sustainable mobility solutions.",
            "Communication networks enable instant information sharing across diverse geographical locations.",
            "Agricultural techniques modernize to enhance food security and sustainable farming practices.",
        ] * 50  # Repeat to get more samples

        return background_texts

    def compute_train_score_and_mavs_and_dists(
        self, id_texts: Sequence[str]
    ) -> tuple[list, np.ndarray, list]:
        """
        Compute training scores, MAVs, and distance distributions.

        Args:
            id_texts: In-distribution training texts.

        Returns:
            scores, mavs, dists for Weibull fitting.
        """
        print("Computing training scores and MAVs...")

        # Extract features for ID and background data
        id_features = self._extract_embeddings(id_texts, name="id_train_openmax")
        background_texts = self._load_background_data()
        bg_features = self._extract_embeddings(background_texts, name="background_openmax")

        # Train binary classifier
        X_train = torch.vstack([id_features, bg_features]).cpu().numpy()
        y_train = np.hstack([np.ones(len(id_features)), np.zeros(len(bg_features))])

        self.binary_classifier = LogisticRegression(
            random_state=42, max_iter=1000, fit_intercept=True
        )
        self.binary_classifier.fit(X_train, y_train)

        # Get predictions and separate by class
        id_predictions = self.binary_classifier.predict_proba(id_features.cpu().numpy())
        bg_predictions = self.binary_classifier.predict_proba(bg_features.cpu().numpy())

        # Create class-wise scores
        scores: list[list[np.ndarray]] = [[] for _ in range(self.nc)]

        # Store ID samples (class 1) that are correctly classified
        id_correct_mask = self.binary_classifier.predict(id_features.cpu().numpy()) == 1
        id_correct_features = id_features[id_correct_mask].cpu().numpy()
        id_correct_probs = id_predictions[id_correct_mask]

        # Store background samples (class 0) that are correctly classified
        bg_correct_mask = self.binary_classifier.predict(bg_features.cpu().numpy()) == 0
        bg_correct_features = bg_features[bg_correct_mask].cpu().numpy()
        bg_correct_probs = bg_predictions[bg_correct_mask]

        # Format as (N, 1, C) for compatibility with original OpenMax
        if len(id_correct_features) > 0:
            scores[1] = id_correct_probs[:, np.newaxis, :]  # ID class
        if len(bg_correct_features) > 0:
            scores[0] = bg_correct_probs[:, np.newaxis, :]  # Background class

        # Handle empty classes
        for i in range(self.nc):
            if len(scores[i]) == 0:
                # Create dummy scores for empty classes
                dummy_scores = np.random.rand(1, 1, 2) * 0.1
                dummy_scores[0, 0, i] = 0.9  # High confidence for own class
                scores[i] = dummy_scores

        # Compute MAVs (Mean Activation Vectors)
        mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)

        # Compute distance distributions
        dists = [
            compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores, strict=False)
        ]

        return scores, mavs, dists

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the OpenMax postprocessor.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed.
        """
        if self.setup_flag:
            print("OpenMax postprocessor already set up.")
            return

        print("\n Setup: Fitting Weibull distributions...")

        # Compute training statistics
        _, mavs, dists = self.compute_train_score_and_mavs_and_dists(id_texts)

        # Fit Weibull model
        self.weibull_model = fit_weibull(
            mavs, dists, self.categories, self.weibull_tail, self.distance_type
        )

        print(f"OpenMax setup completed with {len(self.categories)} classes")
        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores using OpenMax.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            OpenMax confidence scores (higher = more in-distribution).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        assert self.binary_classifier is not None
        assert self.weibull_model is not None

        # Extract features
        features = self._extract_embeddings(texts, name=cache_name)

        # Get classifier predictions
        features_np = features.cpu().numpy()
        scores = self.binary_classifier.predict_proba(features_np)
        scores = scores[:, np.newaxis, :]  # Add channel dimension for compatibility

        # Apply OpenMax to each sample
        openmax_scores = []
        for score in scores:
            openmax_prob, _ = openmax(
                self.weibull_model,
                self.categories,
                score,
                self.eu_weight,
                self.weibull_alpha,
                self.distance_type,
            )

            # Use the maximum known class probability as confidence
            # openmax_prob has shape [class_0, class_1, unknown]
            known_class_prob = max(openmax_prob[:-1])  # Exclude unknown class
            openmax_scores.append(known_class_prob)

        return np.array(openmax_scores)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """
        Predict OOD status using OpenMax threshold.

        Args:
            texts: Input texts.

        Returns:
            Binary predictions (1 for in-distribution, -1 for OOD).
        """
        scores = self.postprocess(texts)
        return np.where(scores > self.weibull_threshold, 1, -1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return OpenMax probability scores.

        Args:
            texts: Input texts.

        Returns:
            OpenMax probability scores.
        """
        return self.postprocess(texts)

    def evaluate(self, id_texts: Sequence[str], ood_texts: Sequence[str]) -> dict[str, float]:
        """
        Evaluate the OpenMax postprocessor.

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
        if len(hyperparam) >= 3:
            self.weibull_alpha = hyperparam[0]
            self.weibull_threshold = hyperparam[1]
            self.weibull_tail = hyperparam[2]

    def get_hyperparam(self) -> list:
        """Get hyperparameters (following reference API)."""
        return [self.weibull_alpha, self.weibull_threshold, self.weibull_tail]


# Convenience function matching other postprocessors
def create_openmax_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> OpenMaxTextPostprocessor:
    """
    Create OpenMax text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for OpenMaxTextPostprocessor.

    Returns:
        Configured OpenMax postprocessor.
    """
    return OpenMaxTextPostprocessor(embedding_model=embedding_model, **kwargs)
