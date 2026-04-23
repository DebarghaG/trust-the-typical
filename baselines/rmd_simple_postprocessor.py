"""
Filename: rmd_simple_postprocessor.py
Simple RMD Postprocessor for Text Embeddings following the exact pattern from reference code

This implementation follows the exact pattern from your reference RMDSPostprocessor:
- Uses whole training set as "background" (no external background data needed)
- Computes class-conditional statistics and whole-dataset statistics
- RMD = class_score - background_score
- Works with any text embedding model (Qwen3-Embedding-0.6B, BGE-M3, E5, etc.)
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import sklearn.covariance
import torch
from sentence_transformers import SentenceTransformer

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class SimpleRMDTextPostprocessor:
    """
    Simple RMD Postprocessor for text embeddings that exactly follows the reference pattern.

    Uses the whole training set as background, no external background data required.
    For text OOD detection, treats in-domain as one "class" and uses whole dataset statistics.
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_simple_rmd",
    ):
        """
        Initialize Simple RMD Text Postprocessor.

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

        # Statistics (following reference pattern exactly)
        self.class_mean: torch.Tensor | None = None  # In-domain class mean
        self.precision: torch.Tensor | None = None  # Class precision matrix
        self.whole_mean: torch.Tensor | None = None  # Whole dataset mean
        self.whole_precision: torch.Tensor | None = None  # Whole dataset precision

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

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the RMD postprocessor following the exact reference pattern.

        Args:
            id_texts: In-distribution training texts.
            random_state: Random seed (for compatibility).
        """
        if self.setup_flag:
            print("Simple RMD postprocessor already set up.")
            return

        print("\n Estimating mean and variance from training set...")

        # Extract all features from training texts
        all_feats = self._extract_embeddings(id_texts, name="id_train")

        # Following the reference pattern more closely for multi-class adaptation to binary OOD:
        # The reference uses class-conditional vs. whole-dataset statistics

        # 1. Compute class-conditional statistics (in-domain class)
        # Use only a subset for class statistics to create difference from whole
        n_samples = all_feats.size(0)
        class_indices = torch.randperm(n_samples)[: max(n_samples // 2, 1)]  # Use half for class
        class_feats = all_feats[class_indices]

        self.class_mean = class_feats.mean(0)  # [feature_dim]

        # Center data for class covariance
        centered_class_data = class_feats - self.class_mean.view(1, -1)

        # Use sklearn covariance estimation (exact same as reference)
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso.fit(centered_class_data.cpu().numpy().astype(np.float32))
        # inverse of covariance (precision matrix)
        self.precision = torch.from_numpy(group_lasso.precision_).float()

        # 2. Compute whole dataset statistics (using all training data as "background")
        self.whole_mean = all_feats.mean(0)  # Use all data for background
        centered_data_whole = all_feats - self.whole_mean.view(1, -1)

        group_lasso_whole = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso_whole.fit(centered_data_whole.cpu().numpy().astype(np.float32))
        self.whole_precision = torch.from_numpy(group_lasso_whole.precision_).float()

        print(f"Class mean shape: {self.class_mean.shape}")
        print(f"Precision matrix shape: {self.precision.shape}")
        print(f"Whole mean shape: {self.whole_mean.shape}")
        print(f"Whole precision shape: {self.whole_precision.shape}")

        self.setup_flag = True
        print("Simple RMD postprocessor setup completed.")

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores following the exact reference pattern.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            OOD confidence scores (higher = more in-distribution).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        assert self.whole_mean is not None
        assert self.whole_precision is not None
        assert self.class_mean is not None
        assert self.precision is not None

        # Extract features
        features = self._extract_embeddings(texts, name=cache_name)

        # Following the reference pattern exactly:

        # Move everything to CPU for computation (matching reference pattern)
        # Ensure consistent dtype (float32) for all tensors
        features_cpu = features.cpu().float()
        whole_mean_cpu = self.whole_mean.cpu().float()
        whole_precision_cpu = self.whole_precision.cpu().float()
        class_mean_cpu = self.class_mean.cpu().float()
        precision_cpu = self.precision.cpu().float()

        # 1. Compute background scores using whole dataset statistics
        tensor1 = features_cpu - whole_mean_cpu.view(1, -1)
        background_scores = -torch.matmul(
            torch.matmul(tensor1, whole_precision_cpu), tensor1.t()
        ).diag()

        # 2. Compute class scores (in-domain class)
        tensor = features_cpu - class_mean_cpu.view(1, -1)
        class_scores = -torch.matmul(torch.matmul(tensor, precision_cpu), tensor.t()).diag()

        # 3. RMD = class_score - background_score (following reference exactly)
        conf = class_scores - background_scores

        return conf.numpy()

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
        Evaluate the Simple RMD postprocessor.

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

    def compute_similarity(self, texts1: Sequence[str], texts2: Sequence[str]) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of texts.

        Args:
            texts1: First set of texts.
            texts2: Second set of texts.

        Returns:
            Similarity matrix of shape (len(texts1), len(texts2)).
        """
        if self.embedding_model is None:
            self._init_embedding_model()

        assert self.embedding_model is not None

        emb1 = self.embedding_model.encode(
            list(texts1),
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).to(self.device)
        emb2 = self.embedding_model.encode(
            list(texts2),
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).to(self.device)

        # Cosine similarity = dot product for normalized embeddings
        sim = emb1 @ emb2.T
        return sim


# Convenience function matching Forte API style
def create_simple_rmd_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> SimpleRMDTextPostprocessor:
    """
    Create Simple RMD text detector with Forte-style API.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for SimpleRMDTextPostprocessor.

    Returns:
        Configured Simple RMD postprocessor.
    """
    return SimpleRMDTextPostprocessor(embedding_model=embedding_model, **kwargs)
