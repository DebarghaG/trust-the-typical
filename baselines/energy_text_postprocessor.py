"""
Filename: energy_text_postprocessor.py
Energy-based OOD Detection for Text Embeddings

This module implements Energy-based OOD detection for text:
"Energy-based Out-of-distribution Detection" (Liu et al., NeurIPS 2020)

Energy approach:
1. Extracts embeddings from text using pre-trained embedding model
2. Treats embeddings as logits and computes energy score: E(x) = -T * log(sum(exp(f(x)/T)))
3. Uses energy as OOD confidence score (higher energy = more ID)
4. Simple yet effective baseline for OOD detection
"""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import numpy as np
import torch
from scipy.special import logsumexp
from sentence_transformers import SentenceTransformer

from .evaluation_utils import evaluate_binary_classifier, print_score_statistics


class EnergyBasedTextPostprocessor:
    """
    Energy-based Postprocessor for text embeddings.

    Uses energy score from embeddings for OOD detection:
    - Extracts embeddings from pre-trained model
    - Computes energy: -T * logsumexp(embeddings/T)
    - Higher energy indicates more in-distribution
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_energy",
        temperature: float = 1.0,  # Temperature parameter T
    ):
        """
        Initialize Energy-based Text Postprocessor.

        Args:
            embedding_model: HuggingFace model name for text embeddings.
            batch_size: Batch size for processing.
            device: Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir: Directory to store embeddings.
            temperature: Temperature parameter T for energy computation.
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.temperature = temperature

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
            normalize_embeddings=False,  # Keep raw embeddings for energy computation
            device=self.device,
        )

        torch.save(embeddings, cache_path)
        print(f"Cached embeddings shape {embeddings.shape} to {cache_path}")

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return embeddings.to(self.device)

    def setup(self, id_texts: Sequence[str], random_state: int = 42) -> None:
        """
        Setup the Energy-based postprocessor.

        Args:
            id_texts: In-distribution training texts (used to initialize embedding model).
            random_state: Random seed (for reproducibility).
        """
        if self.setup_flag:
            print("Energy-based postprocessor already set up.")
            return

        print("\n Setup: Energy-based method requires no training, only embedding extraction...")

        # Initialize embedding model by extracting features once
        _ = self._extract_embeddings(id_texts[: min(10, len(id_texts))], name="id_init")

        print("Energy-based setup completed (no training required).")
        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Postprocess texts to get OOD scores using energy.

        Args:
            texts: Input texts to score.
            cache_name: Cache identifier.

        Returns:
            Energy scores (higher = more in-distribution).
        """
        if not self.setup_flag:
            raise RuntimeError("Must call setup() before postprocess()")

        # Extract embeddings (treated as logits)
        embeddings = self._extract_embeddings(texts, name=cache_name)
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        # Compute energy score: -T * logsumexp(embeddings/T)
        # Note: logsumexp is numerically stable way to compute log(sum(exp(x)))
        energy_scores = -self.temperature * logsumexp(embeddings_np / self.temperature, axis=-1)

        return energy_scores

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
        Evaluate the Energy-based postprocessor.

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
        self.temperature = hyperparam[0]

    def get_hyperparam(self) -> float:
        """Get hyperparameters (following reference API)."""
        return self.temperature


def create_energy_text_detector(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs: typing.Any
) -> EnergyBasedTextPostprocessor:
    """
    Create Energy-based text detector.

    Args:
        embedding_model: Text embedding model to use.
        **kwargs: Additional arguments for EnergyBasedTextPostprocessor.

    Returns:
        Configured Energy-based postprocessor.
    """
    return EnergyBasedTextPostprocessor(embedding_model=embedding_model, **kwargs)
