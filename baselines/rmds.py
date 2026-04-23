"""
Filename: rmds.py
Relative Mahalanobis Distance (RMD) OOD Detection for Conditional Language Models

This module implements RMD-based OOD detection using input and output embeddings
from conditional language models, following the API structure of ForteTextOODDetector.

The RMD approach:
1. Extracts input embeddings (z) by averaging encoder final-layer hidden states
2. Extracts output embeddings (w) by averaging decoder final-layer hidden states
3. Fits Gaussian distributions on training embeddings N(#z, #z) and N(#w, #w)
4. Fits background Gaussians N(#z0, #z0) and N(#w#, #w#) on broad datasets
5. Computes RMD scores: RMD = MD_domain - MD_background

Also includes binary classifier alternative using logistic regression on embeddings.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class RMDTextOODDetector:
    """
    Relative Mahalanobis Distance OOD Detector for conditional language models.

    Uses input and output embeddings from encoder-decoder models to compute
    RMD scores for OOD detection, maintaining the same API as ForteTextOODDetector.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        batch_size: int = 8,
        device: str | None = None,
        embedding_dir: str = "./embeddings_rmd",
        method: Literal[
            "rmd_input", "rmd_output", "rmd_combined", "binary_input", "binary_output"
        ] = "rmd_combined",
        background_data_path: str | None = None,
        max_length: int = 512,
    ):
        """
        Initialize the RMD OOD Detector.

        Args:
            model_name (str): HuggingFace model name for encoder-decoder model.
            batch_size (int): Batch size for processing.
            device (str | None): Device to use ('cuda:0', 'cpu', etc.).
            embedding_dir (str): Directory to store embeddings.
            method (str): Detection method ('rmd_input', 'rmd_output', 'rmd_combined', 'binary_input', 'binary_output').
            background_data_path (str | None): Path to background dataset for RMD.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.embedding_dir = embedding_dir
        self.method = method
        self.background_data_path = background_data_path

        # Model components
        self.model: Any = None
        self.tokenizer: Any = None

        # Training data statistics
        self.mu_z: torch.Tensor | None = None  # Input embedding mean
        self.sigma_z: torch.Tensor | None = None  # Input embedding covariance
        self.mu_w: torch.Tensor | None = None  # Output embedding mean
        self.sigma_w: torch.Tensor | None = None  # Output embedding covariance

        # Background data statistics
        self.mu_z0: torch.Tensor | None = None  # Background input mean
        self.sigma_z0: torch.Tensor | None = None  # Background input covariance
        self.mu_w_delta: torch.Tensor | None = None  # Background output mean
        self.sigma_w_delta: torch.Tensor | None = None  # Background output covariance

        # Binary classifier
        self.binary_classifier: LogisticRegression | None = None

        self.is_fitted = False
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _init_model(self) -> None:
        """Initialize the encoder-decoder model and tokenizer."""
        print(f"Initializing model {self.model_name} on {self.device}...")

        # Set HuggingFace to use cached models to avoid rate limits
        os.environ["HF_HUB_OFFLINE"] = "1"  # Use offline mode if model is cached
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, cache_dir=cache_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _extract_embeddings(
        self,
        input_texts: Sequence[str],
        output_texts: Sequence[str] | None = None,
        name: str = "tmp",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract input and output embeddings from texts.

        Args:
            input_texts: Input text sequences.
            output_texts: Output text sequences (optional, will use input if None).
            name: Cache name for embeddings.

        Returns:
            Tuple of (input_embeddings, output_embeddings) tensors.
        """
        if self.model is None:
            self._init_model()

        if output_texts is None:
            output_texts = input_texts

        # Check for cached embeddings
        input_cache_path = os.path.join(self.embedding_dir, f"{name}_input_embeddings.pt")
        output_cache_path = os.path.join(self.embedding_dir, f"{name}_output_embeddings.pt")

        if os.path.exists(input_cache_path) and os.path.exists(output_cache_path):
            print(f"Loading cached embeddings from {input_cache_path} and {output_cache_path}")
            input_embs = torch.load(input_cache_path, map_location=self.device)
            output_embs = torch.load(output_cache_path, map_location=self.device)
            if input_embs.size(0) == len(input_texts) and output_embs.size(0) == len(output_texts):
                return input_embs, output_embs
            else:
                print("Cached embeddings size mismatch, recomputing...")

        print(
            f"Extracting embeddings for {len(input_texts)} input texts and {len(output_texts)} output texts..."
        )

        input_embeddings = []
        output_embeddings = []

        for i in tqdm(range(0, len(input_texts), self.batch_size), desc="Extracting embeddings"):
            batch_inputs = list(input_texts[i : i + self.batch_size])
            batch_outputs = list(output_texts[i : i + self.batch_size])

            # Tokenize inputs
            input_tokens = self.tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Tokenize outputs
            output_tokens = self.tokenizer(
                batch_outputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                # Get encoder outputs (input embeddings)
                encoder_outputs = self.model.encoder(**input_tokens)
                input_hidden = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

                # Average over sequence length to get z (input embedding)
                input_mask = input_tokens.attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
                z_batch = (input_hidden * input_mask).sum(dim=1) / input_mask.sum(
                    dim=1
                )  # [batch, hidden_dim]
                input_embeddings.append(z_batch)

                # Get decoder outputs (output embeddings)
                # For output embeddings, we need to run decoder
                decoder_outputs = self.model.decoder(
                    input_ids=output_tokens.input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=input_tokens.attention_mask,
                )
                output_hidden = decoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

                # Average over sequence length to get w (output embedding)
                output_mask = output_tokens.attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
                w_batch = (output_hidden * output_mask).sum(dim=1) / output_mask.sum(
                    dim=1
                )  # [batch, hidden_dim]
                output_embeddings.append(w_batch)

        # Concatenate all batches
        input_embs = torch.cat(input_embeddings, dim=0)  # [num_texts, hidden_dim]
        output_embs = torch.cat(output_embeddings, dim=0)  # [num_texts, hidden_dim]

        # Cache embeddings
        torch.save(input_embs, input_cache_path)
        torch.save(output_embs, output_cache_path)
        print(f"Cached input embeddings shape {input_embs.shape} to {input_cache_path}")
        print(f"Cached output embeddings shape {output_embs.shape} to {output_cache_path}")

        return input_embs, output_embs

    def _fit_gaussian(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fit a Gaussian distribution to embeddings.

        Args:
            embeddings: Embeddings tensor [num_samples, embedding_dim].

        Returns:
            Tuple of (mean, covariance_matrix).
        """
        mu = embeddings.mean(dim=0)  # [embedding_dim]
        centered = embeddings - mu  # [num_samples, embedding_dim]
        sigma = torch.cov(centered.T)  # [embedding_dim, embedding_dim]

        # Add regularization for numerical stability
        reg_term = 1e-6 * torch.eye(sigma.size(0), device=sigma.device)
        sigma = sigma + reg_term

        return mu, sigma

    def _mahalanobis_distance(
        self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distance: (x - #)@ #{# (x - #)

        Args:
            x: Input tensor [num_samples, embedding_dim].
            mu: Mean vector [embedding_dim].
            sigma: Covariance matrix [embedding_dim, embedding_dim].

        Returns:
            Mahalanobis distances [num_samples].
        """
        diff = x - mu  # [num_samples, embedding_dim]

        # Compute #{# (x - #) efficiently
        try:
            sigma_inv = torch.linalg.inv(sigma)
            md = torch.sum(diff * (diff @ sigma_inv), dim=1)
        except torch.linalg.LinAlgError:
            # Fall back to pseudoinverse if singular
            sigma_pinv = torch.linalg.pinv(sigma)
            md = torch.sum(diff * (diff @ sigma_pinv), dim=1)

        return md

    def _load_background_data(self) -> tuple[list[str], list[str]]:
        """
        Load background data for RMD computation.

        Returns:
            Tuple of (background_inputs, background_outputs).
        """
        # For now, return dummy background data
        # In practice, this should load from C4, ParaCrawl, etc.
        print("Loading background data...")

        # Dummy background data for demonstration
        background_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Climate change is one of the most pressing issues of our time.",
            "The human brain contains approximately 86 billion neurons.",
            "Photosynthesis is the process by which plants make their food.",
        ] * 100  # Repeat to get more samples

        return background_texts, background_texts

    def fit(
        self,
        id_input_texts: Sequence[str],
        id_output_texts: Sequence[str] | None = None,
        val_split: float = 0.2,
        random_state: int = 42,
    ) -> RMDTextOODDetector:
        """
        Fit the RMD detector on in-distribution text pairs.

        Args:
            id_input_texts: In-distribution input texts.
            id_output_texts: In-distribution output texts (optional).
            val_split: Validation split ratio.
            random_state: Random seed.

        Returns:
            self: The fitted detector.
        """
        start_time = time.time()
        print(f"Fitting RMD detector on {len(id_input_texts)} text pairs...")

        if id_output_texts is None:
            id_output_texts = id_input_texts

        # Split into train/val
        train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
            list(id_input_texts),
            list(id_output_texts),
            test_size=val_split,
            random_state=random_state,
        )

        # Extract embeddings
        print("Extracting training embeddings...")
        train_z, train_w = self._extract_embeddings(train_inputs, train_outputs, name="id_train")

        # Fit Gaussian distributions on training data
        print("Fitting Gaussian distributions on training data...")
        self.mu_z, self.sigma_z = self._fit_gaussian(train_z)
        self.mu_w, self.sigma_w = self._fit_gaussian(train_w)

        # Load and process background data for RMD
        if self.method.startswith("rmd"):
            print("Processing background data for RMD...")
            bg_inputs, bg_outputs = self._load_background_data()
            bg_z, bg_w = self._extract_embeddings(bg_inputs, bg_outputs, name="background")

            # Fit background Gaussian distributions
            self.mu_z0, self.sigma_z0 = self._fit_gaussian(bg_z)
            self.mu_w_delta, self.sigma_w_delta = self._fit_gaussian(bg_w)

        # Train binary classifier if needed
        if self.method.startswith("binary"):
            print("Training binary classifier...")
            bg_inputs, bg_outputs = self._load_background_data()
            bg_z, bg_w = self._extract_embeddings(bg_inputs, bg_outputs, name="background")

            if self.method == "binary_input":
                # Use input embeddings
                X_pos = train_z.cpu().numpy()
                X_neg = bg_z.cpu().numpy()
            else:  # binary_output
                # Use output embeddings
                X_pos = train_w.cpu().numpy()
                X_neg = bg_w.cpu().numpy()

            X = np.vstack([X_pos, X_neg])
            y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

            self.binary_classifier = LogisticRegression(random_state=random_state)
            self.binary_classifier.fit(X, y)

        self.is_fitted = True
        fit_time = time.time() - start_time
        print(f"RMD detector fitted in {fit_time:.2f} seconds.")
        return self

    def _get_ood_scores(
        self,
        input_texts: Sequence[str],
        output_texts: Sequence[str] | None = None,
        cache_name: str = "test",
    ) -> np.ndarray:
        """
        Get OOD scores for text pairs.

        Args:
            input_texts: Input text sequences.
            output_texts: Output text sequences (optional).
            cache_name: Cache identifier.

        Returns:
            Array of OOD scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        if output_texts is None:
            output_texts = input_texts

        # Extract test embeddings
        test_z, test_w = self._extract_embeddings(input_texts, output_texts, name=cache_name)

        if self.method == "rmd_input":
            # Input RMD only
            md_input = self._mahalanobis_distance(test_z, self.mu_z, self.sigma_z)
            md0_input = self._mahalanobis_distance(test_z, self.mu_z0, self.sigma_z0)
            scores = md_input - md0_input

        elif self.method == "rmd_output":
            # Output RMD only
            md_output = self._mahalanobis_distance(test_w, self.mu_w, self.sigma_w)
            md_delta_output = self._mahalanobis_distance(
                test_w, self.mu_w_delta, self.sigma_w_delta
            )
            scores = md_output - md_delta_output

        elif self.method == "rmd_combined":
            # Combined input + output RMD
            md_input = self._mahalanobis_distance(test_z, self.mu_z, self.sigma_z)
            md0_input = self._mahalanobis_distance(test_z, self.mu_z0, self.sigma_z0)
            rmd_input = md_input - md0_input

            md_output = self._mahalanobis_distance(test_w, self.mu_w, self.sigma_w)
            md_delta_output = self._mahalanobis_distance(
                test_w, self.mu_w_delta, self.sigma_w_delta
            )
            rmd_output = md_output - md_delta_output

            # Combine scores (simple average)
            scores = (rmd_input + rmd_output) / 2.0

        elif self.method == "binary_input":
            # Binary classifier on input embeddings
            X_test = test_z.cpu().numpy()
            # Use logits (decision function) as OOD score
            assert self.binary_classifier is not None
            scores = torch.from_numpy(self.binary_classifier.decision_function(X_test))

        elif self.method == "binary_output":
            # Binary classifier on output embeddings
            X_test = test_w.cpu().numpy()
            assert self.binary_classifier is not None
            scores = torch.from_numpy(self.binary_classifier.decision_function(X_test))

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # Convert to numpy and flip sign for RMD methods (negative score = OOD)
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        if self.method.startswith("rmd"):
            scores = -scores  # Flip sign so higher score = more in-distribution

        return scores

    def predict(
        self, input_texts: Sequence[str], output_texts: Sequence[str] | None = None
    ) -> np.ndarray:
        """
        Predict OOD status for text pairs.

        Args:
            input_texts: Input text sequences.
            output_texts: Output text sequences (optional).

        Returns:
            Binary predictions (1 for in-distribution, -1 for OOD).
        """
        scores = self._get_ood_scores(input_texts, output_texts)

        # Use median as threshold
        threshold = float(np.median(scores))
        return np.where(scores > threshold, 1, -1)

    def predict_proba(
        self, input_texts: Sequence[str], output_texts: Sequence[str] | None = None
    ) -> np.ndarray:
        """
        Return probability scores for OOD detection.

        Args:
            input_texts: Input text sequences.
            output_texts: Output text sequences (optional).

        Returns:
            Normalized probability scores.
        """
        scores = self._get_ood_scores(input_texts, output_texts)

        # Normalize to [0, 1]
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5

        return normalized_scores

    def evaluate(
        self,
        id_input_texts: Sequence[str],
        ood_input_texts: Sequence[str],
        id_output_texts: Sequence[str] | None = None,
        ood_output_texts: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate the RMD detector.

        Args:
            id_input_texts: In-distribution input texts.
            ood_input_texts: OOD input texts.
            id_output_texts: In-distribution output texts (optional).
            ood_output_texts: OOD output texts (optional).

        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before evaluation")

        print(
            f"Evaluating on {len(id_input_texts)} ID and {len(ood_input_texts)} OOD text pairs..."
        )

        # Get scores
        id_scores = self._get_ood_scores(id_input_texts, id_output_texts, cache_name="eval_id")
        ood_scores = self._get_ood_scores(ood_input_texts, ood_output_texts, cache_name="eval_ood")

        print("\nScore Statistics:")
        print(f"ID  - Mean: {np.mean(id_scores):.4f}, Std: {np.std(id_scores):.4f}")
        print(f"OOD - Mean: {np.mean(ood_scores):.4f}, Std: {np.std(ood_scores):.4f}")

        # Compute metrics
        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        scores_all = np.concatenate([id_scores, ood_scores])

        auroc = float(roc_auc_score(labels, scores_all))
        fpr, tpr, _ = roc_curve(labels, scores_all)
        idx = int(np.argmin(np.abs(tpr - 0.95)))
        fpr95 = float(fpr[idx]) if idx < len(fpr) else 1.0

        precision_vals, recall_vals, _ = precision_recall_curve(labels, scores_all)
        auprc = float(average_precision_score(labels, scores_all))

        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        f1_score = float(np.max(f1_scores))

        return {"AUROC": auroc, "FPR@95TPR": fpr95, "AUPRC": auprc, "F1": f1_score}
