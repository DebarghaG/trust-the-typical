"""
Filename: forte_text_api.py
Standardized Forte OOD Detection API for Text Embeddings

This module aligns the text OOD API with forte_api.ForteOODDetector:
- Matching public method signatures and flow (fit, predict, predict_proba, evaluate)
- Support for multiple embedding models, concatenating PRDC features across models
- GPU-accelerated custom detectors (TorchGMM, TorchKDE, TorchOCSVM) when available
- CPU fallbacks using scikit-learn / SciPy mirroring forte_api.py

Default text models (Option B - diversity):
  - ('qwen3', 'Qwen/Qwen3-Embedding-0.6B')
  - ('bge-m3', 'BAAI/bge-m3')
  - ('e5',    'intfloat/e5-large-v2')
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from tqdm import tqdm

# Import custom GPU-based detectors from forte_api to keep flow identical
from .forte_api import TorchGMM, TorchKDE, TorchOCSVM


class ForteTextOODDetector:
    """
    Forte OOD Detector adapted for text embeddings with standardized API/flow
    matching ForteOODDetector (vision).
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str | None = None,
        embedding_dir: str = "./embeddings",
        nearest_k: int = 5,
        method: Literal["gmm", "kde", "ocsvm"] = "gmm",
        model_names: list[tuple[str, str]] | None = None,
        use_flash_attention: bool = True,
    ):
        """
        Initialize the ForteTextOODDetector.

        Args:
            batch_size (int): Batch size for processing texts.
            device (str | None): Device to use for computation (e.g., 'cuda:0', 'mps', or 'cpu').
            embedding_dir (str): Directory to store embeddings.
            nearest_k (int): Number of nearest neighbors for PRDC computation.
            method (str): Detector method ('gmm', 'kde', or 'ocsvm').
            model_names (list[tuple[str,str]] | None): List of (short_name, model_id) for SentenceTransformer models.
            use_flash_attention (bool): Best-effort attempt to use FP16/memory optimizations when available.
        """
        self.batch_size = batch_size
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device: str = device
        self.embedding_dir: str = embedding_dir
        self.nearest_k: int = nearest_k
        self.method: Literal["gmm", "kde", "ocsvm"] = method
        self.use_flash_attention: bool = use_flash_attention

        # Default to diverse set of models (Option B)
        if model_names is None:
            model_names = [
                ("qwen3", "Qwen/Qwen3-Embedding-0.6B"),
                ("bge-m3", "BAAI/bge-m3"),
                ("e5", "intfloat/e5-large-v2"),
            ]
        self.model_specs: list[tuple[str, str]] = model_names  # list of (short_name, model_id)
        # will be list of (short_name, st_model, None) to mirror vision API structure
        self.models: list[tuple[str, SentenceTransformer, None]] | None = None

        # Mirror forte_api: GPU => custom detectors, CPU => sklearn/scipy
        self.custom_detector: bool = self.device != "cpu"
        self.is_fitted: bool = False

        # Set during fit
        self.id_train_features: dict[str, torch.Tensor] | None = None  # per-model features
        self.id_train_prdc: torch.Tensor | None = None  # concatenated PRDC features (on device)
        self.detector: Any | None = None

        os.makedirs(self.embedding_dir, exist_ok=True)

    def _init_models(self) -> list[tuple[str, SentenceTransformer, None]]:
        """Initialize the text embedding models used for feature extraction."""
        print(f"Initializing text embedding models on {self.device}...")
        models: list[tuple[str, SentenceTransformer, None]] = []
        for short_name, model_id in self.model_specs:
            try:
                if self.use_flash_attention and self.device != "cpu":
                    # Best-effort FP16 init; fall back if not supported
                    try:
                        st = SentenceTransformer(
                            model_id,
                            device=self.device,
                            model_kwargs={"torch_dtype": torch.float16},
                        )
                        print(f"  - {short_name}: initialized with FP16 (best-effort)")
                    except Exception as e:
                        print(
                            f"  - {short_name}: FP16 init failed ({e}); falling back to default precision"
                        )
                        st = SentenceTransformer(model_id, device=self.device)
                else:
                    st = SentenceTransformer(model_id, device=self.device)
                models.append((short_name, st, None))
            except Exception as e:
                print(f"Error initializing model {model_id} as {short_name}: {e}")
        return models

    def _encode_batch(self, st_model: SentenceTransformer, texts: Sequence[str]) -> torch.Tensor:
        """
        Encode a batch of texts with a SentenceTransformer model.

        Returns:
            torch.Tensor on self.device
        """
        embeddings = st_model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        # Ensure on the target device
        return embeddings.to(self.device)

    def _extract_features_batch(
        self, texts: Sequence[str], batch_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        """
        Extract features for a batch of texts using multiple models.

        Args:
            texts (Sequence[str]): Text strings.
            batch_idx (int): Batch index for progress tracking.

        Returns:
            dict[str, torch.Tensor]: Features per model on device.
        """
        if self.models is None:
            self.models = self._init_models()

        assert self.models is not None
        all_features: dict[str, torch.Tensor] = {}
        for model_name, st_model, _ in self.models:
            try:
                feats = self._encode_batch(st_model, texts)
                all_features[model_name] = feats
            except Exception as e:
                print(f"Error extracting embeddings with {model_name}: {e}")
                all_features[model_name] = torch.empty(0, device=self.device)
        return all_features

    def _extract_features(self, texts: Sequence[str], name: str = "tmp") -> dict[str, torch.Tensor]:
        """
        Extract features from all texts using the models, with per-model caching.

        Args:
            texts (Sequence[str]): Text inputs.
            name (str): Identifier for caching.

        Returns:
            dict[str, torch.Tensor]: Dictionary of features for each model on device.
        """
        if self.models is None:
            self.models = self._init_models()
        assert self.models is not None

        all_features: dict[str, torch.Tensor] = {}
        models_to_process: list[str] = []
        batch_accum: dict[str, list[torch.Tensor]] = {}

        for model_name, _, _ in self.models:
            embedding_file = os.path.join(self.embedding_dir, f"{name}_{model_name}_features.pt")
            if os.path.exists(embedding_file):
                print(f"Loading pre-computed features from {embedding_file}")
                loaded = torch.load(embedding_file, map_location=self.device)
                if loaded.size(0) != len(texts):
                    print(
                        f"Warning: Cached features count ({loaded.size(0)}) doesn't match text count ({len(texts)}). Recomputing for {model_name}."
                    )
                    models_to_process.append(model_name)
                    batch_accum[model_name] = []
                else:
                    print(f"Feature shape for {model_name}: {loaded.shape}")
                    all_features[model_name] = loaded
            else:
                models_to_process.append(model_name)
                batch_accum[model_name] = []

        if not models_to_process:
            return all_features

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting features"):
            batch_texts = list(texts[i : i + self.batch_size])
            batch_features = self._extract_features_batch(batch_texts, i // self.batch_size)
            for model_name, features in batch_features.items():
                if features.numel() > 0 and model_name in models_to_process:
                    batch_accum[model_name].append(features)

        for model_name in models_to_process:
            if batch_accum[model_name]:
                all_features[model_name] = torch.cat(batch_accum[model_name], dim=0)
                embedding_file = os.path.join(
                    self.embedding_dir, f"{name}_{model_name}_features.pt"
                )
                # torch.save(all_features[model_name], embedding_file)
                print(
                    f"Saved {model_name} features with shape {all_features[model_name].shape} to {embedding_file}"
                )
            else:
                all_features[model_name] = torch.empty(0, device=self.device)

        return all_features

    def _compute_pairwise_distance(
        self, data_x: torch.Tensor, data_y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute pairwise distances between two sets of points using torch operations.

        Args:
            data_x (torch.Tensor): Data points.
            data_y (torch.Tensor, optional): Data points.

        Returns:
            torch.Tensor: Pairwise distances.
        """
        if data_y is None:
            data_y = data_x
        data_x = data_x.float()
        data_y = data_y.float()
        return torch.cdist(data_x, data_y, p=2)

    def _get_kth_value(self, unsorted: torch.Tensor, k: int, axis: int = -1) -> torch.Tensor:
        """
        Get the kth smallest values along an axis using torch.topk.

        Args:
            unsorted (torch.Tensor): Input tensor.
            k (int): k value.
            axis (int): Axis.

        Returns:
            torch.Tensor: kth smallest values along the specified axis.
        """
        values, _ = torch.topk(unsorted, k, largest=False)
        return values.max(dim=axis).values

    def _compute_nearest_neighbour_distances(
        self, input_features: torch.Tensor, nearest_k: int
    ) -> torch.Tensor:
        """
        Compute distances to kth nearest neighbours using torch operations.

        Args:
            input_features (torch.Tensor): Input features.
            nearest_k (int): Number of nearest neighbors.

        Returns:
            torch.Tensor: Distances to kth nearest neighbours.
        """
        distances = self._compute_pairwise_distance(input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def _compute_prdc_features(
        self, real_features: torch.Tensor, fake_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PRDC features using GPU-based tensor operations.

        Args:
            real_features (torch.Tensor): Reference features.
            fake_features (torch.Tensor): Query features.

        Returns:
            torch.Tensor: PRDC features (recall, density, precision, coverage).
        """
        num_real = real_features.size(0)
        real_distances = self._compute_nearest_neighbour_distances(real_features, self.nearest_k)
        fake_distances = self._compute_nearest_neighbour_distances(fake_features, self.nearest_k)
        distance_matrix = self._compute_pairwise_distance(real_features, fake_features)

        precision = (distance_matrix < real_distances.unsqueeze(1)).any(dim=0).float()
        recall = (distance_matrix < fake_distances).sum(dim=0).float() / num_real
        density = (1.0 / float(self.nearest_k)) * (
            distance_matrix < real_distances.unsqueeze(1)
        ).sum(dim=0).float()
        coverage = (distance_matrix.min(dim=0).values < fake_distances).float()

        return torch.stack((recall, density, precision, coverage), dim=1)

    def fit(
        self, id_texts: Sequence[str], val_split: float = 0.2, random_state: int = 42
    ) -> ForteTextOODDetector:
        """
        Fit the OOD detector on in-distribution texts.

        Args:
            id_texts (list[str]): In-distribution texts.
            val_split (float): Fraction for validation.
            random_state (int): Random seed.

        Returns:
            self: The fitted detector.
        """
        start_time = time.time()
        print(f"Fitting ForteTextOODDetector on {len(id_texts)} texts...")

        # Split texts into training and validation
        id_train_texts, id_val_texts = train_test_split(
            list(id_texts), test_size=val_split, random_state=random_state
        )

        print(f"Extracting features from {len(id_train_texts)} training texts...")
        self.id_train_features = self._extract_features(id_train_texts, name="id_train")

        print(f"Extracting features from {len(id_val_texts)} validation texts...")
        id_val_features = self._extract_features(id_val_texts, name="id_val")

        # Compute PRDC features for each model using GPU tensor operations
        print("Computing PRDC features...")
        assert self.id_train_features is not None
        X_id_train_prdc: list[torch.Tensor] = []
        X_id_val_prdc: list[torch.Tensor] = []
        for model_name in self.id_train_features:
            print(f"Computing PRDC for {model_name}...")
            features = self.id_train_features[model_name]
            # Use torch-based splitting on device
            train_idx = torch.randperm(features.size(0), device=self.device)
            split = int(features.size(0) * 0.5)
            id_train_part1 = features[train_idx[:split]]
            id_train_part2 = features[train_idx[split:]]

            print(f"  Training PRDC: {id_train_part1.shape} vs {id_train_part2.shape}")
            train_prdc = self._compute_prdc_features(id_train_part1, id_train_part2)
            X_id_train_prdc.append(train_prdc)

            val_feats = id_val_features[model_name]
            print(f"  Validation PRDC: {id_train_part1.shape} vs {val_feats.shape}")
            val_prdc = self._compute_prdc_features(id_train_part1, val_feats)
            X_id_val_prdc.append(val_prdc)

        self.id_train_prdc = torch.cat(X_id_train_prdc, dim=1)  # on device
        id_val_prdc = torch.cat(X_id_val_prdc, dim=1)
        print(
            f"Combined PRDC features - Training: {self.id_train_prdc.shape}, Validation: {id_val_prdc.shape}"
        )

        print(f"Training detector ({self.method}) with custom_detector={self.custom_detector}...")
        if self.method == "gmm":
            best_bic = np.inf
            best_n_components = 1
            best_gmm: Any = None
            for n_components in [1, 2, 4, 8, 16, 32, 64]:
                if self.custom_detector:
                    gmm = TorchGMM(
                        n_components=n_components, max_iter=100, tol=1e-3, device=self.device
                    )
                    gmm.fit(self.id_train_prdc)
                    bic_val = gmm.bic(self.id_train_prdc)
                else:
                    id_train_prdc_cpu = self.id_train_prdc.cpu().numpy()
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type="full",
                        random_state=random_state,
                        max_iter=100,
                    )
                    gmm.fit(id_train_prdc_cpu)
                    bic_val = gmm.bic(id_train_prdc_cpu)
                if bic_val < best_bic:
                    best_bic = bic_val
                    best_n_components = n_components
                    best_gmm = gmm
            print(f"Selected {best_n_components} components for GMM with BIC={best_bic:.2f}")
            self.detector = best_gmm

        elif self.method == "kde":
            self.detector = (
                TorchKDE(self.id_train_prdc.T, bw_method="scott", device=self.device)
                if self.custom_detector
                else gaussian_kde(self.id_train_prdc.cpu().numpy().T, bw_method="scott")
            )

        elif self.method == "ocsvm":
            if self.custom_detector:
                best_accuracy = 0.0
                best_nu = 0.01
                best_model: TorchOCSVM | None = None
                for nu in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    model = TorchOCSVM(nu=nu, n_iters=1000, lr=1e-3, device=self.device)
                    model.fit(self.id_train_prdc)
                    decision = model.decision_function(self.id_train_prdc)
                    accuracy = (
                        torch.where(decision.detach() >= 0, 1, -1).float().mean().item() + 1
                    ) / 2.0
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_nu = nu
                        best_model = model
                print(f"Selected nu={best_nu} for TorchOCSVM with accuracy {best_accuracy:.4f}")
                self.detector = best_model
            else:
                best_accuracy = 0.0
                best_nu = 0.01
                for nu in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    try:
                        id_train_prdc_cpu = self.id_train_prdc.cpu().numpy()
                        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
                        ocsvm.fit(id_train_prdc_cpu)
                        val_pred = ocsvm.predict(id_train_prdc_cpu)
                        accuracy = np.mean(val_pred == 1)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_nu = nu
                    except Exception as e:
                        print(f"Error with nu={nu}: {e}")
                        continue
                print(f"Selected nu={best_nu} for OCSVM with accuracy {best_accuracy:.4f}")
                id_train_prdc_cpu = self.id_train_prdc.cpu().numpy()
                self.detector = OneClassSVM(kernel="rbf", gamma="scale", nu=best_nu)
                self.detector.fit(id_train_prdc_cpu)

        self.is_fitted = True
        fit_time = time.time() - start_time
        print(f"ForteTextOODDetector fitted in {fit_time:.2f} seconds.")
        return self

    def _get_ood_scores(self, texts: Sequence[str], cache_name: str = "test") -> np.ndarray:
        """
        Get OOD scores for a set of texts.

        Args:
            texts (list[str]): Text inputs.
            cache_name (str): Identifier for caching.

        Returns:
            np.ndarray: Array of scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        assert self.id_train_features is not None, "id_train_features not computed"
        assert self.detector is not None, "detector not initialized"

        test_features = self._extract_features(texts, name=cache_name)
        X_test_prdc_parts: list[torch.Tensor] = []
        for model_name in test_features:
            ref_features = self.id_train_features[model_name]
            train_idx = torch.randperm(ref_features.size(0), device=self.device)
            split = int(ref_features.size(0) * 0.5)
            id_train_part1 = ref_features[train_idx[:split]]
            test_tensor = test_features[model_name]
            print(
                f"Computing test PRDC for {model_name}: {id_train_part1.shape} vs {test_tensor.shape}"
            )
            test_prdc = self._compute_prdc_features(id_train_part1, test_tensor)
            X_test_prdc_parts.append(test_prdc)

        X_test_prdc = torch.cat(X_test_prdc_parts, dim=1)
        print(f"Combined test PRDC shape: {X_test_prdc.shape}")

        # For custom (GPU-based) detectors, use torch outputs; then convert to numpy if needed.
        if self.custom_detector:
            if self.method == "gmm":
                scores_t = self.detector.score_samples(X_test_prdc)
                scores = scores_t.detach().cpu().numpy()
            elif self.method == "kde":
                scores_t = self.detector.logpdf(X_test_prdc)
                scores = scores_t.detach().cpu().numpy()
            elif self.method == "ocsvm":
                scores_t = self.detector.decision_function(X_test_prdc)
                scores = scores_t.detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported method: {self.method}")
        else:
            X_test_prdc_cpu = X_test_prdc.cpu().numpy()
            if self.method == "gmm":
                scores = self.detector.score_samples(X_test_prdc_cpu)
            elif self.method == "kde":
                scores = self.detector.logpdf(X_test_prdc_cpu.T)
            elif self.method == "ocsvm":
                scores = self.detector.decision_function(X_test_prdc_cpu)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
        return scores

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """
        Predict OOD status.

        Args:
            texts (list[str]): Text inputs.

        Returns:
            np.ndarray: Binary predictions (1 for in-distribution, -1 for OOD).
        """
        scores = self._get_ood_scores(texts)
        if self.method == "ocsvm":
            threshold = 0.0
        else:
            assert self.id_train_prdc is not None, "id_train_prdc not computed"
            if self.custom_detector:
                ref_features = self.id_train_prdc
                # Use a simple split for threshold estimation
                train_idx = torch.randperm(ref_features.size(0), device=self.device)
                split = int(ref_features.size(0) * 0.5)
                id_train_part1 = ref_features[train_idx[:split]]
                if self.method == "gmm":
                    id_scores = self.detector.score_samples(id_train_part1).detach().cpu().numpy()  # type: ignore[union-attr]
                elif self.method == "kde":
                    scorer = getattr(self.detector, "score_samples", None)
                    if callable(scorer):
                        id_scores = scorer(id_train_part1).detach().cpu().numpy()
                    else:
                        logpdf_func = getattr(self.detector, "logpdf", None)
                        if logpdf_func is not None and callable(logpdf_func):
                            id_scores = logpdf_func(id_train_part1).detach().cpu().numpy()
                        else:
                            id_scores = np.array([])
                else:
                    id_scores = np.array([])
            else:
                id_train_part1_np, _ = train_test_split(
                    self.id_train_prdc.cpu().numpy(), test_size=0.5, random_state=42
                )
                if self.method == "gmm":
                    score_samples_func = getattr(self.detector, "score_samples", None)
                    if score_samples_func is not None and callable(score_samples_func):
                        id_scores = score_samples_func(id_train_part1_np)
                    else:
                        id_scores = np.array([])
                elif self.method == "kde":
                    logpdf_func = getattr(self.detector, "logpdf", None)
                    if logpdf_func is not None and callable(logpdf_func):
                        id_scores = logpdf_func(id_train_part1_np.T)
                    else:
                        id_scores = np.array([])
                else:
                    id_scores = np.array([])
            threshold = float(np.percentile(id_scores, 5)) if id_scores.size > 0 else 0.0
        return np.where(scores > threshold, 1, -1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return normalized probability scores for OOD detection.

        Args:
            texts (list[str]): Text inputs.

        Returns:
            np.ndarray: Normalized scores.
        """
        scores = self._get_ood_scores(texts)
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5
        return normalized_scores

    def evaluate(self, id_texts: Sequence[str], ood_texts: Sequence[str]) -> dict[str, float]:
        """
        Evaluate the detector.

        Args:
            id_texts (list[str]): In-distribution texts.
            ood_texts (list[str]): OOD texts.

        Returns:
            dict: Evaluation metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before evaluation")

        print(f"Evaluating on {len(id_texts)} ID and {len(ood_texts)} OOD texts...")

        # Fuse ID and OOD samples for processing together
        all_texts = list(id_texts) + list(ood_texts)
        all_scores = self._get_ood_scores(all_texts, cache_name="eval_fused")

        # Split the scores back to ID and OOD
        id_scores = all_scores[: len(id_texts)]
        ood_scores = all_scores[len(id_texts) :]

        print("\nScore Statistics:")
        print(
            f"ID  - Mean: {np.mean(id_scores):.4f}, Std: {np.std(id_scores):.4f}, Min: {np.min(id_scores):.4f}, Max: {np.max(id_scores):.4f}"
        )
        print(
            f"OOD - Mean: {np.mean(ood_scores):.4f}, Std: {np.std(ood_scores):.4f}, Min: {np.min(ood_scores):.4f}, Max: {np.max(ood_scores):.4f}"
        )

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

    def compute_similarity(
        self, texts1: Sequence[str], texts2: Sequence[str], model_name: str | None = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of texts using one of the loaded models.

        Args:
            texts1 (list[str]): First set of texts.
            texts2 (list[str]): Second set of texts.
            model_name (str | None): Short name of the model to use; if None, use the first model.

        Returns:
            torch.Tensor: Similarity matrix of shape (len(texts1), len(texts2)).
        """
        if self.models is None:
            self.models = self._init_models()
        assert self.models is not None and len(self.models) > 0

        # Select model
        st_model: SentenceTransformer
        if model_name is not None:
            st_model = next(
                (m for short, m, _ in self.models if short == model_name), self.models[0][1]
            )
        else:
            st_model = self.models[0][1]  # first model

        # Encode with normalization
        emb1 = st_model.encode(list(texts1), convert_to_tensor=True, normalize_embeddings=True).to(
            self.device
        )
        emb2 = st_model.encode(list(texts2), convert_to_tensor=True, normalize_embeddings=True).to(
            self.device
        )

        # Since embeddings are normalized, cosine similarity = dot product
        sim = emb1 @ emb2.T
        return sim
