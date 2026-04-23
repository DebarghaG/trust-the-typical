import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class APILoggerCache:
    """
    A persistent, file-based cache that logs all raw API inputs, outputs,
    and metadata for auditing and reproducibility.
    Uses a content-addressed, hierarchical structure.
    """

    def __init__(self, cache_dir_name: str = "api_cache"):
        """
        Initializes the logger/cache.

        Args:
            cache_dir_name: The name of the root directory for this cache.
        """
        self.root_dir = Path(cache_dir_name)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, text: str, model_id: str) -> Path:
        """
        Computes the unique directory path for a given text and model.
        e.g., <root_dir>/<model_id>/<hash_prefix1>/<hash_prefix2>/<full_hash>/
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Create a hierarchical path to avoid having too many files in one directory
        return self.root_dir / model_id / text_hash[:2] / text_hash[2:4] / text_hash

    def check_and_load(self, text: str, model_id: str) -> dict[str, Any] | None:
        """
        Checks if a valid, successful response exists in the cache and loads it.

        Returns:
            The raw API output dictionary if a valid cache entry is found, otherwise None.
        """
        cache_path = self._get_cache_path(text, model_id)
        output_file = cache_path / "output.json"
        metadata_file = cache_path / "metadata.json"

        if output_file.exists() and metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                # Ensure the cached request was successful
                if metadata.get("success", False):
                    with open(output_file) as f:
                        return json.load(f)
            except (OSError, json.JSONDecodeError):
                # If files are corrupt, treat as a cache miss
                return None
        return None

    def log_and_save(
        self,
        text: str,
        model_id: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        duration_s: float,
        success: bool,
    ) -> None:
        """
        Logs the entire API transaction to the file system.
        """
        cache_path = self._get_cache_path(text, model_id)
        cache_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "duration_seconds": round(duration_s, 4),
            "success": success,
        }

        try:
            with open(cache_path / "input.json", "w") as f:
                json.dump(request_payload, f, indent=2)
            with open(cache_path / "output.json", "w") as f:
                json.dump(response_payload, f, indent=2)
            with open(cache_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except OSError as e:
            print(f"Error: Could not write to cache directory {cache_path}. Error: {e}")
