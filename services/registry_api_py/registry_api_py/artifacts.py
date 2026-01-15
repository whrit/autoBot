"""Artifact storage module for strategy artifacts (JSON configs, ONNX models, etc.)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class ArtifactType(str, Enum):
    """Types of artifacts that can be stored."""

    ONNX = "onnx"
    JSON_CONFIG = "json"
    PARAMETERS = "params"
    METRICS = "metrics"


@dataclass
class ArtifactMetadata:
    """Metadata for a stored artifact."""

    artifact_id: int
    strategy_id: int
    artifact_type: ArtifactType
    filename: str
    checksum: str
    size_bytes: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, str | int]:
        """Convert metadata to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "strategy_id": self.strategy_id,
            "artifact_type": self.artifact_type.value,
            "filename": self.filename,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
        }


class ArtifactNotFoundError(Exception):
    """Raised when an artifact is not found."""

    pass


class ArtifactChecksumError(Exception):
    """Raised when artifact checksum verification fails."""

    pass


class ArtifactStorage:
    """Local filesystem artifact storage with MinIO-compatible interface.

    Directory structure:
        base_path/
            strategies/
                {strategy_id}/
                    {artifact_id}_{artifact_type}.{ext}
            metadata/
                {artifact_id}.json
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize artifact storage.

        Args:
            base_path: Base directory for artifact storage.
        """
        self._base_path = base_path
        self._strategies_path = base_path / "strategies"
        self._metadata_path = base_path / "metadata"
        self._next_artifact_id = 1

        # Create directories
        self._strategies_path.mkdir(parents=True, exist_ok=True)
        self._metadata_path.mkdir(parents=True, exist_ok=True)

        # Load existing metadata to determine next artifact ID
        self._load_existing_metadata()

    def _load_existing_metadata(self) -> None:
        """Load existing metadata files to determine next artifact ID."""
        if not self._metadata_path.exists():
            return

        max_id = 0
        for meta_file in self._metadata_path.glob("*.json"):
            try:
                artifact_id = int(meta_file.stem)
                max_id = max(max_id, artifact_id)
            except ValueError:
                continue

        self._next_artifact_id = max_id + 1

    def _get_extension(self, artifact_type: ArtifactType) -> str:
        """Get file extension for artifact type."""
        extensions = {
            ArtifactType.ONNX: "onnx",
            ArtifactType.JSON_CONFIG: "json",
            ArtifactType.PARAMETERS: "json",
            ArtifactType.METRICS: "json",
        }
        return extensions[artifact_type]

    def _get_strategy_path(self, strategy_id: int) -> Path:
        """Get path to strategy artifact directory."""
        return self._strategies_path / str(strategy_id)

    def _get_artifact_path(
        self, strategy_id: int, artifact_id: int, artifact_type: ArtifactType
    ) -> Path:
        """Get path to artifact file."""
        ext = self._get_extension(artifact_type)
        filename = f"{artifact_id}_{artifact_type.value}.{ext}"
        return self._get_strategy_path(strategy_id) / filename

    def _get_metadata_path(self, artifact_id: int) -> Path:
        """Get path to metadata file."""
        return self._metadata_path / f"{artifact_id}.json"

    def compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data.

        Args:
            data: Bytes to compute checksum for.

        Returns:
            Hexadecimal SHA-256 checksum.
        """
        return hashlib.sha256(data).hexdigest()

    def upload(
        self,
        strategy_id: int,
        artifact_type: ArtifactType,
        data: bytes,
        filename: str | None = None,
    ) -> ArtifactMetadata:
        """Upload an artifact.

        Args:
            strategy_id: ID of the strategy this artifact belongs to.
            artifact_type: Type of artifact being uploaded.
            data: Artifact data as bytes.
            filename: Optional original filename.

        Returns:
            Metadata for the uploaded artifact.
        """
        artifact_id = self._next_artifact_id
        self._next_artifact_id += 1

        # Compute checksum
        checksum = self.compute_checksum(data)

        # Create strategy directory if needed
        strategy_path = self._get_strategy_path(strategy_id)
        strategy_path.mkdir(parents=True, exist_ok=True)

        # Determine filename
        ext = self._get_extension(artifact_type)
        if filename is None:
            filename = f"{artifact_id}_{artifact_type.value}.{ext}"

        # Write artifact data
        artifact_path = self._get_artifact_path(strategy_id, artifact_id, artifact_type)
        artifact_path.write_bytes(data)

        # Create metadata
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            strategy_id=strategy_id,
            artifact_type=artifact_type,
            filename=filename,
            checksum=checksum,
            size_bytes=len(data),
        )

        # Write metadata
        metadata_path = self._get_metadata_path(artifact_id)
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2))

        return metadata

    def download(self, artifact_id: int, verify_checksum: bool = True) -> bytes:
        """Download an artifact.

        Args:
            artifact_id: ID of the artifact to download.
            verify_checksum: Whether to verify the checksum after download.

        Returns:
            Artifact data as bytes.

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist.
            ArtifactChecksumError: If checksum verification fails.
        """
        # Load metadata
        metadata = self.get_metadata(artifact_id)

        # Get artifact path
        artifact_path = self._get_artifact_path(
            metadata.strategy_id, artifact_id, metadata.artifact_type
        )

        if not artifact_path.exists():
            raise ArtifactNotFoundError(f"Artifact file not found: {artifact_id}")

        # Read data
        data = artifact_path.read_bytes()

        # Verify checksum
        if verify_checksum:
            actual_checksum = self.compute_checksum(data)
            if actual_checksum != metadata.checksum:
                raise ArtifactChecksumError(
                    f"Checksum mismatch for artifact {artifact_id}: "
                    f"expected {metadata.checksum}, got {actual_checksum}"
                )

        return data

    def delete(self, artifact_id: int) -> bool:
        """Delete an artifact.

        Args:
            artifact_id: ID of the artifact to delete.

        Returns:
            True if artifact was deleted, False if not found.
        """
        metadata_path = self._get_metadata_path(artifact_id)

        if not metadata_path.exists():
            return False

        # Load metadata to get artifact path
        try:
            metadata = self.get_metadata(artifact_id)
        except ArtifactNotFoundError:
            return False

        # Delete artifact file
        artifact_path = self._get_artifact_path(
            metadata.strategy_id, artifact_id, metadata.artifact_type
        )
        if artifact_path.exists():
            artifact_path.unlink()

        # Delete metadata file
        metadata_path.unlink()

        return True

    def get_metadata(self, artifact_id: int) -> ArtifactMetadata:
        """Get metadata for an artifact.

        Args:
            artifact_id: ID of the artifact.

        Returns:
            Artifact metadata.

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist.
        """
        metadata_path = self._get_metadata_path(artifact_id)

        if not metadata_path.exists():
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_id}")

        data = json.loads(metadata_path.read_text())

        return ArtifactMetadata(
            artifact_id=data["artifact_id"],
            strategy_id=data["strategy_id"],
            artifact_type=ArtifactType(data["artifact_type"]),
            filename=data["filename"],
            checksum=data["checksum"],
            size_bytes=data["size_bytes"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def list_by_strategy(self, strategy_id: int) -> list[ArtifactMetadata]:
        """List all artifacts for a strategy.

        Args:
            strategy_id: ID of the strategy.

        Returns:
            List of artifact metadata.
        """
        artifacts = []
        for meta_file in self._metadata_path.glob("*.json"):
            try:
                metadata = self.get_metadata(int(meta_file.stem))
                if metadata.strategy_id == strategy_id:
                    artifacts.append(metadata)
            except (ValueError, ArtifactNotFoundError):
                continue

        return sorted(artifacts, key=lambda a: a.artifact_id)

    def list_all(self) -> Iterator[ArtifactMetadata]:
        """List all artifacts.

        Yields:
            Artifact metadata for each stored artifact.
        """
        for meta_file in self._metadata_path.glob("*.json"):
            try:
                yield self.get_metadata(int(meta_file.stem))
            except (ValueError, ArtifactNotFoundError):
                continue

    def clear(self) -> int:
        """Delete all artifacts.

        Returns:
            Number of artifacts deleted.
        """
        count = 0
        for meta_file in list(self._metadata_path.glob("*.json")):
            try:
                artifact_id = int(meta_file.stem)
                if self.delete(artifact_id):
                    count += 1
            except ValueError:
                continue

        # Clean up empty strategy directories
        for strategy_dir in self._strategies_path.iterdir():
            if strategy_dir.is_dir() and not any(strategy_dir.iterdir()):
                strategy_dir.rmdir()

        self._next_artifact_id = 1
        return count

    def cleanup(self) -> None:
        """Remove all storage directories."""
        if self._base_path.exists():
            shutil.rmtree(self._base_path)
