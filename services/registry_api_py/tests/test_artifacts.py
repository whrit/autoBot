"""Tests for ArtifactStorage class and artifact API endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from registry_api_py.artifacts import (
    ArtifactNotFoundError,
    ArtifactStorage,
    ArtifactType,
)


class TestArtifactStorage:
    """Tests for ArtifactStorage class."""

    @pytest.fixture
    def storage(self) -> ArtifactStorage:
        """Create a temporary artifact storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ArtifactStorage(Path(tmpdir) / "artifacts")
            yield storage
            storage.cleanup()

    @pytest.fixture
    def sample_json_config(self) -> bytes:
        """Sample JSON configuration data."""
        config = {
            "strategy_name": "momentum_v1",
            "parameters": {"lookback": 20, "threshold": 0.02},
        }
        return json.dumps(config).encode("utf-8")

    @pytest.fixture
    def sample_onnx_data(self) -> bytes:
        """Sample ONNX model data (mock binary)."""
        # Simple mock ONNX header for testing
        return b"ONNX_MOCK_MODEL_DATA_" + bytes(range(256))

    @pytest.fixture
    def sample_metrics_data(self) -> bytes:
        """Sample metrics data."""
        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.15,
            "win_rate": 0.55,
        }
        return json.dumps(metrics).encode("utf-8")

    def test_upload_artifact(self, storage: ArtifactStorage, sample_json_config: bytes) -> None:
        """Test uploading an artifact."""
        metadata = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )

        assert metadata.artifact_id == 1
        assert metadata.strategy_id == 1
        assert metadata.artifact_type == ArtifactType.JSON_CONFIG
        assert metadata.size_bytes == len(sample_json_config)
        assert len(metadata.checksum) == 64  # SHA-256 hex length

    def test_download_artifact(self, storage: ArtifactStorage, sample_json_config: bytes) -> None:
        """Test downloading an artifact."""
        metadata = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )

        downloaded = storage.download(metadata.artifact_id)

        assert downloaded == sample_json_config

    def test_delete_artifact(self, storage: ArtifactStorage, sample_json_config: bytes) -> None:
        """Test deleting an artifact."""
        metadata = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )

        result = storage.delete(metadata.artifact_id)
        assert result is True

        # Verify artifact is gone
        with pytest.raises(ArtifactNotFoundError):
            storage.download(metadata.artifact_id)

    def test_delete_nonexistent_artifact(self, storage: ArtifactStorage) -> None:
        """Test deleting a non-existent artifact returns False."""
        result = storage.delete(9999)
        assert result is False

    def test_checksum_verification(
        self, storage: ArtifactStorage, sample_json_config: bytes
    ) -> None:
        """Test that checksum is computed correctly."""
        expected_checksum = storage.compute_checksum(sample_json_config)

        metadata = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )

        assert metadata.checksum == expected_checksum

    def test_checksum_verification_on_download(
        self, storage: ArtifactStorage, sample_json_config: bytes
    ) -> None:
        """Test that checksum verification works on download."""
        metadata = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )

        # Download with verification enabled (default)
        downloaded = storage.download(metadata.artifact_id, verify_checksum=True)
        assert downloaded == sample_json_config

    def test_onnx_artifact_upload(self, storage: ArtifactStorage, sample_onnx_data: bytes) -> None:
        """Test uploading an ONNX model artifact."""
        metadata = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.ONNX,
            data=sample_onnx_data,
            filename="model.onnx",
        )

        assert metadata.artifact_type == ArtifactType.ONNX
        assert metadata.size_bytes == len(sample_onnx_data)

        # Verify download
        downloaded = storage.download(metadata.artifact_id)
        assert downloaded == sample_onnx_data

    def test_multiple_artifacts_for_strategy(
        self,
        storage: ArtifactStorage,
        sample_json_config: bytes,
        sample_onnx_data: bytes,
        sample_metrics_data: bytes,
    ) -> None:
        """Test uploading multiple artifacts for a single strategy."""
        strategy_id = 1

        # Upload multiple artifacts
        config_meta = storage.upload(
            strategy_id=strategy_id,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )
        onnx_meta = storage.upload(
            strategy_id=strategy_id,
            artifact_type=ArtifactType.ONNX,
            data=sample_onnx_data,
        )
        metrics_meta = storage.upload(
            strategy_id=strategy_id,
            artifact_type=ArtifactType.METRICS,
            data=sample_metrics_data,
        )

        # List artifacts for strategy
        artifacts = storage.list_by_strategy(strategy_id)

        assert len(artifacts) == 3
        artifact_ids = {a.artifact_id for a in artifacts}
        assert artifact_ids == {config_meta.artifact_id, onnx_meta.artifact_id, metrics_meta.artifact_id}

    def test_list_artifacts_by_strategy_empty(self, storage: ArtifactStorage) -> None:
        """Test listing artifacts for a strategy with no artifacts."""
        artifacts = storage.list_by_strategy(999)
        assert artifacts == []

    def test_get_metadata(self, storage: ArtifactStorage, sample_json_config: bytes) -> None:
        """Test getting artifact metadata."""
        original = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )

        metadata = storage.get_metadata(original.artifact_id)

        assert metadata.artifact_id == original.artifact_id
        assert metadata.strategy_id == original.strategy_id
        assert metadata.artifact_type == original.artifact_type
        assert metadata.checksum == original.checksum
        assert metadata.size_bytes == original.size_bytes

    def test_get_metadata_not_found(self, storage: ArtifactStorage) -> None:
        """Test getting metadata for non-existent artifact."""
        with pytest.raises(ArtifactNotFoundError):
            storage.get_metadata(9999)

    def test_artifact_id_increments(
        self, storage: ArtifactStorage, sample_json_config: bytes
    ) -> None:
        """Test that artifact IDs increment correctly."""
        meta1 = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )
        meta2 = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )
        meta3 = storage.upload(
            strategy_id=2,
            artifact_type=ArtifactType.JSON_CONFIG,
            data=sample_json_config,
        )

        assert meta1.artifact_id == 1
        assert meta2.artifact_id == 2
        assert meta3.artifact_id == 3

    def test_list_all_artifacts(
        self,
        storage: ArtifactStorage,
        sample_json_config: bytes,
        sample_onnx_data: bytes,
    ) -> None:
        """Test listing all artifacts."""
        storage.upload(strategy_id=1, artifact_type=ArtifactType.JSON_CONFIG, data=sample_json_config)
        storage.upload(strategy_id=2, artifact_type=ArtifactType.ONNX, data=sample_onnx_data)

        all_artifacts = list(storage.list_all())
        assert len(all_artifacts) == 2

    def test_clear_storage(
        self,
        storage: ArtifactStorage,
        sample_json_config: bytes,
        sample_onnx_data: bytes,
    ) -> None:
        """Test clearing all artifacts."""
        storage.upload(strategy_id=1, artifact_type=ArtifactType.JSON_CONFIG, data=sample_json_config)
        storage.upload(strategy_id=2, artifact_type=ArtifactType.ONNX, data=sample_onnx_data)

        count = storage.clear()
        assert count == 2

        all_artifacts = list(storage.list_all())
        assert len(all_artifacts) == 0

    def test_parameters_artifact_type(self, storage: ArtifactStorage) -> None:
        """Test uploading parameters artifact type."""
        params = json.dumps({"param1": 1.0, "param2": 0.5}).encode()

        metadata = storage.upload(
            strategy_id=1,
            artifact_type=ArtifactType.PARAMETERS,
            data=params,
        )

        assert metadata.artifact_type == ArtifactType.PARAMETERS

        downloaded = storage.download(metadata.artifact_id)
        assert downloaded == params

    def test_download_not_found(self, storage: ArtifactStorage) -> None:
        """Test downloading non-existent artifact raises error."""
        with pytest.raises(ArtifactNotFoundError):
            storage.download(9999)


class TestArtifactStoragePersistence:
    """Tests for ArtifactStorage persistence across instances."""

    def test_storage_persistence(self) -> None:
        """Test that artifacts persist across storage instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "artifacts"
            data = b"test data for persistence"

            # Create first storage instance and upload
            storage1 = ArtifactStorage(base_path)
            metadata = storage1.upload(
                strategy_id=1,
                artifact_type=ArtifactType.JSON_CONFIG,
                data=data,
            )
            artifact_id = metadata.artifact_id

            # Create second storage instance and verify data persists
            storage2 = ArtifactStorage(base_path)
            downloaded = storage2.download(artifact_id)

            assert downloaded == data

    def test_artifact_id_continuity(self) -> None:
        """Test that artifact IDs continue correctly across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "artifacts"
            data = b"test data"

            # Create first instance and upload
            storage1 = ArtifactStorage(base_path)
            meta1 = storage1.upload(strategy_id=1, artifact_type=ArtifactType.JSON_CONFIG, data=data)
            meta2 = storage1.upload(strategy_id=1, artifact_type=ArtifactType.JSON_CONFIG, data=data)

            # Create second instance and upload
            storage2 = ArtifactStorage(base_path)
            meta3 = storage2.upload(strategy_id=1, artifact_type=ArtifactType.JSON_CONFIG, data=data)

            assert meta1.artifact_id == 1
            assert meta2.artifact_id == 2
            assert meta3.artifact_id == 3
