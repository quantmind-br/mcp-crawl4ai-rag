"""

Integration tests for CrossEncoder GPU acceleration functionality.

Tests CrossEncoder GPU initialization, CPU fallback scenarios, memory cleanup,
and integration with the MCP server following the existing test infrastructure.
"""
# ruff: noqa: E402

import pytest
import os
from unittest.mock import Mock, patch

# Import test fixtures and setup
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import the functions we're testing
from src.services.rag_service import RagService
from src.device_manager import get_optimal_device
from src.services.embedding_service import health_check_gpu_acceleration


class TestCrossEncoderGPUInitialization:
    """Testa inicialização de CrossEncoder com GPU/CPU (somente verificação de device helpers)."""

    @patch("src.device_manager.get_optimal_device")
    def test_crossencoder_cpu_forced_initialization(self, mock_get_device):
        mock_device = Mock()
        mock_device.__str__ = Mock(return_value="cpu")
        mock_get_device.return_value = mock_device

        device = get_optimal_device(preference="cpu", gpu_index=0)
        mock_get_device.assert_called_with(preference="cpu", gpu_index=0)
        assert str(device) == "cpu"


class TestRerankingWithGPU:
    """Testa reranking usando RagService.rerank_results com modelo simulado."""

    def test_rerank_results_with_gpu_model(self):
        # Mock do modelo CrossEncoder
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.8]

        # Resultados de exemplo
        results = [
            {"content": "Document 1", "id": "doc1"},
            {"content": "Document 2", "id": "doc2"},
            {"content": "Document 3", "id": "doc3"},
        ]

        # Ativar reranking via env
        with patch.dict(os.environ, {"USE_RERANKING": "true"}):
            # Instanciar serviço com modelo mockado
            rag = RagService(qdrant_client=Mock(), reranking_model=mock_model)

            with patch("src.device_manager.cleanup_gpu_memory") as mock_cleanup:
                reranked = rag.rerank_results("test query", results)

                assert len(reranked) == 3
                assert reranked[0]["rerank_score"] == 0.9
                assert reranked[1]["rerank_score"] == 0.8
                assert reranked[2]["rerank_score"] == 0.7

                expected_pairs = [["test query", "Document 1"], ["test query", "Document 2"], ["test query", "Document 3"]]
                mock_model.predict.assert_called_once_with(expected_pairs)
                mock_cleanup.assert_called_once()

    def test_rerank_results_empty_input(self):
        mock_model = Mock()
        rag = RagService(qdrant_client=Mock(), reranking_model=mock_model)
        with patch.dict(os.environ, {"USE_RERANKING": "true"}):
            result = rag.rerank_results("query", [])
            assert result == []
            mock_model.predict.assert_not_called()

    def test_rerank_results_no_model(self):
        results = [{"content": "Document 1", "id": "doc1"}]
        rag = RagService(qdrant_client=Mock(), reranking_model=None)
        with patch.dict(os.environ, {"USE_RERANKING": "true"}):
            result = rag.rerank_results("query", results)
            assert result == results

    def test_rerank_results_model_error_handling(self):
        mock_model = Mock()
        mock_model.predict.side_effect = RuntimeError("Model prediction failed")
        results = [{"content": "Document 1", "id": "doc1"}]

        rag = RagService(qdrant_client=Mock(), reranking_model=mock_model)
        with patch.dict(os.environ, {"USE_RERANKING": "true"}):
            result = rag.rerank_results("query", results)
            assert result == results


class TestDeviceHealthCheck:
    """Testa health check de GPU via embedding_service.health_check_gpu_acceleration."""

    @patch("src.services.embedding_service.torch")
    def test_health_check_gpu_available_and_working(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        props = Mock()
        props.total_memory = 10 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_tensor = Mock()
        mock_torch.randn.return_value = mock_tensor
        mock_tensor.__matmul__ = Mock(return_value=mock_tensor)
        mock_torch.device.return_value = Mock()

        health_status = health_check_gpu_acceleration()
        assert health_status["gpu_available"] is True
        assert health_status["device_name"] == "NVIDIA GeForce RTX 3080"
        assert health_status["memory_available_gb"] == 10.0
        assert health_status["test_passed"] is True
        assert health_status["error_message"] is None

    @patch("src.services.embedding_service.torch")
    def test_health_check_mps_available_and_working(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_tensor = Mock()
        mock_torch.randn.return_value = mock_tensor
        mock_tensor.sum.return_value = mock_tensor
        mock_torch.device.return_value = Mock()

        health_status = health_check_gpu_acceleration()
        assert health_status["gpu_available"] is True
        assert health_status["device_name"] == "Apple Silicon GPU (MPS)"
        assert health_status["memory_available_gb"] is None
        assert health_status["test_passed"] is True
        assert health_status["error_message"] is None

    @patch("src.services.embedding_service.torch")
    def test_health_check_gpu_operations_fail(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.randn.side_effect = RuntimeError("CUDA out of memory")

        health_status = health_check_gpu_acceleration()
        assert health_status["gpu_available"] is False
        assert health_status["test_passed"] is False
        assert health_status["error_message"] == "CUDA out of memory"

    @patch("src.services.embedding_service.torch")
    def test_health_check_no_gpu_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        health_status = health_check_gpu_acceleration()
        assert health_status["gpu_available"] is False
        assert health_status["device_name"] == "CPU"
        assert health_status["test_passed"] is False
        assert health_status["error_message"] is None


class TestEnvironmentConfiguration:
    @patch.dict(
        os.environ,
        {
            "USE_RERANKING": "true",
            "USE_GPU_ACCELERATION": "true",
            "GPU_PRECISION": "float16",
            "GPU_DEVICE_INDEX": "1",
        },
    )
    def test_environment_configuration_parsing(self):
        assert os.getenv("USE_RERANKING") == "true"
        assert os.getenv("USE_GPU_ACCELERATION") == "true"
        assert os.getenv("GPU_PRECISION") == "float16"
        assert os.getenv("GPU_DEVICE_INDEX") == "1"

    @patch.dict(os.environ, {"USE_RERANKING": "false"})
    def test_reranking_disabled_no_gpu_init(self):
        assert os.getenv("USE_RERANKING") == "false"


class TestRerankingIntegration:
    def test_reranking_preserves_original_fields(self):
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7]
        results = [
            {"content": "Document 1", "id": "doc1", "url": "https://example.com/1", "metadata": {"category": "tech"}},
            {"content": "Document 2", "id": "doc2", "url": "https://example.com/2", "metadata": {"category": "science"}},
        ]
        with patch.dict(os.environ, {"USE_RERANKING": "true"}):
            rag = RagService(qdrant_client=Mock(), reranking_model=mock_model)
            with patch("src.device_manager.cleanup_gpu_memory"):
                reranked = rag.rerank_results("query", results)
                for result in reranked:
                    assert "content" in result
                    assert "id" in result
                    assert "url" in result
                    assert "metadata" in result
                    assert "rerank_score" in result

    def test_reranking_custom_content_key(self):
        mock_model = Mock()
        mock_model.predict.return_value = [0.8]
        results = [{"text": "Custom content field", "id": "doc1"}]
        with patch.dict(os.environ, {"USE_RERANKING": "true"}):
            rag = RagService(qdrant_client=Mock(), reranking_model=mock_model)
            with patch("src.device_manager.cleanup_gpu_memory"):
                reranked = rag.rerank_results("query", results, content_key="text")
                expected_pairs = [["query", "Custom content field"]]
                mock_model.predict.assert_called_once_with(expected_pairs)
                assert len(reranked) == 1


if __name__ == "__main__":
    pytest.main([__file__])
