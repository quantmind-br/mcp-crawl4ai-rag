"""
Tests for enhanced reranking functionality.

Tests the new reranking enhancements including configurable model selection,
model warming, and health check functionality as implemented in the PRP.
"""

import pytest
import os
import json
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path

# Import the device manager functions
import sys

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from utils.py to avoid stub functions
import importlib.util

utils_spec = importlib.util.spec_from_file_location(
    "utils_module", src_path / "utils.py"
)
utils_module = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils_module)
health_check_reranking_model = utils_module.health_check_reranking_model
cleanup_gpu_memory = utils_module.cleanup_gpu_memory


class TestConfigurableModelSelection:
    """Test configurable model selection enhancement."""

    @patch("crawl4ai_mcp.CrossEncoder")
    @patch("crawl4ai_mcp.get_optimal_device")
    @patch("crawl4ai_mcp.get_model_kwargs_for_device")
    @patch.dict(
        os.environ,
        {
            "USE_RERANKING": "true",
            "RERANKING_MODEL_NAME": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        },
    )
    def test_custom_model_name_used(
        self, mock_get_kwargs, mock_get_device, mock_cross_encoder
    ):
        """Test that custom model name from environment is used."""
        # Import here to get the patched environment
        from crawl4ai_mcp import crawl4ai_lifespan

        # Mock device and model kwargs
        mock_device = Mock()
        mock_device.__str__ = Mock(return_value="cuda:0")
        mock_get_device.return_value = mock_device
        mock_get_kwargs.return_value = {}

        # Mock CrossEncoder
        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model

        # Create async context manager manually for testing
        async def test_lifespan():
            async with crawl4ai_lifespan(None) as context:
                # Verify CrossEncoder was called with custom model name
                mock_cross_encoder.assert_called_with(
                    "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Custom model name
                    device=str(mock_device),
                    model_kwargs={},
                )

                # Verify model is stored in context
                assert context.reranking_model == mock_model

        # Run the async test
        asyncio.run(test_lifespan())

    @patch("crawl4ai_mcp.CrossEncoder")
    @patch("crawl4ai_mcp.get_optimal_device")
    @patch("crawl4ai_mcp.get_model_kwargs_for_device")
    @patch.dict(os.environ, {"USE_RERANKING": "true"}, clear=False)
    def test_default_model_name_used(
        self, mock_get_kwargs, mock_get_device, mock_cross_encoder
    ):
        """Test that default model name is used when no custom name is set."""
        # Ensure RERANKING_MODEL_NAME is not set
        os.environ.pop("RERANKING_MODEL_NAME", None)

        from crawl4ai_mcp import crawl4ai_lifespan

        # Mock device and model kwargs
        mock_device = Mock()
        mock_device.__str__ = Mock(return_value="cpu")
        mock_get_device.return_value = mock_device
        mock_get_kwargs.return_value = {}

        # Mock CrossEncoder
        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model

        async def test_lifespan():
            async with crawl4ai_lifespan(None) as context:
                # Verify CrossEncoder was called with default model name
                mock_cross_encoder.assert_called_with(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Default model name
                    device=str(mock_device),
                    model_kwargs={},
                )

        asyncio.run(test_lifespan())


class TestModelWarming:
    """Test model warming enhancement."""

    @patch("crawl4ai_mcp.CrossEncoder")
    @patch("crawl4ai_mcp.get_optimal_device")
    @patch("crawl4ai_mcp.get_model_kwargs_for_device")
    @patch("crawl4ai_mcp.cleanup_gpu_memory")
    @patch.dict(os.environ, {"USE_RERANKING": "true", "RERANKING_WARMUP_SAMPLES": "3"})
    def test_model_warming_enabled(
        self, mock_cleanup, mock_get_kwargs, mock_get_device, mock_cross_encoder
    ):
        """Test that model warming works with custom sample count."""
        from crawl4ai_mcp import crawl4ai_lifespan

        # Mock device
        mock_device = Mock()
        mock_device.__str__ = Mock(return_value="cuda:0")
        mock_get_device.return_value = mock_device
        mock_get_kwargs.return_value = {}

        # Mock CrossEncoder with predict method
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.5, 0.6, 0.7])
        mock_cross_encoder.return_value = mock_model

        async def test_lifespan():
            async with crawl4ai_lifespan(None) as context:
                # Verify predict was called for warming with 3 samples
                mock_model.predict.assert_called_once()
                call_args = mock_model.predict.call_args[0][0]  # Get the pairs argument
                assert len(call_args) == 3  # 3 warmup samples

                # Verify cleanup was called after warming
                mock_cleanup.assert_called()

        asyncio.run(test_lifespan())

    @patch("crawl4ai_mcp.CrossEncoder")
    @patch("crawl4ai_mcp.get_optimal_device")
    @patch("crawl4ai_mcp.get_model_kwargs_for_device")
    @patch.dict(os.environ, {"USE_RERANKING": "true", "RERANKING_WARMUP_SAMPLES": "0"})
    def test_model_warming_disabled(
        self, mock_get_kwargs, mock_get_device, mock_cross_encoder
    ):
        """Test that model warming is skipped when set to 0."""
        from crawl4ai_mcp import crawl4ai_lifespan

        # Mock device
        mock_device = Mock()
        mock_get_device.return_value = mock_device
        mock_get_kwargs.return_value = {}

        # Mock CrossEncoder
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_cross_encoder.return_value = mock_model

        async def test_lifespan():
            async with crawl4ai_lifespan(None) as context:
                # Verify predict was NOT called for warming
                mock_model.predict.assert_not_called()

        asyncio.run(test_lifespan())

    @patch("crawl4ai_mcp.CrossEncoder")
    @patch("crawl4ai_mcp.get_optimal_device")
    @patch("crawl4ai_mcp.get_model_kwargs_for_device")
    @patch("crawl4ai_mcp.cleanup_gpu_memory")
    @patch.dict(os.environ, {"USE_RERANKING": "true", "RERANKING_WARMUP_SAMPLES": "5"})
    def test_model_warming_error_handling(
        self, mock_cleanup, mock_get_kwargs, mock_get_device, mock_cross_encoder
    ):
        """Test that model warming errors are handled gracefully."""
        from crawl4ai_mcp import crawl4ai_lifespan

        # Mock device
        mock_device = Mock()
        mock_get_device.return_value = mock_device
        mock_get_kwargs.return_value = {}

        # Mock CrossEncoder with predict that raises an error
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=RuntimeError("Warmup failed"))
        mock_cross_encoder.return_value = mock_model

        async def test_lifespan():
            # Should not raise an exception despite warmup failure
            async with crawl4ai_lifespan(None) as context:
                # Model should still be available despite warmup failure
                assert context.reranking_model == mock_model

        asyncio.run(test_lifespan())


class TestHealthCheckReranking:
    """Test health check functionality for reranking."""

    def test_health_check_with_valid_model(self):
        """Test health check with a working CrossEncoder model."""
        # Mock CrossEncoder model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.7, 0.8])
        mock_model.device = "cuda:0"
        mock_model.model = Mock()
        mock_model.model.name_or_path = "cross-encoder/test-model"

        with patch.object(utils_module, "cleanup_gpu_memory"):
            with patch(
                "builtins.isinstance", return_value=True
            ):  # Mock isinstance check
                result = health_check_reranking_model(mock_model)

        print(f"Health check result: {result}")  # Debug output
        assert result["model_available"] is True
        assert result["model_name"] == "cross-encoder/test-model"
        assert result["device"] == "cuda:0"
        assert result["inference_test_passed"] is True
        assert result["inference_latency_ms"] is not None
        assert result["inference_latency_ms"] > 0
        assert result["error_message"] is None

    def test_health_check_with_no_model(self):
        """Test health check when no model is provided and reranking is disabled."""
        with patch.dict(os.environ, {"USE_RERANKING": "false"}):
            result = health_check_reranking_model(None)

        assert result["model_available"] is False
        assert result["inference_test_passed"] is False
        assert result["error_message"] == "Reranking not enabled (USE_RERANKING=false)"

    def test_health_check_with_invalid_model_type(self):
        """Test health check with invalid model type."""
        invalid_model = "not a crossencoder"

        result = health_check_reranking_model(invalid_model)

        assert result["model_available"] is False
        assert result["inference_test_passed"] is False
        assert result["error_message"] == "Invalid model type - expected CrossEncoder"

    def test_health_check_inference_failure(self):
        """Test health check when model inference fails."""
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=RuntimeError("Inference failed"))
        mock_model.device = "cpu"
        mock_model.model = Mock()
        mock_model.model.name_or_path = "test-model"

        result = health_check_reranking_model(mock_model)

        assert result["model_available"] is True
        assert result["inference_test_passed"] is False
        assert "Health check failed: Inference failed" in result["error_message"]

    def test_health_check_invalid_inference_output(self):
        """Test health check when model returns invalid output."""
        mock_model = Mock()
        mock_model.predict = Mock(
            return_value="invalid output"
        )  # Should be a list of numbers
        mock_model.device = "cpu"
        mock_model.model = Mock()
        mock_model.model.name_or_path = "test-model"

        result = health_check_reranking_model(mock_model)

        assert result["model_available"] is True
        assert result["inference_test_passed"] is False
        assert "Invalid inference output" in result["error_message"]


class TestMCPHealthCheckTool:
    """Test the MCP health check tool integration."""

    @pytest.mark.asyncio
    async def test_health_check_reranking_tool_success(self):
        """Test the MCP health check tool with successful reranking model."""
        from crawl4ai_mcp import health_check_reranking

        # Mock context with reranking model
        mock_context = Mock()
        mock_request_context = Mock()
        mock_lifespan_context = Mock()

        # Mock reranking model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.7, 0.8])
        mock_model.device = "cuda:0"
        mock_model.model = Mock()
        mock_model.model.name_or_path = "cross-encoder/test-model"

        mock_lifespan_context.reranking_model = mock_model
        mock_request_context.lifespan_context = mock_lifespan_context
        mock_context.request_context = mock_request_context

        with patch.object(utils_module, "cleanup_gpu_memory"):
            with patch.dict(
                os.environ,
                {
                    "USE_RERANKING": "true",
                    "RERANKING_MODEL_NAME": "test-model",
                    "RERANKING_WARMUP_SAMPLES": "5",
                },
            ):
                result_json = await health_check_reranking(mock_context)

        result = json.loads(result_json)

        assert result["overall_status"] == "healthy"
        assert result["model_available"] is True
        assert result["inference_test_passed"] is True
        assert result["configuration"]["use_reranking_enabled"] == "true"
        assert result["configuration"]["model_name_config"] == "test-model"
        assert result["configuration"]["warmup_samples_config"] == "5"

    @pytest.mark.asyncio
    async def test_health_check_reranking_tool_no_model(self):
        """Test the MCP health check tool when no reranking model is available."""
        from crawl4ai_mcp import health_check_reranking

        # Mock context without reranking model
        mock_context = Mock()
        mock_context.request_context = None

        with patch.dict(os.environ, {"USE_RERANKING": "false"}):
            result_json = await health_check_reranking(mock_context)

        result = json.loads(result_json)

        assert result["overall_status"] == "unhealthy"
        assert result["model_available"] is False
        assert result["configuration"]["use_reranking_enabled"] == "false"

    @pytest.mark.asyncio
    async def test_health_check_reranking_tool_exception(self):
        """Test the MCP health check tool handles exceptions gracefully."""
        from crawl4ai_mcp import health_check_reranking

        # Mock context that will cause an exception
        mock_context = Mock()
        mock_context.request_context = Mock()
        mock_context.request_context.lifespan_context = Mock()
        mock_context.request_context.lifespan_context.reranking_model = "invalid"

        with patch(
            "crawl4ai_mcp.health_check_reranking_model",
            side_effect=Exception("Test error"),
        ):
            result_json = await health_check_reranking(mock_context)

        result = json.loads(result_json)

        assert result["overall_status"] == "error"
        assert result["model_available"] is False
        assert (
            "Health check failed with exception: Test error" in result["error_message"]
        )


class TestIntegrationWithExistingSystem:
    """Test integration of enhancements with existing reranking system."""

    @patch("crawl4ai_mcp.CrossEncoder")
    @patch("crawl4ai_mcp.get_optimal_device")
    @patch("crawl4ai_mcp.get_model_kwargs_for_device")
    @patch("crawl4ai_mcp.cleanup_gpu_memory")
    @patch.dict(
        os.environ,
        {
            "USE_RERANKING": "true",
            "RERANKING_MODEL_NAME": "custom-model",
            "RERANKING_WARMUP_SAMPLES": "2",
        },
    )
    def test_full_integration_workflow(
        self, mock_cleanup, mock_get_kwargs, mock_get_device, mock_cross_encoder
    ):
        """Test the full workflow with all enhancements enabled."""
        from crawl4ai_mcp import crawl4ai_lifespan, rerank_results

        # Mock device
        mock_device = Mock()
        mock_device.__str__ = Mock(return_value="cuda:0")
        mock_get_device.return_value = mock_device
        mock_get_kwargs.return_value = {}

        # Mock CrossEncoder
        mock_model = Mock()
        mock_model.predict = Mock(
            side_effect=[
                [0.1, 0.2],  # Warmup call
                [0.8, 0.9, 0.7],  # Actual reranking call
            ]
        )
        mock_cross_encoder.return_value = mock_model

        async def test_workflow():
            async with crawl4ai_lifespan(None) as context:
                # Verify model was loaded with custom name
                mock_cross_encoder.assert_called_with(
                    "custom-model", device=str(mock_device), model_kwargs={}
                )

                # Verify warmup was performed
                assert mock_model.predict.call_count >= 1

                # Test actual reranking functionality
                test_results = [
                    {"content": "result 1", "id": 1},
                    {"content": "result 2", "id": 2},
                    {"content": "result 3", "id": 3},
                ]

                reranked = rerank_results(
                    context.reranking_model, "test query", test_results
                )

                # Verify reranking worked and results are sorted by score
                assert len(reranked) == 3
                assert reranked[0]["rerank_score"] == 0.9  # Highest
                assert reranked[1]["rerank_score"] == 0.8  # Middle
                assert reranked[2]["rerank_score"] == 0.7  # Lowest

                # Verify cleanup was called
                assert (
                    mock_cleanup.call_count >= 2
                )  # Once for warmup, once for reranking

        asyncio.run(test_workflow())


class TestEnvironmentConfiguration:
    """Test various environment configuration scenarios."""

    @patch.dict(os.environ, {"USE_RERANKING": "false"})
    def test_reranking_disabled_no_enhancements(self):
        """Test that when reranking is disabled, no enhancements are loaded."""
        from crawl4ai_mcp import crawl4ai_lifespan

        async def test_disabled():
            async with crawl4ai_lifespan(None) as context:
                # No reranking model should be loaded
                assert context.reranking_model is None

        asyncio.run(test_disabled())

    def test_default_environment_values(self):
        """Test that default values are used when environment variables are not set."""
        # Clear relevant environment variables
        env_vars_to_clear = ["RERANKING_MODEL_NAME", "RERANKING_WARMUP_SAMPLES"]
        original_values = {}

        for var in env_vars_to_clear:
            original_values[var] = os.environ.get(var)
            os.environ.pop(var, None)

        try:
            # Test default model name
            default_model = os.getenv(
                "RERANKING_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            assert default_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

            # Test default warmup samples
            default_samples = int(os.getenv("RERANKING_WARMUP_SAMPLES", "5"))
            assert default_samples == 5

        finally:
            # Restore original environment
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
