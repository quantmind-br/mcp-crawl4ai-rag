"""
Tests for device management functionality.

Tests device detection, fallback mechanisms, error handling, and GPU memory management
following the patterns established in conftest.py.
"""

import pytest
import os
from unittest.mock import Mock, patch

# Import the device manager functions
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from device_manager import (
    get_optimal_device,
    device_detection_with_fallback,
    cleanup_gpu_memory,
    get_device_info,
    get_model_kwargs_for_device,
    get_gpu_preference,
    DeviceConfig,
    DeviceInfo,
)


class TestDeviceDetection:
    """Test device detection and selection logic."""

    def test_cpu_device_forced(self):
        """CPU device always works when explicitly requested."""
        with patch("device_manager.TORCH_AVAILABLE", True):
            with patch("device_manager.torch") as mock_torch:
                mock_torch.device.return_value = Mock()
                mock_torch.device.return_value.__str__ = Mock(return_value="cpu")

                device = get_optimal_device(preference="cpu")
                assert str(device) == "cpu"

    def test_cpu_fallback_when_torch_unavailable(self):
        """Falls back to CPU when PyTorch is not available."""
        with patch("device_manager.TORCH_AVAILABLE", False):
            with patch("device_manager.torch", None):
                device = get_optimal_device(preference="auto")
                # Should return a mock device or handle gracefully
                assert device is not None

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_gpu_detection_when_cuda_available(self, mock_torch):
        """GPU detection when CUDA is available and working."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"

        # Mock device creation and tensor operations
        mock_device = Mock()
        mock_device.__str__ = Mock(return_value="cuda:0")
        mock_torch.device.return_value = mock_device

        # Mock successful tensor operations
        mock_tensor = Mock()
        mock_torch.randn.return_value = mock_tensor
        mock_tensor.__matmul__ = Mock(return_value=mock_tensor)

        device = get_optimal_device(preference="cuda")

        # Verify CUDA operations were tested
        mock_torch.randn.assert_called_once()
        assert str(device) == "cuda:0"

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_fallback_to_cpu_when_gpu_fails(self, mock_torch):
        """Fallback to CPU when GPU operations fail."""
        # Mock CUDA availability but operations fail
        mock_torch.cuda.is_available.return_value = True
        mock_torch.randn.side_effect = RuntimeError("GPU operation failed")

        # Mock CPU device as fallback
        cpu_device = Mock()
        cpu_device.__str__ = Mock(return_value="cpu")

        def device_side_effect(device_str):
            if "cuda" in device_str:
                raise RuntimeError("CUDA device failed")
            return cpu_device

        mock_torch.device.side_effect = device_side_effect

        device = get_optimal_device(preference="auto")
        assert str(device) == "cpu"

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_mps_detection_when_available(self, mock_torch):
        """MPS detection on Apple Silicon when available."""
        # Mock MPS availability
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        # Mock MPS device and operations
        mock_device = Mock()
        mock_device.__str__ = Mock(return_value="mps")
        mock_torch.device.return_value = mock_device

        mock_tensor = Mock()
        mock_torch.randn.return_value = mock_tensor
        mock_tensor.sum.return_value = mock_tensor

        device = get_optimal_device(preference="mps")
        assert str(device) == "mps"


class TestDeviceConfiguration:
    """Test device configuration and environment variable handling."""

    def test_device_config_creation(self):
        """DeviceConfig dataclass creation works correctly."""
        config = DeviceConfig(
            device_type="cuda", device_index=0, precision="float16", memory_fraction=0.8
        )

        assert config.device_type == "cuda"
        assert config.device_index == 0
        assert config.precision == "float16"
        assert config.memory_fraction == 0.8

    def test_device_info_creation(self):
        """DeviceInfo dataclass creation works correctly."""
        info = DeviceInfo(
            device="cuda:0", name="Test GPU", memory_total=8.0, is_available=True
        )

        assert info.device == "cuda:0"
        assert info.name == "Test GPU"
        assert info.memory_total == 8.0
        assert info.is_available is True

    def test_gpu_preference_from_env(self):
        """GPU preference correctly reads from environment variables."""
        test_cases = [
            ("true", "auto"),
            ("false", "cpu"),
            ("auto", "auto"),
            ("cuda", "cuda"),
            ("mps", "mps"),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"USE_GPU_ACCELERATION": env_value}):
                preference = get_gpu_preference()
                assert preference == expected


class TestModelKwargs:
    """Test model_kwargs generation for different devices and precisions."""

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_float32_precision_no_kwargs(self, mock_torch):
        """Float32 precision returns empty model_kwargs."""
        mock_device = Mock()
        mock_device.type = "cuda"

        kwargs = get_model_kwargs_for_device(mock_device, "float32")
        assert kwargs == {}

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_float16_precision_on_gpu(self, mock_torch):
        """Float16 precision on GPU returns correct torch_dtype."""
        mock_device = Mock()
        mock_device.type = "cuda"
        mock_torch.float16 = "float16_value"

        kwargs = get_model_kwargs_for_device(mock_device, "float16")
        assert kwargs == {"torch_dtype": "float16_value"}

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_bfloat16_precision_on_gpu(self, mock_torch):
        """BFloat16 precision on GPU returns correct torch_dtype."""
        mock_device = Mock()
        mock_device.type = "cuda"
        mock_torch.bfloat16 = "bfloat16_value"

        kwargs = get_model_kwargs_for_device(mock_device, "bfloat16")
        assert kwargs == {"torch_dtype": "bfloat16_value"}

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_precision_on_cpu_ignored(self, mock_torch):
        """Precision settings ignored on CPU device."""
        mock_device = Mock()
        mock_device.type = "cpu"

        kwargs = get_model_kwargs_for_device(mock_device, "float16")
        assert kwargs == {}


class TestMemoryManagement:
    """Test GPU memory management and cleanup."""

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_cleanup_gpu_memory_when_available(self, mock_torch):
        """GPU memory cleanup called when CUDA available."""
        mock_torch.cuda.is_available.return_value = True

        cleanup_gpu_memory()

        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_cleanup_gpu_memory_when_unavailable(self, mock_torch):
        """GPU memory cleanup safe when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False

        # Should not raise exception
        cleanup_gpu_memory()

        mock_torch.cuda.empty_cache.assert_not_called()

    @patch("device_manager.TORCH_AVAILABLE", False)
    def test_cleanup_gpu_memory_no_torch(self):
        """GPU memory cleanup safe when PyTorch unavailable."""
        # Should not raise exception
        cleanup_gpu_memory()


class TestDeviceInfo:
    """Test comprehensive device information gathering."""

    @patch("device_manager.TORCH_AVAILABLE", False)
    def test_device_info_no_torch(self):
        """Device info when PyTorch not available."""
        info = get_device_info()

        expected = {
            "torch_available": False,
            "cuda_available": False,
            "mps_available": False,
            "device_count": 0,
            "devices": [],
        }

        assert info == expected

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_device_info_cuda_available(self, mock_torch):
        """Device info when CUDA is available."""
        # Mock CUDA availability, no MPS
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.current_device.return_value = 0

        # Mock no MPS availability
        mock_torch.backends.mps.is_available.return_value = False

        # Mock device properties for two GPUs
        mock_props = [
            Mock(name="GPU 1", total_memory=8 * 1024**3),
            Mock(name="GPU 2", total_memory=16 * 1024**3),
        ]
        mock_props[0].name = "GPU 1"
        mock_props[1].name = "GPU 2"
        mock_torch.cuda.get_device_properties.side_effect = mock_props
        mock_torch.cuda.memory_allocated.side_effect = [1024**3, 2 * 1024**3]

        info = get_device_info()

        assert info["torch_available"] is True
        assert info["cuda_available"] is True
        assert info["device_count"] == 2
        assert len(info["devices"]) == 2

        # Check first device info
        device_0 = info["devices"][0]
        assert device_0["index"] == 0
        assert device_0["name"] == "GPU 1"
        assert device_0["memory_total_gb"] == 8.0
        assert device_0["is_current"] is True

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_device_info_mps_available(self, mock_torch):
        """Device info when MPS is available."""
        # Mock MPS availability, no CUDA
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        info = get_device_info()

        assert info["torch_available"] is True
        assert info["cuda_available"] is False
        assert info["mps_available"] is True
        assert len(info["devices"]) == 1

        mps_device = info["devices"][0]
        assert mps_device["name"] == "Apple Silicon GPU (MPS)"
        assert mps_device["type"] == "mps"


class TestDeviceDetectionWithFallback:
    """Test comprehensive device detection with fallback strategy."""

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_device_detection_with_config(self, mock_torch):
        """Device detection with custom configuration."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value = Mock(
            total_memory=8 * 1024**3
        )

        # Mock device and tensor operations
        mock_device = Mock()
        mock_device.type = "cuda"
        mock_device.__str__ = Mock(return_value="cuda:0")
        mock_torch.device.return_value = mock_device

        mock_tensor = Mock()
        mock_torch.randn.return_value = mock_tensor
        mock_tensor.__matmul__ = Mock(return_value=mock_tensor)

        # Test with custom config
        config = DeviceConfig(
            device_type="cuda", device_index=0, precision="float16", memory_fraction=0.8
        )

        device, device_info = device_detection_with_fallback(config)

        assert str(device) == "cuda:0"
        assert device_info.name == "Test GPU"
        assert device_info.memory_total == 8.0
        assert device_info.is_available is True

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    @patch.dict(
        os.environ,
        {
            "USE_GPU_ACCELERATION": "auto",
            "GPU_DEVICE_INDEX": "1",
            "GPU_PRECISION": "float16",
            "GPU_MEMORY_FRACTION": "0.9",
        },
    )
    def test_device_detection_from_env(self, mock_torch):
        """Device detection using environment variables."""
        # Mock CPU fallback
        mock_torch.cuda.is_available.return_value = False
        cpu_device = Mock()
        cpu_device.type = "cpu"
        cpu_device.__str__ = Mock(return_value="cpu")
        mock_torch.device.return_value = cpu_device

        device, device_info = device_detection_with_fallback()

        assert str(device) == "cpu"
        assert device_info.name == "CPU"
        assert device_info.is_available is True


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_gpu_operations_exception_handling(self, mock_torch):
        """GPU operations exceptions are handled gracefully."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.randn.side_effect = RuntimeError("Out of memory")

        # Mock MPS not available
        mock_torch.backends.mps.is_available.return_value = False

        # Mock CPU device for fallback
        cpu_device = Mock()
        cpu_device.__str__ = Mock(return_value="cpu")
        mock_torch.device.return_value = cpu_device

        # Should not raise exception, should return CPU
        device = get_optimal_device(preference="auto")
        assert str(device) == "cpu"

    @patch("device_manager.TORCH_AVAILABLE", True)
    @patch("device_manager.torch")
    def test_device_info_partial_failure(self, mock_torch):
        """Device info gathering handles partial failures."""
        # Mock CUDA available but some operations fail, no MPS
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.backends.mps.is_available.return_value = False

        # First device works, second fails
        working_gpu_props = Mock(name="Working GPU", total_memory=8 * 1024**3)
        working_gpu_props.name = "Working GPU"
        mock_torch.cuda.get_device_properties.side_effect = [
            working_gpu_props,
            RuntimeError("Device error"),
        ]
        mock_torch.cuda.memory_allocated.side_effect = [
            1024**3,
            RuntimeError("Memory error"),
        ]
        mock_torch.cuda.current_device.return_value = 0

        info = get_device_info()

        # Should still return info for working device
        assert info["cuda_available"] is True
        assert len(info["devices"]) == 1  # Only successful device included
        assert info["devices"][0]["name"] == "Working GPU"

    def test_unknown_precision_warning(self):
        """Unknown precision values are handled with warning."""
        mock_device = Mock()
        mock_device.type = "cuda"

        with patch("device_manager.logging") as mock_logging:
            kwargs = get_model_kwargs_for_device(mock_device, "unknown_precision")

            # Should return empty dict and log warning
            assert kwargs == {}
            mock_logging.warning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
