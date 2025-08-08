"""
Testes para as funcionalidades de device_manager.py.

Este módulo contém testes para o gerenciamento de dispositivos GPU/CPU, incluindo:
- Detecção de dispositivos ótimos
- Configuração de GPU/CPU
- Gerenciamento de memória
- Fallback para CPU
- Informações de dispositivo
"""

import os
from unittest.mock import Mock, patch

# Importa as funções do device_manager
try:
    from src.device_manager import (
        get_optimal_device,
        device_detection_with_fallback,
        cleanup_gpu_memory,
        get_device_info,
        get_model_kwargs_for_device,
        get_gpu_preference,
        DeviceConfig,
        DeviceInfo,
    )
except ImportError:
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


class TestDeviceManagerBasicFunctions:
    """Testes para funções básicas do device_manager."""

    def test_get_gpu_preference_default(self):
        """Testa obtenção da preferência de GPU padrão."""
        with patch.dict(os.environ, {}, clear=True):
            preference = get_gpu_preference()
            assert preference == "auto"

    def test_get_gpu_preference_custom(self):
        """Testa obtenção da preferência de GPU customizada."""
        with patch.dict(os.environ, {"GPU_PREFERENCE": "cpu"}):
            preference = get_gpu_preference()
            assert preference == "cpu"

    def test_get_gpu_preference_cuda(self):
        """Testa obtenção da preferência de GPU CUDA."""
        with patch.dict(os.environ, {"GPU_PREFERENCE": "cuda"}):
            preference = get_gpu_preference()
            assert preference == "cuda"

    def test_device_config_creation(self):
        """Testa criação de configuração de dispositivo."""
        config = DeviceConfig(
            device_type="cuda", device_index=0, precision="float32", memory_fraction=0.8
        )

        assert config.device_type == "cuda"
        assert config.device_index == 0
        assert config.precision == "float32"
        assert config.memory_fraction == 0.8

    def test_device_info_creation(self):
        """Testa criação de informações de dispositivo."""
        info = DeviceInfo(
            device="cuda:0",
            device_type="cuda",
            name="NVIDIA GeForce RTX 3080",
            memory_total=10.0,
            is_available=True,
            model_kwargs={"device": "cuda:0"},
        )

        assert info.device == "cuda:0"
        assert info.device_type == "cuda"
        assert info.name == "NVIDIA GeForce RTX 3080"
        assert info.memory_total == 10.0
        assert info.is_available is True
        assert "device" in info.model_kwargs


class TestDeviceManagerDetection:
    """Testes para detecção de dispositivos."""

    @patch("src.device_manager.TORCH_AVAILABLE", False)
    def test_get_optimal_device_no_torch(self):
        """Testa detecção quando PyTorch não está disponível."""
        result = get_optimal_device()

        assert result.device == "cpu"
        assert result.device_type == "cpu"
        assert "PyTorch unavailable" in result.name
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_cpu_preference(self, mock_torch):
        """Testa detecção com preferência de CPU."""
        result = get_optimal_device(preference="cpu")

        assert result.device == "cpu"
        assert result.device_type == "cpu"
        assert result.name == "CPU"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_cuda_available(self, mock_torch):
        """Testa detecção quando CUDA está disponível."""
        # Configura mock para CUDA disponível
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda:0"
        mock_torch.randn.return_value = Mock()
        mock_tensor = Mock()
        mock_tensor.__matmul__ = Mock(return_value=Mock())
        mock_torch.randn.return_value = mock_tensor

        result = get_optimal_device(preference="cuda")

        assert result.device == "cuda:0"
        assert result.device_type == "cuda"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_cuda_unavailable(self, mock_torch):
        """Testa detecção quando CUDA não está disponível."""
        # Configura mock para CUDA indisponível
        mock_torch.cuda.is_available.return_value = False

        result = get_optimal_device(preference="cuda")

        assert result.device == "cpu"
        assert result.device_type == "cpu"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_cuda_error(self, mock_torch):
        """Testa detecção quando CUDA gera erro."""
        # Configura mock para CUDA disponível mas com erro
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.side_effect = Exception("CUDA error")

        result = get_optimal_device(preference="cuda")

        assert result.device == "cpu"
        assert result.device_type == "cpu"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_mps_available(self, mock_torch):
        """Testa detecção quando MPS está disponível."""
        # Configura mock para MPS disponível
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device.return_value = "mps"
        mock_torch.randn.return_value = Mock()
        mock_tensor = Mock()
        mock_tensor.__matmul__ = Mock(return_value=Mock())
        mock_torch.randn.return_value = mock_tensor

        result = get_optimal_device(preference="auto")

        assert result.device == "mps"
        assert result.device_type == "mps"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_mps_unavailable(self, mock_torch):
        """Testa detecção quando MPS não está disponível."""
        # Configura mock para MPS indisponível
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        result = get_optimal_device(preference="auto")

        assert result.device == "cpu"
        assert result.device_type == "cpu"
        assert result.is_available is True


class TestDeviceManagerConfiguration:
    """Testes para configuração de dispositivos."""

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_device_detection_with_fallback_default(self, mock_torch):
        """Testa detecção de dispositivo com fallback padrão."""
        # Configura mock para CPU
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        result = device_detection_with_fallback()

        assert result.device == "cpu"
        assert result.device_type == "cpu"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_device_detection_with_fallback_custom_config(self, mock_torch):
        """Testa detecção de dispositivo com configuração customizada."""
        # Configura mock para CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda:1"
        mock_torch.randn.return_value = Mock()
        mock_tensor = Mock()
        mock_tensor.__matmul__ = Mock(return_value=Mock())
        mock_torch.randn.return_value = mock_tensor

        config = DeviceConfig(
            device_type="cuda", device_index=1, precision="float16", memory_fraction=0.7
        )

        result = device_detection_with_fallback(config)

        assert result.device == "cuda:1"
        assert result.device_type == "cuda"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_model_kwargs_for_device_cuda(self, mock_torch):
        """Testa obtenção de kwargs do modelo para CUDA."""
        mock_device = Mock()
        mock_device.type = "cuda"

        kwargs = get_model_kwargs_for_device(mock_device, precision="float16")

        assert "device" in kwargs
        assert kwargs["device"] == mock_device

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_model_kwargs_for_device_cpu(self, mock_torch):
        """Testa obtenção de kwargs do modelo para CPU."""
        mock_device = Mock()
        mock_device.type = "cpu"

        kwargs = get_model_kwargs_for_device(mock_device, precision="float32")

        assert "device" in kwargs
        assert kwargs["device"] == mock_device


class TestDeviceManagerMemoryManagement:
    """Testes para gerenciamento de memória."""

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_cleanup_gpu_memory_cuda_available(self, mock_torch):
        """Testa limpeza de memória GPU quando CUDA está disponível."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()

        cleanup_gpu_memory()

        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.synchronize.assert_called_once()

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_cleanup_gpu_memory_cuda_unavailable(self, mock_torch):
        """Testa limpeza de memória GPU quando CUDA não está disponível."""
        mock_torch.cuda.is_available.return_value = False

        # Não deve gerar erro
        cleanup_gpu_memory()

    @patch("src.device_manager.TORCH_AVAILABLE", False)
    def test_cleanup_gpu_memory_no_torch(self):
        """Testa limpeza de memória GPU quando PyTorch não está disponível."""
        # Não deve gerar erro
        cleanup_gpu_memory()


class TestDeviceManagerInformation:
    """Testes para informações de dispositivo."""

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_device_info_cuda(self, mock_torch):
        """Testa obtenção de informações de dispositivo CUDA."""
        # Configura mock para CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_torch.cuda.get_device_properties.return_value = Mock(
            total_memory=10737418240  # 10GB em bytes
        )
        mock_torch.device.return_value = "cuda:0"
        mock_torch.randn.return_value = Mock()
        mock_tensor = Mock()
        mock_tensor.__matmul__ = Mock(return_value=Mock())
        mock_torch.randn.return_value = mock_tensor

        info = get_device_info()

        assert "cuda" in info
        assert info["cuda"]["available"] is True
        assert info["cuda"]["device_count"] == 2
        assert "NVIDIA GeForce RTX 3080" in info["cuda"]["devices"][0]["name"]
        assert info["cuda"]["devices"][0]["memory_total"] == 10.0

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_device_info_cpu_only(self, mock_torch):
        """Testa obtenção de informações de dispositivo apenas CPU."""
        # Configura mock para apenas CPU
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        info = get_device_info()

        assert "cpu" in info
        assert info["cpu"]["available"] is True
        assert "cuda" in info
        assert info["cuda"]["available"] is False

    @patch("src.device_manager.TORCH_AVAILABLE", False)
    def test_get_device_info_no_torch(self):
        """Testa obtenção de informações de dispositivo sem PyTorch."""
        info = get_device_info()

        assert "cpu" in info
        assert info["cpu"]["available"] is True
        assert "cuda" in info
        assert info["cuda"]["available"] is False
        assert "torch_available" in info
        assert info["torch_available"] is False

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_device_info_mps_available(self, mock_torch):
        """Testa obtenção de informações de dispositivo com MPS."""
        # Configura mock para MPS
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device.return_value = "mps"
        mock_torch.randn.return_value = Mock()
        mock_tensor = Mock()
        mock_tensor.__matmul__ = Mock(return_value=Mock())
        mock_torch.randn.return_value = mock_tensor

        info = get_device_info()

        assert "mps" in info
        assert info["mps"]["available"] is True


class TestDeviceManagerEdgeCases:
    """Testes para casos extremos do device_manager."""

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_multi_gpu(self, mock_torch):
        """Testa detecção de dispositivo com múltiplas GPUs."""
        # Configura mock para múltiplas GPUs
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_torch.device.return_value = "cuda:2"
        mock_torch.randn.return_value = Mock()
        mock_tensor = Mock()
        mock_tensor.__matmul__ = Mock(return_value=Mock())
        mock_torch.randn.return_value = mock_tensor

        result = get_optimal_device(gpu_index=2)

        assert result.device == "cuda:2"
        assert result.device_type == "cuda"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_invalid_gpu_index(self, mock_torch):
        """Testa detecção de dispositivo com índice de GPU inválido."""
        # Configura mock para GPU indisponível
        mock_torch.cuda.is_available.return_value = False

        result = get_optimal_device(gpu_index=999)

        assert result.device == "cpu"
        assert result.device_type == "cpu"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_optimal_device_different_precisions(self, mock_torch):
        """Testa detecção de dispositivo com diferentes precisões."""
        # Configura mock para CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda:0"
        mock_torch.randn.return_value = Mock()
        mock_tensor = Mock()
        mock_tensor.__matmul__ = Mock(return_value=Mock())
        mock_torch.randn.return_value = mock_tensor

        # Testa diferentes precisões
        for precision in ["float32", "float16", "bfloat16"]:
            with patch.dict(os.environ, {"GPU_PRECISION": precision}):
                result = get_optimal_device()
                assert result.device == "cuda:0"
                assert result.device_type == "cuda"

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_device_detection_with_fallback_error_handling(self, mock_torch):
        """Testa tratamento de erro na detecção de dispositivo."""
        # Configura mock para gerar erro
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")

        result = device_detection_with_fallback()

        assert result.device == "cpu"
        assert result.device_type == "cpu"
        assert result.is_available is True

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_get_device_info_error_handling(self, mock_torch):
        """Testa tratamento de erro na obtenção de informações de dispositivo."""
        # Configura mock para gerar erro
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")

        info = get_device_info()

        assert "cpu" in info
        assert info["cpu"]["available"] is True
        assert "error" in info


class TestDeviceManagerIntegration:
    """Testes de integração para device_manager."""

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_complete_device_workflow(self, mock_torch):
        """Testa workflow completo de gerenciamento de dispositivo."""
        # Configura mock para CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_torch.cuda.get_device_properties.return_value = Mock(
            total_memory=10737418240
        )
        mock_torch.device.return_value = "cuda:0"
        mock_torch.randn.return_value = Mock()
        mock_tensor = Mock()
        mock_tensor.__matmul__ = Mock(return_value=Mock())
        mock_torch.randn.return_value = mock_tensor
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()

        # Testa obtenção de dispositivo ótimo
        device_info = get_optimal_device()
        assert device_info.device == "cuda:0"
        assert device_info.device_type == "cuda"

        # Testa obtenção de informações de dispositivo
        device_info_dict = get_device_info()
        assert device_info_dict["cuda"]["available"] is True

        # Testa limpeza de memória
        cleanup_gpu_memory()
        mock_torch.cuda.empty_cache.assert_called_once()

        # Testa obtenção de kwargs do modelo
        kwargs = get_model_kwargs_for_device(device_info.device, precision="float32")
        assert "device" in kwargs

    @patch("src.device_manager.TORCH_AVAILABLE", False)
    def test_complete_device_workflow_no_torch(self):
        """Testa workflow completo sem PyTorch."""
        # Testa obtenção de dispositivo ótimo
        device_info = get_optimal_device()
        assert device_info.device == "cpu"
        assert device_info.device_type == "cpu"

        # Testa obtenção de informações de dispositivo
        device_info_dict = get_device_info()
        assert device_info_dict["cpu"]["available"] is True
        assert device_info_dict["torch_available"] is False

        # Testa limpeza de memória (não deve gerar erro)
        cleanup_gpu_memory()

    @patch("src.device_manager.TORCH_AVAILABLE", True)
    @patch("src.device_manager.torch")
    def test_device_configuration_workflow(self, mock_torch):
        """Testa workflow de configuração de dispositivo."""
        # Configura mock para CPU
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Testa configuração customizada
        config = DeviceConfig(
            device_type="cpu",
            device_index=None,
            precision="float32",
            memory_fraction=1.0,
        )

        device_info = device_detection_with_fallback(config)
        assert device_info.device == "cpu"
        assert device_info.device_type == "cpu"

        # Testa obtenção de kwargs do modelo
        kwargs = get_model_kwargs_for_device(device_info.device, precision="float32")
        assert "device" in kwargs
        assert kwargs["device"] == "cpu"

