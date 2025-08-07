"""
Device Management for GPU/CPU CrossEncoder Acceleration.

Provides robust device detection, GPU memory management, and graceful fallback
for CrossEncoder models. Follows production-ready patterns with actual GPU
testing beyond availability flags.
"""

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Suppress future warnings from PyTorch for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


@dataclass
class DeviceConfig:
    """Configuration for device selection and GPU settings."""

    device_type: str  # "cuda", "cpu", "mps"
    device_index: Optional[int]  # GPU index for multi-GPU systems
    precision: str  # "float32", "float16", "bfloat16"
    memory_fraction: float  # GPU memory fraction to use


@dataclass
class DeviceInfo:
    """Information about detected device capabilities."""

    device: str  # PyTorch device string
    device_type: str  # Device type for compatibility
    name: str  # Human-readable device name
    memory_total: Optional[float]  # Total memory in GB
    is_available: bool  # Whether device is truly available
    model_kwargs: Dict[str, Any]  # Model kwargs for CrossEncoder


def get_optimal_device(preference: str = "auto", gpu_index: int = 0) -> DeviceInfo:
    """
    Get optimal device for CrossEncoder model with robust detection.

    CRITICAL: Tests actual GPU operations beyond availability flags.
    Always provides CPU fallback for production reliability.

    Args:
        preference: Device preference - "auto", "cuda", "cpu", "mps"
        gpu_index: GPU index for multi-GPU systems (default: 0)

    Returns:
        DeviceInfo: Optimal device info for model initialization

    Raises:
        None - Always returns valid device with fallback to CPU
    """
    precision = os.getenv("GPU_PRECISION", "float32")
    
    if not TORCH_AVAILABLE:
        logging.warning("PyTorch not available. Using CPU.")
        return DeviceInfo(
            device="cpu",
            device_type="cpu",
            name="CPU (PyTorch unavailable)",
            memory_total=None,
            is_available=True,
            model_kwargs={}
        )

    # Force CPU if requested
    if preference == "cpu":
        logging.info("CPU device forced by preference")
        return DeviceInfo(
            device="cpu",
            device_type="cpu", 
            name="CPU",
            memory_total=None,
            is_available=True,
            model_kwargs={}
        )

    # Try GPU (CUDA or MPS) if requested or auto
    if preference in ["auto", "cuda"] and torch.cuda.is_available():
        try:
            # CRITICAL: Test actual GPU operations, not just availability
            device = torch.device(f"cuda:{gpu_index}")

            # Verify GPU works with actual tensor operations
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor.T  # Matrix multiplication test

            logging.info(
                f"GPU device verified: {device} ({torch.cuda.get_device_name(device)})"
            )
            
            # Get model kwargs for GPU device
            model_kwargs = get_model_kwargs_for_device(device, precision)
            
            return DeviceInfo(
                device=str(device),
                device_type="cuda",
                name=torch.cuda.get_device_name(device),
                memory_total=torch.cuda.get_device_properties(device).total_memory / (1024**3),
                is_available=True,
                model_kwargs=model_kwargs
            )

        except Exception as e:
            logging.warning(f"GPU test failed: {e}. Falling back to CPU.")

    # Try MPS (Apple Silicon) if requested or auto
    if (
        preference in ["auto", "mps"]
        and hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        try:
            device = torch.device("mps")

            # Test MPS with simple operation
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor.sum()

            logging.info(f"MPS device verified: {device}")
            
            # Get model kwargs for MPS device  
            model_kwargs = get_model_kwargs_for_device(device, precision)
            
            return DeviceInfo(
                device=str(device),
                device_type="mps",
                name="Apple Silicon GPU (MPS)",
                memory_total=None,
                is_available=True,
                model_kwargs=model_kwargs
            )

        except Exception as e:
            logging.warning(f"MPS test failed: {e}. Falling back to CPU.")

    # FALLBACK: Always return CPU as last resort
    logging.info("Using CPU device (fallback)")
    return DeviceInfo(
        device="cpu",
        device_type="cpu",
        name="CPU",
        memory_total=None,
        is_available=True,
        model_kwargs={}
    )


def device_detection_with_fallback(
    config: Optional[DeviceConfig] = None,
) -> DeviceInfo:
    """
    Comprehensive device detection with fallback strategy.

    Args:
        config: Device configuration preferences

    Returns:
        DeviceInfo with detected device and metadata
    """
    if config is None:
        # Default configuration from environment variables
        config = DeviceConfig(
            device_type=os.getenv("USE_GPU_ACCELERATION", "auto"),
            device_index=int(os.getenv("GPU_DEVICE_INDEX", "0")),
            precision=os.getenv("GPU_PRECISION", "float32"),
            memory_fraction=float(os.getenv("GPU_MEMORY_FRACTION", "0.8")),
        )

    # Get optimal device info
    return get_optimal_device(config.device_type, config.device_index)


def cleanup_gpu_memory() -> None:
    """
    Clean up GPU memory to prevent OOM in long-running processes.

    CRITICAL: Required for CrossEncoder in production environments.
    Safe to call even when GPU is not available.
    """
    if not TORCH_AVAILABLE:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.debug("GPU memory cache cleared")
    except Exception as e:
        logging.debug(f"GPU memory cleanup failed: {e}")


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for diagnostics.

    Returns:
        Dict with device capabilities and status information
    """
    info = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "devices": [],
    }

    if not TORCH_AVAILABLE:
        return info

    # CUDA information
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["device_count"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            try:
                device = torch.device(f"cuda:{i}")
                props = torch.cuda.get_device_properties(device)
                device_info = {
                    "index": i,
                    "name": props.name,
                    "memory_total_gb": props.total_memory / (1024**3),
                    "memory_allocated_gb": torch.cuda.memory_allocated(device)
                    / (1024**3),
                    "is_current": i == torch.cuda.current_device(),
                }
                info["devices"].append(device_info)
            except Exception as e:
                logging.debug(f"Could not get info for CUDA device {i}: {e}")

    # MPS information
    if (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        info["mps_available"] = True
        info["devices"].append(
            {
                "name": "Apple Silicon GPU (MPS)",
                "type": "mps",
                "memory_total_gb": None,  # Not available for MPS
                "is_available": True,
            }
        )

    return info


def get_model_kwargs_for_device(
    device: torch.device, precision: str = "float32"
) -> Dict[str, Any]:
    """
    Get model_kwargs for CrossEncoder based on device and precision.

    Args:
        device: Target device
        precision: Desired precision ("float32", "float16", "bfloat16")

    Returns:
        Dict with model_kwargs for CrossEncoder initialization
    """
    model_kwargs = {}

    # Apply precision settings for GPU devices
    if device.type in ["cuda", "mps"] and precision != "float32":
        if precision == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif precision == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            logging.warning(f"Unknown precision '{precision}', using float32")

    return model_kwargs


# Environment variable pattern compatibility
def get_gpu_preference() -> str:
    """
    Get GPU preference from environment variables.

    Returns:
        GPU preference string compatible with get_optimal_device()
    """
    env_value = os.getenv("USE_GPU_ACCELERATION", "auto").lower()

    # Handle boolean-style values for backward compatibility
    if env_value in ["true", "1", "yes"]:
        return "auto"
    elif env_value in ["false", "0", "no"]:
        return "cpu"
    else:
        return env_value  # "auto", "cuda", "mps", etc.
