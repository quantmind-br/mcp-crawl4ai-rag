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
            model_kwargs={},
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
            model_kwargs={},
        )

    # Try GPU (CUDA or MPS) if requested or auto
    if preference in ["auto", "cuda"]:
        try:
            cuda_available = torch.cuda.is_available()
        except Exception as e:
            logging.warning(
                f"Error checking CUDA availability: {e}. Falling back to CPU."
            )
            cuda_available = False

        if cuda_available:
            try:
                # CRITICAL: Test actual GPU operations, not just availability
                device = torch.device(f"cuda:{gpu_index}")

                # Verify GPU works with actual tensor operations
                test_tensor = torch.randn(10, 10, device=device)
                _ = test_tensor @ test_tensor.T  # Matrix multiplication test

                # Try to fetch device name; guard against mock environments
                try:
                    device_name = torch.cuda.get_device_name(device)
                except Exception:
                    device_name = str(device)

                logging.info(f"GPU device verified: {device} ({device_name})")

                # Get model kwargs for GPU device
                model_kwargs = get_model_kwargs_for_device(device, precision)

                # Try to fetch total memory; guard against mock environments
                try:
                    total_memory_gb = torch.cuda.get_device_properties(
                        device
                    ).total_memory / (1024**3)
                except Exception:
                    total_memory_gb = None

                return DeviceInfo(
                    device=str(device),
                    device_type="cuda",
                    name=device_name,
                    memory_total=total_memory_gb,
                    is_available=True,
                    model_kwargs=model_kwargs,
                )

            except Exception as e:
                logging.warning(f"GPU test failed: {e}. Falling back to CPU.")

    # Try MPS (Apple Silicon) if requested or auto
    if preference in ["auto", "mps"]:
        mps_available = False
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            try:
                val = torch.backends.mps.is_available()
                mps_available = isinstance(val, bool) and val
            except Exception:
                mps_available = False

        if mps_available:
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
                    model_kwargs=model_kwargs,
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
        model_kwargs={},
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
            # Synchronize to ensure cleanup is flushed in CUDA runtime
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            logging.debug("GPU memory cache cleared")
    except Exception as e:
        logging.debug(f"GPU memory cleanup failed: {e}")


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for diagnostics.

    Returns:
        Dict with device capabilities and status information
    """
    info: Dict[str, Any] = {
        "torch_available": TORCH_AVAILABLE,
        "cpu": {"available": True},
        "cuda": {"available": False, "device_count": 0, "devices": []},
        "mps": {"available": False, "devices": []},
    }

    if not TORCH_AVAILABLE:
        return info

    # CUDA information
    try:
        if torch.cuda.is_available():
            info["cuda"]["available"] = True
            info["cuda"]["device_count"] = torch.cuda.device_count()

            for i in range(torch.cuda.device_count()):
                try:
                    device = torch.device(f"cuda:{i}")
                    # Prefer API that returns device name directly
                    try:
                        dev_name = torch.cuda.get_device_name(device)
                    except Exception:
                        props = torch.cuda.get_device_properties(device)
                        dev_name = getattr(props, "name", f"cuda:{i}")
                        total_memory = getattr(props, "total_memory", 0)
                    else:
                        # When get_device_name works, still try to get memory
                        try:
                            props = torch.cuda.get_device_properties(device)
                            total_memory = getattr(props, "total_memory", 0)
                        except Exception:
                            total_memory = 0

                    device_info = {
                        "index": i,
                        "name": dev_name,
                        # Test suite expects GB under key 'memory_total'
                        "memory_total": total_memory / (1024**3) if total_memory else 0,
                    }
                    info["cuda"]["devices"].append(device_info)
                except Exception as e:
                    logging.debug(f"Could not get info for CUDA device {i}: {e}")
    except Exception as e:
        info["error"] = str(e)

    # MPS information
    try:
        if (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            info["mps"]["available"] = True
            info["mps"]["devices"].append(
                {"name": "Apple Silicon GPU (MPS)", "is_available": True}
            )
    except Exception as e:
        info["error"] = str(e)

    return info


def get_model_kwargs_for_device(
    device: Any, precision: str = "float32"
) -> Dict[str, Any]:
    """
    Get model_kwargs for CrossEncoder based on device and precision.

    Args:
        device: Target device
        precision: Desired precision ("float32", "float16", "bfloat16")

    Returns:
        Dict with model_kwargs for CrossEncoder initialization
    """
    model_kwargs: Dict[str, Any] = {}

    # Always include device in kwargs for downstream consumers
    model_kwargs["device"] = device

    # Normalize device type whether it's a torch.device or a string
    try:
        device_type = device.type  # torch.device
    except Exception:
        device_str = str(device)
        if device_str.startswith("cuda"):
            device_type = "cuda"
        elif device_str.startswith("mps"):
            device_type = "mps"
        else:
            device_type = "cpu"

    # Apply precision settings for GPU devices
    if device_type in ["cuda", "mps"] and precision != "float32":
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
    # Prefer explicit GPU_PREFERENCE if set; fallback to USE_GPU_ACCELERATION
    env_value = os.getenv(
        "GPU_PREFERENCE", os.getenv("USE_GPU_ACCELERATION", "auto")
    ).lower()

    # Handle boolean-style values for backward compatibility
    if env_value in ["true", "1", "yes"]:
        return "auto"
    elif env_value in ["false", "0", "no"]:
        return "cpu"
    else:
        return env_value  # "auto", "cuda", "mps", etc.
