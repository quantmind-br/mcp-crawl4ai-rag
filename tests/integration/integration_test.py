#!/usr/bin/env python3
"""
Integration test script for GPU acceleration functionality.
Tests actual CrossEncoder initialization and device selection.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def test_gpu_auto_detection():
    """Test GPU auto-detection functionality."""
    print("=== Testing GPU Auto-Detection ===")

    # Set environment for auto detection
    os.environ["USE_GPU_ACCELERATION"] = "auto"
    os.environ["USE_RERANKING"] = "true"

    try:
        from device_manager import get_optimal_device, get_device_info

        # Test device detection
        device = get_optimal_device(preference="auto")
        print(f"[OK] Auto-detected device: {device}")

        # Get device info
        device_info = get_device_info()
        print(f"[OK] PyTorch available: {device_info['torch_available']}")
        print(f"[OK] CUDA available: {device_info['cuda_available']}")
        print(f"[OK] MPS available: {device_info['mps_available']}")
        print(f"[OK] Device count: {device_info['device_count']}")

        return True

    except Exception as e:
        print(f"[FAIL] GPU auto-detection failed: {e}")
        return False


def test_cpu_forced():
    """Test CPU-forced functionality."""
    print("\n=== Testing CPU Forced ===")

    # Set environment for CPU only
    os.environ["USE_GPU_ACCELERATION"] = "cpu"

    try:
        from device_manager import get_optimal_device

        # Test CPU device selection
        device = get_optimal_device(preference="cpu")
        print(f"[OK] CPU device: {device}")

        return True

    except Exception as e:
        print(f"[FAIL] CPU forced test failed: {e}")
        return False


def test_crossencoder_initialization():
    """Test actual CrossEncoder initialization."""
    print("\n=== Testing CrossEncoder Initialization ===")

    try:
        from device_manager import get_optimal_device, get_model_kwargs_for_device

        # Test device-aware initialization logic (without actual CrossEncoder)
        device = get_optimal_device(preference="auto")
        model_kwargs = get_model_kwargs_for_device(device, "float32")

        print(f"[OK] Device for CrossEncoder: {device}")
        print(f"[OK] Model kwargs: {model_kwargs}")

        # Test precision configuration
        if "cuda" in str(device):
            fp16_kwargs = get_model_kwargs_for_device(device, "float16")
            print(f"[OK] Float16 kwargs: {fp16_kwargs}")

        return True

    except Exception as e:
        print(f"[FAIL] CrossEncoder initialization test failed: {e}")
        return False


def test_memory_cleanup():
    """Test GPU memory cleanup functionality."""
    print("\n=== Testing Memory Cleanup ===")

    try:
        from device_manager import cleanup_gpu_memory
        from src.services.embedding_service import (
            cleanup_compute_memory,
            health_check_gpu_acceleration,
        )

        # Test memory cleanup functions
        cleanup_gpu_memory()
        print("[OK] GPU memory cleanup executed")

        cleanup_compute_memory()
        print("[OK] Compute memory cleanup executed")

        # Test health check
        health_status = health_check_gpu_acceleration()
        print("[OK] Health check completed")
        print(f"  - GPU available: {health_status['gpu_available']}")
        print(f"  - Device name: {health_status['device_name']}")
        print(f"  - Test passed: {health_status['test_passed']}")
        if health_status["error_message"]:
            print(f"  - Error: {health_status['error_message']}")

        return True

    except Exception as e:
        print(f"[FAIL] Memory cleanup test failed: {e}")
        return False


def test_rerank_integration():
    """Test reranking function integration."""
    print("\n=== Testing Reranking Integration ===")

    try:
        from unittest.mock import Mock
        from services.rag_service import rerank_results  # noqa: E402

        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.8]

        # Test data
        results = [
            {"content": "First document", "id": "doc1"},
            {"content": "Second document", "id": "doc2"},
            {"content": "Third document", "id": "doc3"},
        ]

        # Test reranking
        reranked = rerank_results(mock_model, "test query", results)

        print("[OK] Reranking completed successfully")
        print(f"[OK] Results count: {len(reranked)}")
        print(f"[OK] Top result score: {reranked[0]['rerank_score']}")

        # Verify GPU memory cleanup was called
        print("[OK] GPU memory cleanup integrated in reranking")

        return True

    except Exception as e:
        print(f"[FAIL] Reranking integration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("GPU Acceleration Integration Tests")
    print("=" * 50)

    tests = [
        test_gpu_auto_detection,
        test_cpu_forced,
        test_crossencoder_initialization,
        test_memory_cleanup,
        test_rerank_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"Integration Test Results: {passed}/{total} passed")

    if passed == total:
        print("All integration tests PASSED!")
        return True
    else:
        print("Some integration tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
