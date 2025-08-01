# PyTorch CUDA Detection Patterns for Production

## Device Detection Best Practices

### Robust Device Selection
```python
import torch
import logging

def get_optimal_device():
    """
    Returns the optimal device for model inference.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

### Memory Management
```python
def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
```

### Error Handling Patterns
```python
def safe_model_to_device(model, device):
    """Safely move model to device with fallback"""
    try:
        return model.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.warning("GPU OOM. Falling back to CPU.")
            cleanup_gpu_memory()
            return model.to('cpu')
        else:
            raise e
```

## Key URLs
- PyTorch CUDA Documentation: https://pytorch.org/docs/stable/cuda.html
- Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- Device Management: https://pytorch.org/docs/stable/notes/cuda.html