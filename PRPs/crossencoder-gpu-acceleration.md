# CrossEncoder GPU Acceleration Feature PRP

## Goal

Enable GPU acceleration for the CrossEncoder model used in reranking functionality. When CUDA drivers are detected, the model should automatically utilize GPU instead of CPU, with graceful fallback to CPU when GPU is unavailable or encounters errors.

## Why

- **Performance Improvement**: GPU acceleration provides 5-10x faster inference for cross-encoder reranking
- **Resource Optimization**: Better utilization of available hardware, especially for heavy workloads
- **User Experience**: Significantly faster response times for search operations with reranking enabled
- **Scalability**: Improved throughput for high-volume search scenarios
- **Cost Efficiency**: Better performance per dollar on GPU-enabled infrastructure

## What

Modify the existing CrossEncoder model initialization to:
1. Automatically detect CUDA availability and compatible GPU hardware
2. Initialize CrossEncoder on GPU when available, CPU as fallback
3. Handle GPU memory issues gracefully (OOM, unavailable GPU)
4. Provide configuration options for GPU preferences
5. Maintain full backward compatibility with CPU-only environments
6. Add proper error handling and memory management

### Success Criteria

- [ ] CrossEncoder model automatically uses GPU when CUDA is available
- [ ] Graceful fallback to CPU when GPU is unavailable or encounters errors
- [ ] Configurable GPU preferences via environment variables
- [ ] GPU memory management with cleanup mechanisms
- [ ] 5-10x performance improvement on GPU-enabled systems
- [ ] Zero breaking changes for existing CPU-only deployments
- [ ] Comprehensive test coverage for both GPU and CPU scenarios
- [ ] Proper logging for device selection and fallback scenarios

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://pytorch.org/docs/stable/cuda.html
  why: CUDA detection and device management patterns

- url: https://sbert.net/docs/cross_encoder/
  why: Official CrossEncoder documentation and device parameter usage

- url: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
  why: GPU memory management best practices and OOM handling

- docfile: PRPs/ai_docs/pytorch-cuda-detection-patterns.md
  why: Production-ready CUDA detection and error handling patterns

- docfile: PRPs/ai_docs/crossencoder-gpu-optimization.md
  why: CrossEncoder GPU initialization and performance optimization patterns

- file: src/crawl4ai_mcp.py:170
  why: Current CrossEncoder initialization location and usage pattern

- file: src/crawl4ai_mcp.py:243-279
  why: rerank_results function that uses the CrossEncoder model

- file: .env.example:30
  why: Existing USE_RERANKING configuration pattern to follow
```

### Current Codebase Tree

```bash
mcp-crawl4ai-rag/
├── src/
│   ├── crawl4ai_mcp.py          # Main MCP server with CrossEncoder usage
│   ├── utils.py                 # Utility functions
│   ├── qdrant_wrapper.py        # Database client wrapper
│   └── __init__.py
├── tests/
│   ├── conftest.py              # Test fixtures and setup
│   ├── test_mcp_server.py       # MCP server integration tests
│   └── test_*.py                # Other test files
├── .env.example                 # Environment configuration template
├── pyproject.toml              # Dependencies including PyTorch
└── README.md
```

### Desired Codebase Tree

```bash
mcp-crawl4ai-rag/
├── src/
│   ├── crawl4ai_mcp.py          # Enhanced with GPU device detection
│   ├── utils.py                 # Add GPU utility functions
│   ├── device_manager.py        # NEW: Device detection and management
│   └── ...
├── tests/
│   ├── test_device_manager.py   # NEW: GPU detection tests
│   ├── test_gpu_integration.py  # NEW: GPU/CPU integration tests
│   └── ...
├── .env.example                 # Add GPU configuration options
└── ...
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: sentence-transformers CrossEncoder device parameter
# Must be specified during initialization, cannot be changed later
model = CrossEncoder(model_name, device="cuda")  # ✓ Correct
model.to("cuda")  # ❌ Not supported for CrossEncoder

# CRITICAL: PyTorch CUDA memory management
# Always clear cache after processing batches to prevent OOM
torch.cuda.empty_cache()  # Required for long-running processes

# CRITICAL: Device detection edge cases
# torch.cuda.is_available() can return True but initialization can fail
# Always test actual tensor operations for true GPU availability

# CRITICAL: CrossEncoder model_kwargs for precision
# Float16 requires model_kwargs during initialization
model_kwargs = {"torch_dtype": "float16"}
model = CrossEncoder(name, device="cuda", model_kwargs=model_kwargs)

# CRITICAL: Environment variable patterns in this codebase
# Use string comparison with "true"/"false" (see USE_RERANKING pattern)
gpu_enabled = os.getenv("USE_GPU_ACCELERATION", "auto") == "true"
```

## Implementation Blueprint

### Data Models and Structure

```python
# Device configuration and management structures
@dataclass
class DeviceConfig:
    device_type: str              # "cuda", "cpu", "mps"
    device_index: Optional[int]   # GPU index for multi-GPU systems
    precision: str                # "float32", "float16", "bfloat16"
    memory_fraction: float        # GPU memory fraction to use

@dataclass
class DeviceInfo:
    device: str                   # PyTorch device string
    name: str                     # Human-readable device name
    memory_total: Optional[float] # Total memory in GB
    is_available: bool            # Whether device is truly available
```

### List of Tasks (Implementation Order)

```yaml
Task 1:
CREATE src/device_manager.py:
  - IMPLEMENT get_optimal_device() function
  - IMPLEMENT device_detection_with_fallback() function
  - IMPLEMENT gpu_memory_cleanup() utility
  - FOLLOW pattern from: PRPs/ai_docs/pytorch-cuda-detection-patterns.md

Task 2:
MODIFY .env.example:
  - ADD USE_GPU_ACCELERATION configuration option
  - ADD GPU_PRECISION configuration option
  - ADD GPU_MEMORY_FRACTION configuration option
  - FOLLOW pattern from: existing USE_RERANKING configuration

Task 3:
MODIFY src/crawl4ai_mcp.py:
  - IMPORT device_manager utilities
  - FIND pattern: "reranking_model = CrossEncoder"
  - REPLACE with: device-aware initialization
  - PRESERVE existing error handling structure
  - KEEP backward compatibility intact

Task 4:
ENHANCE src/utils.py:
  - ADD get_device_info() diagnostic function
  - ADD memory monitoring utilities
  - INTEGRATE with existing utility patterns

Task 5:
CREATE tests/test_device_manager.py:
  - TEST device detection logic
  - TEST fallback mechanisms
  - TEST error handling scenarios
  - MIRROR pattern from: tests/conftest.py fixtures

Task 6:
CREATE tests/test_gpu_integration.py:
  - TEST CrossEncoder GPU initialization
  - TEST CPU fallback scenarios
  - TEST memory cleanup functionality
  - INTEGRATE with existing test infrastructure

Task 7:
UPDATE README.md:
  - ADD GPU acceleration documentation
  - ADD configuration examples
  - ADD troubleshooting section
  - FOLLOW existing documentation style
```

### Per Task Pseudocode

```python
# Task 1: Device Manager Implementation
def get_optimal_device(preference="auto", gpu_index=0):
    """
    PATTERN: Robust device detection with multiple fallbacks
    CRITICAL: Test actual GPU operations, not just availability flags
    """
    if preference == "cpu":
        return torch.device("cpu")
    
    if preference in ["auto", "cuda"] and torch.cuda.is_available():
        try:
            # GOTCHA: Test actual tensor operations to verify GPU works
            device = torch.device(f"cuda:{gpu_index}")
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor.T  # Verify operations work
            return device
        except Exception as e:
            logging.warning(f"GPU test failed: {e}. Falling back to CPU.")
    
    # FALLBACK: Always return CPU as last resort
    return torch.device("cpu")

# Task 3: CrossEncoder Enhancement
def initialize_reranking_model():
    """
    PATTERN: Follow existing error handling structure
    CRITICAL: Maintain exact same fallback behavior as current code
    """
    if os.getenv("USE_RERANKING", "false") != "true":
        return None
    
    try:
        # NEW: Device detection with configuration
        device = get_optimal_device(
            preference=os.getenv("USE_GPU_ACCELERATION", "auto"),
            gpu_index=int(os.getenv("GPU_DEVICE_INDEX", "0"))
        )
        
        # NEW: Precision configuration
        precision = os.getenv("GPU_PRECISION", "float32")
        model_kwargs = {}
        if precision != "float32" and "cuda" in str(device):
            model_kwargs = {"torch_dtype": precision}
        
        # PRESERVE: Exact same model name and structure
        model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=str(device),
            model_kwargs=model_kwargs
        )
        
        logging.info(f"CrossEncoder loaded on device: {device}")
        return model
        
    except Exception as e:
        # PRESERVE: Exact same error handling as current code
        print(f"Failed to load reranking model: {e}")
        return None
```

### Integration Points

```yaml
ENVIRONMENT:
  - add to: .env.example
  - pattern: "USE_GPU_ACCELERATION=auto  # auto, true, false"
  - pattern: "GPU_PRECISION=float32      # float32, float16, bfloat16"
  - pattern: "GPU_DEVICE_INDEX=0         # GPU index for multi-GPU systems"

LOGGING:
  - enhance: existing print statements in crawl4ai_mcp.py
  - pattern: Replace print() with logging.info() for device selection
  - preserve: existing error handling print statements

IMPORTS:
  - add to: src/crawl4ai_mcp.py
  - pattern: "from .device_manager import get_optimal_device, cleanup_gpu_memory"
  - preserve: all existing imports
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
python -m py_compile src/device_manager.py
python -m py_compile src/crawl4ai_mcp.py

# Expected: No compilation errors. If errors, READ and fix syntax issues.
```

### Level 2: Unit Tests

```python
# CREATE tests/test_device_manager.py
def test_cpu_device_detection():
    """CPU device always works"""
    device = get_optimal_device(preference="cpu")
    assert str(device) == "cpu"

def test_gpu_detection_when_available():
    """GPU detection when CUDA available"""
    if torch.cuda.is_available():
        device = get_optimal_device(preference="cuda")
        assert "cuda" in str(device)

def test_fallback_to_cpu():
    """Fallback when GPU unavailable"""
    with mock.patch('torch.cuda.is_available', return_value=False):
        device = get_optimal_device(preference="auto")
        assert str(device) == "cpu"

def test_crossencoder_initialization():
    """CrossEncoder initializes with device parameter"""
    with mock.patch.dict(os.environ, {"USE_RERANKING": "true"}):
        model = initialize_reranking_model()
        assert model is not None
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_device_manager.py -v
uv run pytest tests/test_gpu_integration.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test

```bash
# Test GPU-enabled environment
export USE_RERANKING=true
export USE_GPU_ACCELERATION=auto
uv run -m src.crawl4ai_mcp

# Test search with reranking to verify GPU usage
# Expected: Device selection logged, reranking works correctly

# Test CPU fallback
export USE_GPU_ACCELERATION=false
uv run -m src.crawl4ai_mcp

# Expected: CPU device selected, functionality preserved
```

### Level 4: Performance Validation

```bash
# Benchmark GPU vs CPU performance
python -c "
import time
from src.device_manager import get_optimal_device
from sentence_transformers import CrossEncoder

# GPU benchmark
gpu_device = get_optimal_device('cuda')
gpu_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=str(gpu_device))

# CPU benchmark  
cpu_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

query_pairs = [('test query', f'test passage {i}') for i in range(100)]

# Time GPU inference
gpu_start = time.time()
gpu_scores = gpu_model.predict(query_pairs)
gpu_time = time.time() - gpu_start

# Time CPU inference
cpu_start = time.time()
cpu_scores = cpu_model.predict(query_pairs)
cpu_time = time.time() - cpu_start

print(f'GPU time: {gpu_time:.2f}s')
print(f'CPU time: {cpu_time:.2f}s')
print(f'Speedup: {cpu_time/gpu_time:.2f}x')
"

# Expected: 2-10x speedup on GPU systems
```

## Final Validation Checklist

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No compilation errors: `python -m py_compile src/crawl4ai_mcp.py`
- [ ] GPU detection works: Test on CUDA-enabled system
- [ ] CPU fallback works: Test with USE_GPU_ACCELERATION=false
- [ ] Memory cleanup works: No GPU memory leaks in long-running tests
- [ ] Backward compatibility: Existing functionality unchanged
- [ ] Configuration works: Environment variables control behavior
- [ ] Performance improved: Measurable speedup on GPU systems
- [ ] Error handling robust: Graceful handling of GPU failures
- [ ] Logging informative: Clear device selection messages

---

## Anti-Patterns to Avoid

- ❌ Don't use model.to(device) - CrossEncoder doesn't support it
- ❌ Don't skip GPU memory cleanup in long-running processes
- ❌ Don't assume torch.cuda.is_available() means GPU works
- ❌ Don't break existing CPU-only deployments
- ❌ Don't hardcode device selection - make it configurable
- ❌ Don't ignore GPU OOM errors - implement proper fallback
- ❌ Don't change existing error handling patterns
- ❌ Don't modify model precision without user consent

## PRP Quality Score: 9/10

**Confidence Level**: Very High - This PRP provides comprehensive context, production-ready patterns, extensive validation, and maintains backward compatibility while delivering significant performance improvements.