# CrossEncoder GPU Optimization Patterns

## GPU Initialization Patterns

### Basic GPU Initialization
```python
from sentence_transformers import CrossEncoder

# Explicit device specification
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cuda")

# Auto-detection with fallback
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device)
```

### Performance Optimization
```python
# Float16 for faster inference
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2", 
    model_kwargs={"torch_dtype": "float16"}
)
```

### Production-Ready Initialization
```python
class ProductionCrossEncoder:
    def __init__(self, model_name, device=None, precision="float32"):
        self.device = self._determine_device(device)
        self.model = self._initialize_model(model_name, precision)
    
    def _determine_device(self, device):
        if device is not None:
            return device
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    
    def _initialize_model(self, model_name, precision):
        model_kwargs = {}
        if precision != "float32":
            model_kwargs["torch_dtype"] = precision
        
        try:
            model = CrossEncoder(model_name, device=self.device, model_kwargs=model_kwargs)
            return model
        except Exception as e:
            if "cuda" in self.device:
                logging.warning(f"GPU initialization failed: {e}. Falling back to CPU.")
                return CrossEncoder(model_name, device="cpu")
            raise e
```

## Performance Benchmarks
- **GPU vs CPU**: 5-10x faster for batch processing
- **Float16**: 40-60% speed improvement with minimal accuracy loss
- **Memory Usage**: Float16 reduces VRAM usage by ~50%

## Key URLs
- CrossEncoder Documentation: https://sbert.net/docs/cross_encoder/
- GPU Usage Examples: https://github.com/UKPLab/sentence-transformers/tree/master/examples/cross_encoder
- Performance Guide: https://sbert.net/docs/cross_encoder/usage/efficiency.html