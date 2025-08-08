# AsyncIO + ThreadPoolExecutor Integration Patterns for ML Workloads

This document provides comprehensive implementation patterns for integrating ThreadPoolExecutor with asyncio applications, specifically for ML workloads like CrossEncoder reranking and batch embedding generation.

## Core Integration Pattern

The fundamental pattern uses `asyncio.get_running_loop().run_in_executor()` to bridge synchronous CPU-bound operations with the async event loop:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Callable

class AsyncMLExecutor:
    def __init__(self, max_workers: Optional[int] = None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop = None
    
    async def run_cpu_bound(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-bound function in thread pool"""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        
        return await self._loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def shutdown(self):
        """Graceful shutdown with resource cleanup"""
        if self.executor:
            self.executor.shutdown(wait=True)
```

## ML Model Threading Patterns

### Thread-Safe CrossEncoder Pattern

CrossEncoder models need special handling for thread safety and memory management:

```python
from sentence_transformers import CrossEncoder
import threading
from typing import List, Tuple
import torch

class ThreadSafeCrossEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._thread_local = threading.local()
    
    def _get_model(self):
        """Get thread-local model instance"""
        if not hasattr(self._thread_local, 'model'):
            self._thread_local.model = CrossEncoder(self.model_name)
            if hasattr(self._thread_local.model, 'model'):
                self._thread_local.model.model.eval()
        return self._thread_local.model
    
    def predict(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        """Thread-safe prediction with memory management"""
        model = self._get_model()
        
        with torch.no_grad():  # Critical for memory efficiency
            scores = model.predict(sentence_pairs, 
                                 batch_size=32, 
                                 show_progress_bar=False)
        
        return scores.tolist() if hasattr(scores, 'tolist') else scores
```

### Async Service Integration Pattern

```python
class MLService:
    def __init__(self, model_name: str, executor: ThreadPoolExecutor):
        self.cross_encoder = ThreadSafeCrossEncoder(model_name)
        self.executor = executor
    
    async def predict_async(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        """Async wrapper for CPU-bound ML inference"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.cross_encoder.predict, 
            sentence_pairs
        )
```

## FastAPI Lifespan Integration

### Application Lifecycle Management

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import os

# Global executor instance
cpu_executor: Optional[ThreadPoolExecutor] = None

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # Startup
    global cpu_executor
    cpu_executor = ThreadPoolExecutor(
        max_workers=min(os.cpu_count() or 1, 8),  # Cap for ML workloads
        thread_name_prefix="ml_worker"
    )
    
    try:
        yield
    finally:
        # Shutdown - Critical for resource cleanup
        if cpu_executor:
            cpu_executor.shutdown(wait=True, timeout=30)

app = FastAPI(lifespan=app_lifespan)
```

### Context Integration Pattern

For applications using context/dependency injection:

```python
@dataclass
class ApplicationContext:
    # Existing components
    database_client: Any
    cache: Any
    
    # ThreadPool executors
    cpu_executor: Optional[ThreadPoolExecutor] = None
    io_executor: Optional[ThreadPoolExecutor] = None

async def initialize_context() -> ApplicationContext:
    cpu_executor = ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="cpu_bound"
    )
    
    return ApplicationContext(
        database_client=db_client,
        cache=cache_instance,
        cpu_executor=cpu_executor
    )
```

## Worker Count Optimization

### CPU-Bound ML Workloads

```python
import os
import psutil

def calculate_ml_workers() -> int:
    """Calculate optimal workers for ML inference"""
    cpu_count = os.cpu_count() or 1
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # ML models are memory-intensive: ~2GB per worker for typical models
    memory_based_workers = max(1, int(memory_gb / 2))
    
    # Never exceed CPU cores, cap at 8 for memory reasons
    return min(memory_based_workers, cpu_count, 8)

# Usage
ml_executor = ThreadPoolExecutor(
    max_workers=calculate_ml_workers(),
    thread_name_prefix="ml_inference"
)
```

## Error Handling and Resilience

### Comprehensive Error Handling

```python
import logging
from typing import Optional
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

def async_executor_retry(max_retries: int = 3, timeout: Optional[float] = None):
    """Decorator for robust executor operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    if timeout:
                        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    else:
                        return await func(*args, **kwargs)
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout in {func.__name__}, attempt {attempt + 1}")
                    last_exception = "Timeout"
                    
                except Exception as e:
                    logger.error(f"Error in {func.__name__}, attempt {attempt + 1}: {e}")
                    last_exception = e
                    
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            raise RuntimeError(f"Failed after {max_retries} attempts: {last_exception}")
        return wrapper
    return decorator
```

### Graceful Degradation Pattern

```python
class ResilientMLService:
    def __init__(self, model_name: str, executor: Optional[ThreadPoolExecutor] = None):
        self.model_name = model_name
        self.executor = executor
        self.fallback_model = None  # Lightweight fallback
    
    async def predict_with_fallback(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """ML prediction with graceful degradation"""
        if self.executor:
            try:
                return await self._predict_async(pairs)
            except Exception as e:
                logger.warning(f"Executor prediction failed: {e}, using fallback")
        
        # Fallback to synchronous execution
        return self._predict_sync(pairs)
    
    async def _predict_async(self, pairs: List[Tuple[str, str]]) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._thread_safe_predict, 
            pairs
        )
    
    def _predict_sync(self, pairs: List[Tuple[str, str]]) -> List[float]:
        # Synchronous fallback implementation
        if not self.fallback_model:
            self.fallback_model = self._init_fallback_model()
        return self.fallback_model.predict(pairs)
```

## Memory Management for ML Models

### Memory-Efficient Processing

```python
class MemoryEfficientMLExecutor:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(
            max_workers=min(4, os.cpu_count()),
            thread_name_prefix="memory_optimized"
        )
    
    async def batch_predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Memory-efficient batch processing"""
        if len(pairs) <= self.batch_size:
            return await self._single_batch_predict(pairs)
        
        results = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_results = await self._single_batch_predict(batch)
            results.extend(batch_results)
            
            # Allow garbage collection between batches
            await asyncio.sleep(0)
        
        return results
    
    async def _single_batch_predict(self, batch: List[Tuple[str, str]]) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self._memory_safe_predict,
            batch
        )
    
    def _memory_safe_predict(self, batch: List[Tuple[str, str]]) -> List[float]:
        """Memory-safe prediction with cleanup"""
        model = self._get_thread_local_model()
        
        try:
            with torch.no_grad():
                scores = model.predict(batch, show_progress_bar=False)
            return scores.tolist() if hasattr(scores, 'tolist') else scores
        finally:
            # Explicit cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

## Performance Monitoring

### Metrics Collection

```python
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class ExecutorMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    errors: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0

class MonitoredMLExecutor:
    def __init__(self, model_name: str):
        self.model = ThreadSafeCrossEncoder(model_name)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics = ExecutorMetrics()
    
    async def predict_with_monitoring(self, pairs: List[Tuple[str, str]]) -> List[float]:
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self.model.predict, 
                pairs
            )
            
            self.metrics.successful_requests += 1
            return result
            
        except Exception as e:
            self.metrics.errors += 1
            raise
        finally:
            execution_time = time.time() - start_time
            self.metrics.total_time += execution_time
            
            if self.metrics.total_requests > 0:
                self.metrics.avg_response_time = (
                    self.metrics.total_time / self.metrics.total_requests
                )
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        total = self.metrics.total_requests
        if total == 0:
            return {"status": "no_requests"}
            
        return {
            "success_rate": self.metrics.successful_requests / total,
            "error_rate": self.metrics.errors / total,
            "avg_response_time": self.metrics.avg_response_time,
            "total_requests": total,
            "throughput_per_sec": 1.0 / self.metrics.avg_response_time if self.metrics.avg_response_time > 0 else 0
        }
```

## Testing Patterns

### Concurrent Testing

```python
import pytest
import concurrent.futures
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_concurrent_ml_operations():
    """Test thread safety of ML operations"""
    executor = ThreadPoolExecutor(max_workers=4)
    ml_service = MLService("test-model", executor)
    
    # Create concurrent requests
    test_pairs = [("query1", "doc1"), ("query2", "doc2")] * 10
    
    tasks = [
        ml_service.predict_async(test_pairs[i:i+2]) 
        for i in range(0, len(test_pairs), 2)
    ]
    
    try:
        results = await asyncio.gather(*tasks)
        
        # Verify all tasks completed successfully
        assert len(results) == len(tasks)
        assert all(isinstance(result, list) for result in results)
        
    finally:
        executor.shutdown(wait=True)

@pytest.mark.asyncio
async def test_executor_error_handling():
    """Test error handling in executor operations"""
    executor = ThreadPoolExecutor(max_workers=2)
    
    def failing_operation():
        raise ValueError("Simulated ML model error")
    
    with pytest.raises(ValueError, match="Simulated ML model error"):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, failing_operation)
    
    executor.shutdown(wait=True)
```

## Best Practices Summary

### Critical Implementation Guidelines

1. **Thread Safety**: Always use thread-local storage for ML models
2. **Memory Management**: Use `torch.no_grad()` and explicit cache clearing
3. **Worker Sizing**: Cap ML workers at `min(cpu_count, 8)` for memory reasons
4. **Error Handling**: Implement graceful degradation with fallback options
5. **Resource Cleanup**: Always use proper shutdown with timeout
6. **Batch Processing**: Process in smaller batches to manage memory
7. **Monitoring**: Track performance metrics for optimization
8. **Testing**: Include concurrent access and error handling tests

### Performance Expectations

- **Throughput Improvement**: 3-5x for multi-core systems with proper batching
- **Memory Usage**: ~2GB per worker for typical CrossEncoder models
- **Latency**: Single request latency may increase slightly (5-10ms overhead)
- **CPU Utilization**: Should see multi-core utilization instead of single-core blocking

This document provides the foundation for implementing ThreadPoolExecutor with asyncio in ML-intensive applications while maintaining performance, reliability, and resource efficiency.