# Python Async Processing Patterns for Unified Data Pipelines

## Overview

This document consolidates best practices for implementing high-performance async processing patterns in Python, specifically for unified data pipeline systems that handle repository processing, code analysis, and integration with multiple data stores (Neo4j, Vector DBs).

## Core Async Architecture Patterns

### 1. Producer-Consumer Pattern with AsyncIO

**Pattern**: Efficient file processing with controlled concurrency

```python
import asyncio
import aiofiles
from asyncio import Queue, Semaphore
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Callable, Any, List
from pathlib import Path
import logging

class AsyncFileProcessor:
    """High-performance async file processor with backpressure control."""
    
    def __init__(
        self,
        max_concurrent_files: int = 50,
        max_concurrent_operations: int = 10,
        queue_size: int = 100
    ):
        self.file_semaphore = Semaphore(max_concurrent_files)
        self.operation_semaphore = Semaphore(max_concurrent_operations)
        self.file_queue: Queue = Queue(maxsize=queue_size)
        self.result_queue: Queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound tasks
        
    async def process_repository_files(
        self,
        file_paths: List[Path],
        processor_func: Callable[[Path], Any]
    ) -> AsyncIterator[Any]:
        """Process files with controlled concurrency and backpressure."""
        
        # Start producer and consumer tasks
        producer_task = asyncio.create_task(
            self._produce_files(file_paths)
        )
        
        consumer_tasks = [
            asyncio.create_task(
                self._consume_and_process(processor_func)
            ) 
            for _ in range(5)  # 5 consumer workers
        ]
        
        # Process results as they become available
        processed_count = 0
        target_count = len(file_paths)
        
        try:
            while processed_count < target_count:
                result = await self.result_queue.get()
                if result is not None:
                    yield result
                processed_count += 1
                
        finally:
            # Cleanup
            producer_task.cancel()
            for task in consumer_tasks:
                task.cancel()
            
            await asyncio.gather(
                producer_task, 
                *consumer_tasks, 
                return_exceptions=True
            )
    
    async def _produce_files(self, file_paths: List[Path]):
        """Producer: Add files to processing queue."""
        for file_path in file_paths:
            await self.file_queue.put(file_path)
        
        # Signal end of files
        for _ in range(5):  # One sentinel per consumer
            await self.file_queue.put(None)
    
    async def _consume_and_process(self, processor_func: Callable):
        """Consumer: Process files from queue with resource limiting."""
        while True:
            file_path = await self.file_queue.get()
            if file_path is None:  # Sentinel value
                break
            
            async with self.file_semaphore:
                try:
                    result = await self._process_single_file(file_path, processor_func)
                    await self.result_queue.put(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    await self.result_queue.put(None)  # Signal error
                finally:
                    self.file_queue.task_done()
    
    async def _process_single_file(self, file_path: Path, processor_func: Callable) -> Any:
        """Process individual file with hybrid async/thread approach."""
        
        # I/O operations: Use async
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # CPU-bound operations: Use thread pool
        async with self.operation_semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                processor_func, 
                content, 
                str(file_path)
            )
        
        return {
            'file_path': str(file_path),
            'result': result,
            'size': len(content)
        }
```

### 2. Batch Processing with Async Coordination

**Pattern**: Efficient batch operations for database writes

```python
from typing import TypeVar, Generic, List, Dict, Any, Optional
from dataclasses import dataclass
import time

T = TypeVar('T')

@dataclass
class BatchResult:
    """Result of batch processing operation."""
    successful_items: int
    failed_items: int
    processing_time: float
    errors: List[str]

class AsyncBatchProcessor(Generic[T]):
    """Generic async batch processor with adaptive batching."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_concurrent_batches: int = 5,
        adaptive_batching: bool = True
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.adaptive_batching = adaptive_batching
        self.performance_metrics = []
        self.batch_semaphore = Semaphore(max_concurrent_batches)
    
    async def process_items(
        self,
        items: List[T],
        batch_processor: Callable[[List[T]], Any]
    ) -> BatchResult:
        """Process items in optimized batches."""
        
        start_time = time.time()
        successful_items = 0
        failed_items = 0
        errors = []
        
        # Adaptive batch sizing based on performance
        current_batch_size = self._calculate_optimal_batch_size()
        
        # Create batches
        batches = [
            items[i:i + current_batch_size]
            for i in range(0, len(items), current_batch_size)
        ]
        
        # Process batches concurrently
        batch_tasks = [
            asyncio.create_task(
                self._process_batch(batch, batch_processor)
            )
            for batch in batches
        ]
        
        # Collect results
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                failed_items += len(batches[i])
                errors.append(f"Batch {i} failed: {str(result)}")
            else:
                successful_items += result.get('processed_count', 0)
                if 'errors' in result:
                    errors.extend(result['errors'])
                    failed_items += len(result['errors'])
        
        processing_time = time.time() - start_time
        
        # Update performance metrics for adaptive batching
        if self.adaptive_batching:
            self._update_performance_metrics(
                current_batch_size, 
                processing_time, 
                successful_items
            )
        
        return BatchResult(
            successful_items=successful_items,
            failed_items=failed_items,
            processing_time=processing_time,
            errors=errors
        )
    
    async def _process_batch(
        self, 
        batch: List[T], 
        processor: Callable[[List[T]], Any]
    ) -> Dict[str, Any]:
        """Process individual batch with resource control."""
        
        async with self.batch_semaphore:
            try:
                # Execute processor (may be sync or async)
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(batch)
                else:
                    # Run sync processor in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, processor, batch)
                
                return {
                    'processed_count': len(batch),
                    'result': result
                }
                
            except Exception as e:
                return {
                    'processed_count': 0,
                    'errors': [f"Batch processing error: {str(e)}"]
                }
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on performance history."""
        if not self.adaptive_batching or len(self.performance_metrics) < 5:
            return self.batch_size
        
        # Simple adaptive algorithm: find batch size with best throughput
        recent_metrics = self.performance_metrics[-10:]
        best_throughput = 0
        best_batch_size = self.batch_size
        
        for metric in recent_metrics:
            throughput = metric['items_per_second']
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = metric['batch_size']
        
        # Gradual adjustment to avoid oscillation
        if best_batch_size != self.batch_size:
            adjustment = (best_batch_size - self.batch_size) * 0.3
            return max(10, int(self.batch_size + adjustment))
        
        return self.batch_size
    
    def _update_performance_metrics(
        self, 
        batch_size: int, 
        processing_time: float, 
        items_processed: int
    ):
        """Update performance metrics for adaptive batching."""
        items_per_second = items_processed / max(processing_time, 0.001)
        
        self.performance_metrics.append({
            'batch_size': batch_size,
            'processing_time': processing_time,
            'items_processed': items_processed,
            'items_per_second': items_per_second,
            'timestamp': time.time()
        })
        
        # Keep only recent metrics
        if len(self.performance_metrics) > 50:
            self.performance_metrics = self.performance_metrics[-50:]
```

### 3. Resource Management and Cleanup

**Pattern**: Robust resource management for long-running pipelines

```python
from contextlib import AsyncExitStack
import weakref
from typing import AsyncContextManager

class ResourceManager:
    """Manages resources in async pipelines with automatic cleanup."""
    
    def __init__(self):
        self._resources = weakref.WeakSet()
        self._cleanup_tasks = []
        self._shutdown_event = asyncio.Event()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup_all_resources()
    
    def register_resource(self, resource: Any, cleanup_func: Optional[Callable] = None):
        """Register resource for automatic cleanup."""
        self._resources.add(resource)
        if cleanup_func:
            self._cleanup_tasks.append(cleanup_func)
    
    async def cleanup_all_resources(self):
        """Clean up all registered resources."""
        # Signal shutdown to all components
        self._shutdown_event.set()
        
        # Execute custom cleanup functions
        cleanup_results = await asyncio.gather(
            *[cleanup() for cleanup in self._cleanup_tasks],
            return_exceptions=True
        )
        
        for i, result in enumerate(cleanup_results):
            if isinstance(result, Exception):
                logger.error(f"Cleanup task {i} failed: {result}")
        
        # Clear resources
        self._resources.clear()
        self._cleanup_tasks.clear()

class AsyncDatabaseConnection:
    """Async database connection with proper resource management."""
    
    def __init__(self, connection_string: str, resource_manager: ResourceManager):
        self.connection_string = connection_string
        self.resource_manager = resource_manager
        self.connection = None
        self._connection_lock = asyncio.Lock()
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def connect(self):
        """Establish database connection with retry logic."""
        async with self._connection_lock:
            if self.connection is not None:
                return
            
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Simulated async database connection
                    self.connection = await self._create_connection()
                    
                    # Register for cleanup
                    self.resource_manager.register_resource(
                        self.connection, 
                        self.close
                    )
                    
                    logger.info("Database connection established")
                    return
                    
                except Exception as e:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(retry_delay * (2 ** attempt))
    
    async def close(self):
        """Close database connection."""
        if self.connection:
            try:
                await self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self.connection = None
    
    async def execute_batch(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Execute batch operations with automatic reconnection."""
        if not self.connection:
            await self.connect()
        
        try:
            # Execute operations in batch
            results = []
            for operation in operations:
                result = await self._execute_single_operation(operation)
                results.append(result)
            return results
            
        except ConnectionError:
            # Reconnect and retry once
            logger.warning("Connection lost, attempting reconnection")
            await self.close()
            await self.connect()
            
            results = []
            for operation in operations:
                result = await self._execute_single_operation(operation)
                results.append(result)
            return results
    
    async def _create_connection(self):
        """Create actual database connection (implementation-specific)."""
        # This would be implemented for specific database types
        await asyncio.sleep(0.1)  # Simulate connection time
        return MockConnection()
    
    async def _execute_single_operation(self, operation: Dict[str, Any]) -> Any:
        """Execute single database operation."""
        # Implementation-specific operation execution
        await asyncio.sleep(0.01)  # Simulate operation time
        return {'status': 'success', 'operation': operation}

class MockConnection:
    """Mock connection for example purposes."""
    async def close(self):
        pass
```

### 4. Progress Tracking and Monitoring

**Pattern**: Real-time progress tracking for long-running operations

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
import asyncio

@dataclass
class ProgressMetrics:
    """Progress tracking metrics."""
    total_items: int
    processed_items: int = 0
    failed_items: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    current_operation: str = "Starting..."
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def elapsed_time(self) -> timedelta:
        return datetime.now() - self.start_time
    
    @property
    def processing_rate(self) -> float:
        elapsed_seconds = self.elapsed_time.total_seconds()
        if elapsed_seconds == 0:
            return 0.0
        return self.processed_items / elapsed_seconds
    
    def update_completion_estimate(self):
        """Update estimated completion time."""
        if self.processing_rate > 0:
            remaining_items = self.total_items - self.processed_items
            remaining_seconds = remaining_items / self.processing_rate
            self.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)

class ProgressTracker:
    """Async progress tracker for long-running operations."""
    
    def __init__(
        self, 
        total_items: int,
        update_callback: Optional[Callable[[ProgressMetrics], None]] = None,
        update_interval: float = 1.0
    ):
        self.metrics = ProgressMetrics(total_items=total_items)
        self.update_callback = update_callback
        self.update_interval = update_interval
        self._update_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """Start progress tracking."""
        if self._update_task is None:
            self._update_task = asyncio.create_task(self._progress_update_loop())
    
    async def stop(self):
        """Stop progress tracking."""
        self._stop_event.set()
        if self._update_task:
            await self._update_task
            self._update_task = None
    
    def update_progress(
        self, 
        processed_increment: int = 1, 
        failed_increment: int = 0,
        current_operation: Optional[str] = None
    ):
        """Update progress metrics."""
        self.metrics.processed_items += processed_increment
        self.metrics.failed_items += failed_increment
        
        if current_operation:
            self.metrics.current_operation = current_operation
        
        self.metrics.update_completion_estimate()
    
    async def _progress_update_loop(self):
        """Background task for periodic progress updates."""
        while not self._stop_event.is_set():
            try:
                if self.update_callback:
                    self.update_callback(self.metrics)
                
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.update_interval
                )
                
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue updating
            except Exception as e:
                logger.error(f"Progress update error: {e}")
```

### 5. Complete Async Pipeline Example

**Pattern**: Complete async pipeline for repository processing

```python
class UnifiedAsyncPipeline:
    """Complete async pipeline for repository processing."""
    
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.file_processor = AsyncFileProcessor()
        self.batch_processor = AsyncBatchProcessor[Dict[str, Any]]()
        
    async def process_repository(
        self,
        repo_path: Path,
        output_connections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complete repository processing pipeline."""
        
        async with self.resource_manager:
            # Discover files
            file_paths = await self._discover_files(repo_path)
            
            # Setup progress tracking
            async with ProgressTracker(
                total_items=len(file_paths),
                update_callback=self._log_progress
            ) as progress:
                
                # Process files with async streaming
                processed_files = []
                async for result in self.file_processor.process_repository_files(
                    file_paths,
                    self._analyze_file_content
                ):
                    if result:
                        processed_files.append(result)
                        progress.update_progress(
                            current_operation=f"Processed {result['file_path']}"
                        )
                
                # Batch process results to databases
                progress.update_progress(
                    current_operation="Storing results in databases"
                )
                
                storage_result = await self.batch_processor.process_items(
                    processed_files,
                    lambda batch: self._store_batch_results(batch, output_connections)
                )
                
                return {
                    'files_discovered': len(file_paths),
                    'files_processed': len(processed_files),
                    'storage_result': storage_result,
                    'total_processing_time': progress.metrics.elapsed_time.total_seconds()
                }
    
    async def _discover_files(self, repo_path: Path) -> List[Path]:
        """Async file discovery."""
        supported_extensions = {'.py', '.js', '.ts', '.java', '.md'}
        
        discovered_files = []
        
        def scan_directory():
            return [
                file_path
                for file_path in repo_path.rglob('*')
                if file_path.is_file() and file_path.suffix in supported_extensions
            ]
        
        # Run file discovery in thread pool for large repositories
        loop = asyncio.get_event_loop()
        discovered_files = await loop.run_in_executor(None, scan_directory)
        
        return discovered_files
    
    def _analyze_file_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze file content (CPU-bound operation)."""
        # This would include Tree-sitter parsing, AST analysis, etc.
        # Simulated analysis
        return {
            'file_path': file_path,
            'content_hash': hash(content),
            'line_count': len(content.splitlines()),
            'analysis_result': f"Analyzed {len(content)} characters"
        }
    
    async def _store_batch_results(
        self, 
        batch: List[Dict[str, Any]], 
        connections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store batch results in multiple databases."""
        
        # Parallel storage in multiple systems
        neo4j_task = asyncio.create_task(
            self._store_in_neo4j(batch, connections['neo4j'])
        )
        
        vector_task = asyncio.create_task(
            self._store_in_vector_db(batch, connections['vector_db'])
        )
        
        results = await asyncio.gather(neo4j_task, vector_task, return_exceptions=True)
        
        return {
            'neo4j_result': results[0] if not isinstance(results[0], Exception) else str(results[0]),
            'vector_db_result': results[1] if not isinstance(results[1], Exception) else str(results[1]),
            'batch_size': len(batch)
        }
    
    async def _store_in_neo4j(self, batch: List[Dict[str, Any]], connection) -> str:
        """Store batch in Neo4j."""
        await asyncio.sleep(0.1)  # Simulate database operation
        return f"Stored {len(batch)} items in Neo4j"
    
    async def _store_in_vector_db(self, batch: List[Dict[str, Any]], connection) -> str:
        """Store batch in vector database."""
        await asyncio.sleep(0.05)  # Simulate database operation  
        return f"Stored {len(batch)} items in vector DB"
    
    def _log_progress(self, metrics: ProgressMetrics):
        """Log progress updates."""
        logger.info(
            f"Progress: {metrics.completion_percentage:.1f}% "
            f"({metrics.processed_items}/{metrics.total_items}) - "
            f"{metrics.current_operation}"
        )

# Usage example
async def main():
    pipeline = UnifiedAsyncPipeline()
    
    connections = {
        'neo4j': "neo4j://localhost:7687",
        'vector_db': "vector_db_connection"
    }
    
    result = await pipeline.process_repository(
        Path("/path/to/repository"),
        connections
    )
    
    print(f"Processing complete: {result}")

# Run the pipeline
if __name__ == "__main__":
    asyncio.run(main())
```

This comprehensive guide provides production-ready async patterns for building high-performance unified data pipelines that can efficiently process large code repositories while maintaining resource control and monitoring capabilities.