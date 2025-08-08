#!/usr/bin/env python3
"""
Python test fixture for Tree-sitter parsing tests.

This file contains various Python language constructs to test comprehensive parsing.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import json


@dataclass
class Configuration:
    """Configuration data class with type hints."""

    name: str
    version: str = "1.0.0"
    debug: bool = False
    settings: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Union[str, bool, Dict]]:
        return {
            "name": self.name,
            "version": self.version,
            "debug": self.debug,
            "settings": self.settings,
        }


class BaseProcessor(ABC):
    """Abstract base class for processors."""

    def __init__(self, config: Configuration):
        self.config = config
        self._initialized = False

    @abstractmethod
    def process(self, data: List[str]) -> List[str]:
        """Process the given data."""
        pass

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class DataProcessor(BaseProcessor):
    """Concrete data processor implementation."""

    def __init__(self, config: Configuration):
        super().__init__(config)
        self.results: List[str] = []
        self.error_count = 0
        self._initialized = True

    def process(self, data: List[str]) -> List[str]:
        """Process data with error handling."""
        processed = []
        for item in data:
            try:
                result = self._process_item(item)
                processed.append(result)
                self.results.append(result)
            except ValueError as e:
                self.error_count += 1
                self._handle_error(e, item)
        return processed

    def _process_item(self, item: str) -> str:
        if not item or len(item.strip()) == 0:
            raise ValueError("Empty item")
        return f"processed_{item.strip().lower()}"

    def _handle_error(self, error: Exception, item: str):
        if self.config.debug:
            print(f"Error processing '{item}': {error}")

    async def process_async(self, data: List[str]) -> List[str]:
        """Asynchronous processing method."""
        results = []
        for item in data:
            result = await self._process_item_async(item)
            results.append(result)
        return results

    async def _process_item_async(self, item: str) -> str:
        # Simulate async work
        await asyncio.sleep(0.001)
        return self._process_item(item)

    def get_stats(self) -> Dict[str, int]:
        return {"total_processed": len(self.results), "error_count": self.error_count}

    @classmethod
    def from_config_file(cls, config_path: Path) -> "DataProcessor":
        """Create processor from configuration file."""
        with open(config_path, "r") as f:
            config_data = json.load(f)

        config = Configuration(
            name=config_data["name"],
            version=config_data.get("version", "1.0.0"),
            debug=config_data.get("debug", False),
        )
        return cls(config)

    @staticmethod
    def validate_data(data: List[str]) -> bool:
        """Validate input data."""
        return isinstance(data, list) and all(isinstance(item, str) for item in data)


def create_processor(name: str = "default", debug: bool = False) -> DataProcessor:
    """Factory function for creating processors."""
    config = Configuration(name=name, debug=debug)
    return DataProcessor(config)


def batch_process(items: List[str], processor: DataProcessor) -> Dict[str, List[str]]:
    """Batch process items with a processor."""
    if not DataProcessor.validate_data(items):
        raise ValueError("Invalid input data")

    results = processor.process(items)
    stats = processor.get_stats()

    return {"results": results, "stats": [f"{k}:{v}" for k, v in stats.items()]}


async def async_batch_process(
    items: List[str],
    processor: DataProcessor,
    callback: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Asynchronous batch processing with callback."""
    results = []

    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(processor._process_item_async(item)) for item in items
        ]

    for task in tasks:
        result = await task
        results.append(result)
        if callback:
            callback(result)

    return results


# Context manager example
class ProcessingContext:
    """Context manager for processing operations."""

    def __init__(self, processor: DataProcessor):
        self.processor = processor
        self.start_time = None

    def __enter__(self) -> DataProcessor:
        import time

        self.start_time = time.time()
        return self.processor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            import time

            duration = time.time() - self.start_time
            print(f"Processing completed in {duration:.2f} seconds")


# Generator function
def generate_test_data(count: int = 10) -> str:
    """Generator for test data."""
    for i in range(count):
        yield f"item_{i:03d}"


# Decorator example
def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""

    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return wrapper


@timing_decorator
def process_with_timing(data: List[str]) -> Dict[str, List[str]]:
    """Process data with timing information."""
    processor = create_processor("timed", debug=True)
    return batch_process(data, processor)


# Main execution
if __name__ == "__main__":
    # Example usage
    test_data = list(generate_test_data(5))

    # Synchronous processing
    processor = create_processor("main", debug=True)

    with ProcessingContext(processor) as ctx:
        results = batch_process(test_data, ctx)

    print(f"Processed {len(results['results'])} items")

    # Asynchronous processing
    async def main():
        async_results = await async_batch_process(
            test_data, processor, lambda r: print(f"Completed: {r}")
        )
        return async_results

    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
