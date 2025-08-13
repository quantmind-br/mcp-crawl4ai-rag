"""
Pipeline stage coordination for optimized repository indexing.

This module provides the OptimizedIndexingPipeline class that orchestrates
the multi-stage processing pipeline for maximum performance and resource utilization.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass

from .file_processor import process_file_batch, detect_file_language, is_code_file
from ..unified_indexing_service import FileProcessingResult, ProgressTracker
from ...utils.performance_config import PerformanceConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageResult:
    """Result from a pipeline stage."""

    stage_name: str
    files_processed: int
    processing_time_seconds: float
    success_count: int
    error_count: int
    errors: List[str]


@dataclass
class FileBatch:
    """Represents a batch of files for processing."""

    batch_id: int
    file_paths: List[Path]
    repo_path: Path
    repo_name: str


class OptimizedIndexingPipeline:
    """
    Multi-stage pipeline for optimized repository indexing.

    Coordinates the following stages:
    1. Parallel file reading (I/O-bound, ThreadPoolExecutor)
    2. CPU-bound parsing (ProcessPoolExecutor)
    3. Batch embedding generation
    4. Bulk database writes
    """

    def __init__(
        self,
        io_executor: ThreadPoolExecutor,
        cpu_executor: ProcessPoolExecutor,
        config: PerformanceConfig,
    ):
        """
        Initialize the optimized indexing pipeline.

        Args:
            io_executor: ThreadPoolExecutor for I/O-bound operations
            cpu_executor: ProcessPoolExecutor for CPU-bound operations
            config: Performance configuration
        """
        self.io_executor = io_executor
        self.cpu_executor = cpu_executor
        self.config = config

        # Pipeline statistics
        self.stats = {
            "total_files_processed": 0,
            "total_processing_time": 0.0,
            "stage_results": [],
            "batches_processed": 0,
        }

    async def process_files_optimized(
        self,
        files: List[Path],
        repo_path: Path,
        repo_name: str,
        should_process_rag: bool = True,
        should_process_kg: bool = True,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> AsyncIterator[List[FileProcessingResult]]:
        """
        Process files using the optimized multi-stage pipeline.

        Args:
            files: List of file paths to process
            repo_path: Repository root path
            repo_name: Repository name
            should_process_rag: Whether to process for RAG
            should_process_kg: Whether to process for knowledge graph
            progress_tracker: Optional progress tracker

        Yields:
            Lists of FileProcessingResult objects for each batch
        """
        logger.info(f"Starting optimized pipeline processing for {len(files)} files")
        pipeline_start_time = time.time()

        # Create file batches for processing
        batches = self._create_file_batches(files, repo_path, repo_name)
        logger.info(f"Created {len(batches)} batches for processing")

        # Process batches
        for batch in batches:
            batch_start_time = time.time()

            try:
                # Stage 1 & 2: File reading and parsing (combined for efficiency)
                logger.debug(
                    f"Processing batch {batch.batch_id} with {len(batch.file_paths)} files"
                )

                file_paths_str = [str(fp) for fp in batch.file_paths]
                processed_files = await process_file_batch(
                    file_paths_str,
                    self.cpu_executor,
                    should_process_kg,
                    batch.repo_name,
                )

                # Stage 3: Convert to FileProcessingResult objects
                batch_results = await self._convert_to_processing_results(
                    processed_files,
                    batch.repo_path,
                    should_process_rag,
                    should_process_kg,
                )

                # Update statistics
                batch_time = time.time() - batch_start_time
                self.stats["batches_processed"] += 1
                self.stats["total_files_processed"] += len(batch_results)
                self.stats["total_processing_time"] += batch_time

                # Update progress tracker
                if progress_tracker:
                    progress_tracker.update_progress(
                        processed_increment=len(batch_results),
                        operation=f"Completed batch {batch.batch_id}",
                    )

                logger.debug(f"Batch {batch.batch_id} completed in {batch_time:.2f}s")
                yield batch_results

            except Exception as e:
                logger.error(f"Error processing batch {batch.batch_id}: {e}")

                # Create error results for the batch
                error_results = []
                for file_path in batch.file_paths:
                    relative_path = str(file_path.relative_to(batch.repo_path))
                    error_result = FileProcessingResult(
                        file_id=f"{repo_name}:{relative_path}",
                        relative_path=relative_path,
                        file_path=str(file_path),
                        language=detect_file_language(str(file_path)),
                        errors=[f"Batch processing error: {str(e)}"],
                    )
                    error_results.append(error_result)

                if progress_tracker:
                    progress_tracker.update_progress(
                        failed_increment=len(error_results),
                        operation=f"Failed batch {batch.batch_id}",
                    )

                yield error_results

        # Log final statistics
        total_time = time.time() - pipeline_start_time
        logger.info(
            f"Pipeline completed: {self.stats['total_files_processed']} files "
            f"in {total_time:.2f}s ({self.stats['total_files_processed'] / total_time:.1f} files/sec)"
        )

    def _create_file_batches(
        self,
        files: List[Path],
        repo_path: Path,
        repo_name: str,
    ) -> List[FileBatch]:
        """
        Create batches of files for processing.

        Args:
            files: List of file paths
            repo_path: Repository root path
            repo_name: Repository name

        Returns:
            List of FileBatch objects
        """
        batch_size = self.config.batch_size_file_processing
        batches = []

        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            batch = FileBatch(
                batch_id=i // batch_size,
                file_paths=batch_files,
                repo_path=repo_path,
                repo_name=repo_name,
            )
            batches.append(batch)

        return batches

    async def _convert_to_processing_results(
        self,
        processed_files: List[Dict[str, Any]],
        repo_path: Path,
        should_process_rag: bool,
        should_process_kg: bool,
    ) -> List[FileProcessingResult]:
        """
        Convert processed file data to FileProcessingResult objects.

        Args:
            processed_files: List of processed file dictionaries
            repo_path: Repository root path
            should_process_rag: Whether RAG processing was requested
            should_process_kg: Whether KG processing was requested

        Returns:
            List of FileProcessingResult objects
        """
        results = []

        for file_data in processed_files:
            file_path = Path(file_data["file_path"])

            try:
                # Calculate relative path
                relative_path = str(file_path.relative_to(repo_path))
            except ValueError:
                relative_path = file_path.name

            # Generate file_id
            file_id = f"{repo_path.name}:{relative_path}"

            # Determine processing success
            processed_for_rag = (
                should_process_rag and len(file_data.get("content", "")) > 0
            )
            processed_for_kg = should_process_kg and file_data.get(
                "processed_for_kg", False
            )

            # Estimate entities and chunks
            kg_entities = 0
            if processed_for_kg and file_data.get("kg_analysis"):
                kg_analysis = file_data["kg_analysis"]
                kg_entities = len(kg_analysis.get("classes", [])) + len(
                    kg_analysis.get("functions", [])
                )

            rag_chunks = 0
            if processed_for_rag:
                content = file_data.get("content", "")
                # Rough estimate: 5000 chars per chunk
                rag_chunks = max(1, len(content) // 5000) if content else 0

            result = FileProcessingResult(
                file_id=file_id,
                relative_path=relative_path,
                file_path=str(file_path),
                language=file_data.get("language", "unknown"),
                file_type=file_path.suffix,
                processed_for_rag=processed_for_rag,
                processed_for_kg=processed_for_kg,
                rag_chunks=rag_chunks,
                kg_entities=kg_entities,
                processing_time_seconds=0.0,  # Will be calculated at batch level
                errors=file_data.get("processing_errors", []),
            )

            results.append(result)

        return results

    async def stage_read_files(self, file_paths: List[str]) -> List[Tuple[str, str]]:
        """
        Stage 1: Parallel file reading using async I/O.

        Args:
            file_paths: List of file paths to read

        Returns:
            List of (file_path, content) tuples
        """
        stage_start = time.time()
        logger.debug(f"Stage 1: Reading {len(file_paths)} files")

        # Import aiofiles for async file operations
        import aiofiles

        async def read_single_file(file_path: str) -> tuple[str, str]:
            try:
                async with aiofiles.open(
                    file_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    content = await f.read()
                return file_path, content
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return file_path, ""

        # Read all files concurrently
        tasks = [read_single_file(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        file_contents = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"File reading exception: {result}")
            else:
                file_contents.append(result)

        stage_time = time.time() - stage_start
        logger.debug(
            f"Stage 1 completed: {len(file_contents)} files in {stage_time:.2f}s"
        )

        return file_contents

    async def stage_parse_files(
        self,
        file_contents: List[tuple[str, str]],
        repo_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: CPU-bound parsing using ProcessPoolExecutor.

        Args:
            file_contents: List of (file_path, content) tuples
            repo_name: Repository name

        Returns:
            List of parsing results
        """
        stage_start = time.time()
        logger.debug(f"Stage 2: Parsing {len(file_contents)} files")

        # Filter for code files only
        code_files = []
        for file_path, content in file_contents:
            if is_code_file(file_path) and content.strip():
                code_files.append((file_path, content))

        if not code_files:
            logger.debug("No code files to parse")
            return []

        # Submit parsing tasks to ProcessPoolExecutor
        from .file_processor import parse_file_for_kg

        parse_futures = []
        for file_path, content in code_files:
            language = detect_file_language(file_path)
            future = self.cpu_executor.submit(
                parse_file_for_kg,
                file_path,
                content,
                language,
                repo_name,
            )
            parse_futures.append((future, file_path, content))

        # Collect results
        parse_results = []
        for future, file_path, content in parse_futures:
            try:
                kg_result = future.result(timeout=60)
                result = {
                    "file_path": file_path,
                    "content": content,
                    "kg_analysis": kg_result,
                    "processed_for_kg": kg_result is not None,
                }
                parse_results.append(result)
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                result = {
                    "file_path": file_path,
                    "content": content,
                    "kg_analysis": None,
                    "processed_for_kg": False,
                    "parsing_error": str(e),
                }
                parse_results.append(result)

        stage_time = time.time() - stage_start
        logger.debug(
            f"Stage 2 completed: {len(parse_results)} files in {stage_time:.2f}s"
        )

        return parse_results

    async def stage_generate_embeddings(
        self,
        file_contents: List[str],
    ) -> List[List[float]]:
        """
        Stage 3: Batch embedding generation.

        Args:
            file_contents: List of text content to embed

        Returns:
            List of embedding vectors
        """
        stage_start = time.time()
        logger.debug(f"Stage 3: Generating embeddings for {len(file_contents)} texts")

        # Import embedding service
        try:
            from ...services.embedding_service import create_embeddings_batch
        except ImportError:
            from services.embedding_service import create_embeddings_batch

        # Generate embeddings in optimal batch sizes
        batch_size = self.config.batch_size_embeddings
        all_embeddings = []

        for i in range(0, len(file_contents), batch_size):
            batch_texts = file_contents[i : i + batch_size]
            batch_embeddings = create_embeddings_batch(batch_texts)

            # Handle hybrid search mode
            if isinstance(batch_embeddings, tuple):
                # Hybrid mode: (dense_vectors, sparse_vectors)
                dense_vectors, _ = batch_embeddings
                all_embeddings.extend(dense_vectors)
            else:
                # Regular mode: list of embeddings
                all_embeddings.extend(batch_embeddings)

        stage_time = time.time() - stage_start
        logger.debug(
            f"Stage 3 completed: {len(all_embeddings)} embeddings in {stage_time:.2f}s"
        )

        return all_embeddings

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        return {
            "total_files_processed": self.stats["total_files_processed"],
            "total_processing_time": self.stats["total_processing_time"],
            "batches_processed": self.stats["batches_processed"],
            "average_files_per_second": (
                self.stats["total_files_processed"]
                / self.stats["total_processing_time"]
                if self.stats["total_processing_time"] > 0
                else 0
            ),
            "stage_results": self.stats["stage_results"],
        }
