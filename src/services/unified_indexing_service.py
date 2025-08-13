"""
Unified indexing service for coordinated repository processing.

This service orchestrates the unified repository processing pipeline that
handles both Qdrant (RAG) and Neo4j (Knowledge Graph) indexing with
consistent file_id linking and resource management.
"""

import asyncio
import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, AsyncIterator
from concurrent.futures import ThreadPoolExecutor

# Import utilities and models
try:
    from ..utils.file_id_generator import generate_file_id, extract_repo_name
    from ..models.unified_indexing_models import (
        UnifiedIndexingRequest,
        UnifiedIndexingResponse,
        FileProcessingResult,
        IndexingDestination,
    )
    from ..services.rag_service import add_documents_to_vector_db, update_source_info
    from ..services.file_classifier import FileClassifier
    from ..features.github.repository.git_operations import GitRepository
    from ..clients.qdrant_client import get_qdrant_client
    from ..k_graph.services.repository_parser import DirectNeo4jExtractor

    # Embedding service imported when needed
    from ..event_loop_fix import setup_event_loop
except ImportError:
    # Fallback with correct relative imports
    try:
        from utils.file_id_generator import generate_file_id, extract_repo_name
        from models.unified_indexing_models import (
            UnifiedIndexingRequest,
            UnifiedIndexingResponse,
            FileProcessingResult,
            IndexingDestination,
        )
        from services.rag_service import add_documents_to_vector_db, update_source_info
        from services.file_classifier import FileClassifier
        from features.github.repository.git_operations import GitRepository
        from clients.qdrant_client import get_qdrant_client
        from src.k_graph.services.repository_parser import DirectNeo4jExtractor
        from event_loop_fix import setup_event_loop
    except ImportError:
        # Final fallback - add project root to path
        import os
        import sys

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.utils.file_id_generator import generate_file_id, extract_repo_name
        from src.models.unified_indexing_models import (
            UnifiedIndexingRequest,
            UnifiedIndexingResponse,
            FileProcessingResult,
            IndexingDestination,
        )
        from src.services.rag_service import (
            add_documents_to_vector_db,
            update_source_info,
        )
        from src.features.github.repository.git_operations import GitRepository
        from src.clients.qdrant_client import get_qdrant_client
        from src.k_graph.services.repository_parser import DirectNeo4jExtractor
        from src.event_loop_fix import setup_event_loop

logger = logging.getLogger(__name__)


# Inline model definitions for unified indexing


# All models (IndexingDestination, UnifiedIndexingRequest, UnifiedIndexingResponse, FileProcessingResult) are now imported from models


class ResourceManager:
    """Manages resources in async pipelines with automatic cleanup."""

    def __init__(self):
        self._temp_directories = []
        self._cleanup_tasks = []
        self._shutdown_event = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup_all_resources()

    def register_temp_directory(self, temp_dir: Path):
        """Register temporary directory for cleanup."""
        self._temp_directories.append(temp_dir)

    def register_cleanup_task(self, cleanup_func):
        """Register custom cleanup function."""
        self._cleanup_tasks.append(cleanup_func)

    async def cleanup_all_resources(self):
        """Clean up all registered resources."""
        logger.info("Starting resource cleanup...")

        # Execute custom cleanup functions
        for cleanup_func in self._cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func()
                else:
                    cleanup_func()
            except Exception as e:
                logger.error(f"Error in cleanup function: {e}")

        # Clean up temporary directories
        for temp_dir in self._temp_directories:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up {temp_dir}: {e}")

        # Clear lists
        self._temp_directories.clear()
        self._cleanup_tasks.clear()
        logger.info("Resource cleanup completed")


class ProgressTracker:
    """Progress tracking for repository processing operations."""

    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.failed_files = 0
        self.start_time = datetime.now()
        self.current_operation = "Starting..."

    def update_progress(
        self,
        processed_increment: int = 1,
        failed_increment: int = 0,
        operation: Optional[str] = None,
    ):
        """Update progress metrics."""
        self.processed_files += processed_increment
        self.failed_files += failed_increment

        if operation:
            self.current_operation = operation

        # Log progress periodically
        if self.processed_files % 10 == 0 or self.processed_files == self.total_files:
            completion_percentage = (
                (self.processed_files / self.total_files) * 100
                if self.total_files > 0
                else 0
            )
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            logger.info(
                f"Progress: {completion_percentage:.1f}% "
                f"({self.processed_files}/{self.total_files}) - "
                f"{self.current_operation} - "
                f"Elapsed: {elapsed_time:.1f}s"
            )


class UnifiedIndexingService:
    """
    Unified indexing service that coordinates repository processing for both
    Qdrant (RAG) and Neo4j (Knowledge Graph) with consistent file_id linking.
    """

    def __init__(self, qdrant_client=None, neo4j_parser=None, context=None):
        """
        Initialize the unified indexing service.

        Args:
            qdrant_client: Optional Qdrant client, will create if not provided
            neo4j_parser: Optional Neo4j repository parser, will create if not provided
            context: Optional Crawl4AIContext with executors and performance config
        """
        import os

        self.qdrant_client = qdrant_client or get_qdrant_client()
        # Only store Neo4j parser if provided, don't create one unless needed
        self.neo4j_parser = neo4j_parser
        self._neo4j_config = (
            None
            if neo4j_parser
            else {
                "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "user": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password"),
            }
        )
        self.github_repository = GitRepository()

        # NEW: Initialize file classifier for intelligent routing
        self.file_classifier = FileClassifier()
        logger.debug("FileClassifier initialized for intelligent routing")

        # Performance optimization components
        self.context = context
        if context and context.io_executor and context.cpu_executor:
            # Use context executors for optimized performance
            self.io_executor = context.io_executor
            self.cpu_executor = context.cpu_executor
            self.performance_config = context.performance_config
            logger.info("Using optimized executors from context")
        else:
            # Fallback to legacy executor for compatibility
            self.executor = ThreadPoolExecutor(max_workers=4)
            self.io_executor = None
            self.cpu_executor = None
            self.performance_config = None
            logger.info("Using legacy ThreadPoolExecutor for compatibility")

        # Initialize optimized pipeline if executors are available
        self.optimized_pipeline = None
        if self.io_executor and self.cpu_executor and self.performance_config:
            try:
                from .batch_processing.pipeline_stages import OptimizedIndexingPipeline

                self.optimized_pipeline = OptimizedIndexingPipeline(
                    self.io_executor,
                    self.cpu_executor,
                    self.performance_config,
                )
                logger.info("Optimized indexing pipeline initialized successfully")
            except ImportError as e:
                logger.warning(f"Could not initialize optimized pipeline: {e}")
                self.optimized_pipeline = None

    async def process_repository_unified(
        self, request: UnifiedIndexingRequest
    ) -> UnifiedIndexingResponse:
        """
        Process a GitHub repository for unified indexing across RAG and/or Neo4j systems.

        Args:
            request: Unified indexing request with repository URL and processing options

        Returns:
            Unified indexing response with comprehensive results and statistics
        """
        logger.info(f"Starting unified repository processing for {request.repo_url}")

        # Initialize response
        repo_name = extract_repo_name(request.repo_url)
        response = UnifiedIndexingResponse(
            success=False,
            repo_url=request.repo_url,
            repo_name=repo_name,
            destination=str(request.destination),
            files_processed=0,
            start_time=datetime.now(),
        )

        async with ResourceManager() as resource_manager:
            try:
                # Step 0: Initialize Neo4j parser if needed for this request
                if request.should_process_kg:
                    if not self.neo4j_parser and self._neo4j_config:
                        # Create Neo4j parser only when needed
                        self.neo4j_parser = DirectNeo4jExtractor(
                            self._neo4j_config["uri"],
                            self._neo4j_config["user"],
                            self._neo4j_config["password"],
                        )
                    if self.neo4j_parser:
                        await self.neo4j_parser.initialize()
                        logger.debug("Neo4j parser initialized successfully")

                # Step 1: Clone repository to temporary directory
                temp_dir = await self._clone_repository(
                    request.repo_url, request.max_size_mb, resource_manager
                )

                # Step 2: Discover and filter files
                discovered_files = await self._discover_repository_files(
                    temp_dir, request.file_types, request.max_files
                )

                logger.info(f"Discovered {len(discovered_files)} files for processing")

                if not discovered_files:
                    response.errors.append(
                        "No files found matching the specified criteria"
                    )
                    response.finalize()
                    return response

                # Step 3: Initialize progress tracking
                progress = ProgressTracker(len(discovered_files))

                # Step 4: Process files with coordinated indexing
                async for file_result in self._process_files_unified(
                    discovered_files, request, progress, temp_dir
                ):
                    if file_result:
                        response.add_file_result(file_result)

                # Step 5: Process collected Neo4j analyses if any
                neo4j_result = None
                if hasattr(self, "_neo4j_analyses") and self._neo4j_analyses:
                    neo4j_result = await self._batch_process_neo4j_analyses()

                    # Update response with Neo4j processing results
                    if neo4j_result:
                        entities_created = neo4j_result.get("entities_created", 0)
                        # Update the response totals with actual Neo4j results
                        response.neo4j_nodes = entities_created

                        # Update individual file results with correct kg_entities
                        files_processed_neo4j = neo4j_result.get("files_processed", 0)
                        if files_processed_neo4j > 0:
                            # Distribute entities across processed files
                            avg_entities_per_file = (
                                entities_created // files_processed_neo4j
                            )
                            remainder = entities_created % files_processed_neo4j

                            # Update kg_entities for files that were processed for Neo4j
                            neo4j_file_count = 0
                            for file_result in response.file_results:
                                if (
                                    file_result.processed_for_kg
                                    and neo4j_file_count < files_processed_neo4j
                                ):
                                    file_result.kg_entities = avg_entities_per_file
                                    if remainder > 0:
                                        file_result.kg_entities += 1
                                        remainder -= 1
                                    neo4j_file_count += 1

                # Step 6: Finalize response
                response.finalize()
                logger.info(
                    f"Completed unified processing: {response.success_rate:.1f}% success rate"
                )

                return response

            except Exception as e:
                logger.error(f"Error in unified repository processing: {e}")
                response.errors.append(f"Processing error: {str(e)}")
                response.finalize()
                return response

    async def _clone_repository(
        self, repo_url: str, max_size_mb: int, resource_manager: ResourceManager
    ) -> Path:
        """
        Clone repository to temporary directory with size validation.

        Args:
            repo_url: GitHub repository URL
            max_size_mb: Maximum repository size in MB
            resource_manager: Resource manager for cleanup registration

        Returns:
            Path to cloned repository

        Raises:
            ValueError: If repository is too large or clone fails
        """
        logger.info(f"Cloning repository {repo_url}")

        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="unified_repo_"))
        resource_manager.register_temp_directory(temp_dir)

        try:
            # Clone repository using GitRepository
            cloned_temp_dir = await asyncio.to_thread(
                self.github_repository.clone_repository, repo_url, max_size_mb
            )

            cloned_path = Path(cloned_temp_dir)
            logger.info(f"Successfully cloned repository to {cloned_path}")

            return cloned_path

        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise

    async def _discover_repository_files(
        self, repo_path: Path, file_types: List[str], max_files: int
    ) -> List[Path]:
        """
        Discover files in repository matching specified criteria.

        Args:
            repo_path: Path to cloned repository
            file_types: List of file extensions to include
            max_files: Maximum number of files to process

        Returns:
            List of file paths to process
        """
        logger.info(f"Discovering files in {repo_path} with extensions {file_types}")

        def scan_files():
            """Scan files synchronously in thread pool."""
            discovered = []

            for file_path in repo_path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in [ext.lower() for ext in file_types]
                    and len(discovered) < max_files
                ):
                    discovered.append(file_path)

            return discovered[:max_files]

        # Run file discovery in thread pool for large repositories
        discovered_files = await asyncio.to_thread(scan_files)

        logger.info(f"Discovered {len(discovered_files)} files for processing")
        return discovered_files

    async def _process_files_unified(
        self,
        files: List[Path],
        request: UnifiedIndexingRequest,
        progress: ProgressTracker,
        repo_path: Path,
    ) -> AsyncIterator[Optional[FileProcessingResult]]:
        """
        Process files with unified indexing across RAG and/or Neo4j systems.

        Uses optimized multi-stage pipeline when available, falls back to legacy
        processing for compatibility.

        Args:
            files: List of file paths to process
            request: Processing request configuration
            progress: Progress tracker
            repo_path: Repository root path for relative path calculation

        Yields:
            FileProcessingResult objects for each processed file
        """
        logger.info(f"Starting unified processing of {len(files)} files")

        # Use optimized pipeline if available
        if self.optimized_pipeline:
            logger.info("Using optimized multi-stage pipeline for processing")

            repo_name = extract_repo_name(request.repo_url)

            # Process files using the optimized pipeline
            async for batch_results in self.optimized_pipeline.process_files_optimized(
                files=files,
                repo_path=repo_path,
                repo_name=repo_name,
                should_process_rag=request.should_process_rag,
                should_process_kg=request.should_process_kg,
                progress_tracker=progress,
            ):
                # Yield each result from the batch
                for result in batch_results:
                    yield result
        else:
            # Fallback to legacy processing method
            logger.info("Using legacy file processing method")

            # Use performance config batch size if available, otherwise default
            if self.performance_config:
                batch_size = self.performance_config.batch_size_file_processing
            else:
                batch_size = 5  # Legacy default

            for i in range(0, len(files), batch_size):
                batch_files = files[i : i + batch_size]

                # Create tasks for batch processing
                tasks = []
                for file_path in batch_files:
                    task = asyncio.create_task(
                        self._process_single_file(file_path, request, repo_path)
                    )
                    tasks.append(task)

                # Process batch concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Yield results and update progress
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error processing file {batch_files[j]}: {result}"
                        )
                        progress.update_progress(failed_increment=1)
                        yield None
                    else:
                        progress.update_progress(
                            operation=f"Processed {result.relative_path}"
                            if result
                            else "Processing..."
                        )
                        yield result

    async def _process_single_file(
        self, file_path: Path, request: UnifiedIndexingRequest, repo_path: Path
    ) -> Optional[FileProcessingResult]:
        """
        Process a single file for unified indexing.

        Args:
            file_path: Path to file to process
            request: Processing request configuration
            repo_path: Repository root path

        Returns:
            FileProcessingResult with processing statistics
        """
        start_time = time.time()

        # Calculate relative path
        try:
            relative_path = str(file_path.relative_to(repo_path))
        except ValueError:
            relative_path = file_path.name

        # Generate consistent file_id
        file_id = generate_file_id(request.repo_url, relative_path)

        # Initialize result
        result = FileProcessingResult(
            file_id=file_id,
            file_path=str(file_path),
            relative_path=relative_path,
            language=self._detect_file_language(file_path),
            file_type=file_path.suffix,
        )

        try:
            # Read file content
            content = await asyncio.to_thread(
                lambda: open(file_path, "r", encoding="utf-8", errors="ignore").read()
            )

            # NEW: Intelligent classification integration
            if request.routing_config.enable_intelligent_routing:
                # Use intelligent classification to determine processing
                classification = self.file_classifier.classify_file(
                    str(file_path), request.routing_config
                )
                should_process_rag = classification.should_process_rag
                should_process_kg = classification.should_process_kg

                # Store classification metadata
                result.classification_result = classification
                result.routing_decision = (
                    f"{classification.destination.value}: {classification.reasoning}"
                )
                result.classification_time_ms = classification.classification_time_ms

                logger.debug(
                    f"Intelligent routing for {relative_path}: {classification.destination.value} (confidence: {classification.confidence:.2f})"
                )
            else:
                # PATTERN: Backward compatibility fallback - use static request settings
                should_process_rag = request.should_process_rag
                should_process_kg = request.should_process_kg
                result.routing_decision = f"Static routing: {'RAG' if should_process_rag else ''}{'+' if should_process_rag and should_process_kg else ''}{'KG' if should_process_kg else ''}"

                logger.debug(
                    f"Static routing for {relative_path}: destination={request.destination.value}"
                )

            # Process for RAG if determined by routing logic
            if should_process_rag:
                rag_success = await self._process_file_for_rag(
                    file_path, content, file_id, request
                )
                if rag_success:
                    result.processed_for_rag = True
                    result.rag_chunks = len(
                        self._chunk_content(content, request.chunk_size)
                    )

            # Process for Neo4j if determined by routing logic
            if should_process_kg:
                kg_success = await self._process_file_for_neo4j(
                    file_path, content, file_id, request
                )
                if kg_success:
                    result.processed_for_kg = True
                    result.kg_entities = await self._estimate_kg_entities(
                        content, result.language
                    )

        except Exception as e:
            error_msg = f"Error processing {relative_path}: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        # Calculate processing time
        result.processing_time_seconds = time.time() - start_time

        return result

    async def _process_file_for_rag(
        self,
        file_path: Path,
        content: str,
        file_id: str,
        request: UnifiedIndexingRequest,
    ) -> bool:
        """
        Process file for RAG (Qdrant) indexing.

        Args:
            file_path: Path to file
            content: File content
            file_id: Generated file ID
            request: Processing request

        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Chunk content
            chunks = self._chunk_content(content, request.chunk_size)

            if not chunks:
                return True  # Empty file is not an error

            # Prepare data for vector database
            urls = [request.repo_url] * len(chunks)
            chunk_numbers = list(range(len(chunks)))
            metadatas = []
            file_ids = []

            for i, chunk in enumerate(chunks):
                metadata = {
                    "chunk_index": i,
                    "file_path": str(file_path),
                    "relative_path": str(
                        file_path.relative_to(file_path.parent.parent)
                    ),
                    "language": self._detect_file_language(file_path),
                    "source": extract_repo_name(request.repo_url),
                    "processing_time": datetime.now().isoformat(),
                }
                metadatas.append(metadata)
                file_ids.append(file_id)

            # Update source information
            source_id = extract_repo_name(request.repo_url)
            source_summary = f"Repository: {source_id}"
            total_words = len(content.split())

            update_source_info(
                self.qdrant_client, source_id, source_summary, total_words
            )

            # Add documents to vector database with file_id support
            add_documents_to_vector_db(
                self.qdrant_client,
                urls,
                chunk_numbers,
                chunks,
                metadatas,
                {request.repo_url: content},  # url_to_full_document mapping
                batch_size=50,
                file_ids=file_ids,
            )

            logger.debug(
                f"Successfully processed {file_path.name} for RAG with {len(chunks)} chunks"
            )
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path} for RAG: {e}")
            return False

    async def _process_file_for_neo4j(
        self,
        file_path: Path,
        content: str,
        file_id: str,
        request: UnifiedIndexingRequest,
    ) -> bool:
        """
        Process file for Neo4j (Knowledge Graph) indexing.

        Args:
            file_path: Path to file
            content: File content
            file_id: Generated file ID
            request: Processing request

        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Skip non-code files for Neo4j processing
            if not self._is_code_file(file_path):
                return True

            # Process with Neo4j parser in thread pool
            parse_result = await asyncio.to_thread(
                self._parse_file_for_neo4j,
                file_path,
                content,
                file_id,
                request.repo_url,
            )

            if parse_result:
                logger.debug(f"Successfully processed {file_path.name} for Neo4j")
                return True
            else:
                logger.warning(f"Failed to process {file_path.name} for Neo4j")
                return False

        except Exception as e:
            logger.error(f"Error processing {file_path} for Neo4j: {e}")
            return False

    def _parse_file_for_neo4j(
        self, file_path: Path, content: str, file_id: str, repo_url: str
    ) -> bool:
        """
        Parse file for Neo4j in thread pool (synchronous).

        Args:
            file_path: Path to file
            content: File content
            file_id: Generated file ID
            repo_url: Repository URL

        Returns:
            True if parsing successful, False otherwise
        """
        try:
            # Extract repository name and create temporary directory structure
            repo_name = extract_repo_name(repo_url)

            # Use the Neo4j analyzer to parse the file
            if not hasattr(self.neo4j_parser, "analyzer"):
                logger.debug(f"Neo4j parser not properly initialized for {file_path}")
                return True  # Don't fail, just skip

            # Create a temporary file path that mimics the repository structure
            temp_repo_root = file_path.parent
            while temp_repo_root.parent != temp_repo_root and not any(
                (temp_repo_root / marker).exists()
                for marker in [".git", "go.mod", "package.json", "pyproject.toml"]
            ):
                temp_repo_root = temp_repo_root.parent

            # Try to identify project modules for import filtering
            project_modules = {repo_name, file_path.stem}
            if hasattr(self, "_project_modules"):
                project_modules.update(self._project_modules)

            # Analyze the file using the Neo4j analyzer
            analysis = self.neo4j_parser.analyzer.analyze_file(
                file_path, temp_repo_root, project_modules
            )

            if analysis:
                logger.debug(
                    f"Successfully analyzed {file_path.name} for Neo4j: "
                    f"{len(analysis.get('classes', []))} classes, "
                    f"{len(analysis.get('functions', []))} functions"
                )

                # Store the analysis data for later batch processing
                if not hasattr(self, "_neo4j_analyses"):
                    self._neo4j_analyses = []
                    self._neo4j_repo_name = repo_name

                analysis["file_id"] = file_id
                self._neo4j_analyses.append(analysis)
                return True
            else:
                logger.debug(f"No analysis data extracted for {file_path}")
                return True  # Don't treat as failure

        except Exception as e:
            logger.error(f"Error in Neo4j parsing for {file_path}: {e}")
            return False

    async def _batch_process_neo4j_analyses(self):
        """
        Process collected Neo4j analyses in batch for improved performance.

        This method takes all accumulated analyses from individual file processing
        and processes them through the Neo4j extractor's _create_graph method.
        """
        if not hasattr(self, "_neo4j_analyses") or not self._neo4j_analyses:
            logger.debug("No Neo4j analyses to process")
            return

        repo_name = getattr(self, "_neo4j_repo_name", "unknown")
        analyses_count = len(self._neo4j_analyses)

        logger.info(
            f"Starting batch processing of {analyses_count} Neo4j analyses for {repo_name}"
        )

        try:
            # Process analyses in thread pool to avoid blocking the event loop
            async def batch_process():
                logger.debug(
                    f"Calling _create_graph with repo_name={repo_name}, modules_count={analyses_count}"
                )

                # Use the existing _create_graph method from DirectNeo4jExtractor
                await self.neo4j_parser._create_graph(
                    repo_name=repo_name, modules_data=self._neo4j_analyses
                )

                # Verify the data was actually created
                async with self.neo4j_parser.driver.session() as session:
                    # Check if repository was created
                    repo_result = await session.run(
                        "MATCH (r:Repository {name: $repo_name}) RETURN count(r) as count",
                        repo_name=repo_name,
                    )
                    repo_record = await repo_result.single()
                    repo_count = repo_record["count"] if repo_record else 0

                    # Check total nodes created
                    node_result = await session.run(
                        "MATCH (n) RETURN count(n) as count"
                    )
                    node_record = await node_result.single()
                    total_nodes = node_record["count"] if node_record else 0

                    logger.info(
                        f"Verification: Repository '{repo_name}' exists: {repo_count > 0}, Total nodes in DB: {total_nodes}"
                    )

                # Return basic statistics
                total_classes = sum(
                    len(analysis.get("classes", []))
                    for analysis in self._neo4j_analyses
                )
                total_functions = sum(
                    len(analysis.get("functions", []))
                    for analysis in self._neo4j_analyses
                )
                total_methods = sum(
                    len(cls.get("methods", []))
                    for analysis in self._neo4j_analyses
                    for cls in analysis.get("classes", [])
                )

                return {
                    "entities_created": total_classes + total_functions + total_methods,
                    "relationships_created": analyses_count,  # At least one file->entity per analysis
                    "files_processed": analyses_count,
                }

            # Run batch processing
            processing_result = await batch_process()

            if processing_result:
                entities_created = processing_result.get("entities_created", 0)
                files_processed = processing_result.get("files_processed", 0)

                logger.info(
                    f"Successfully processed {files_processed} files for Neo4j: "
                    f"{entities_created} entities created"
                )
            else:
                logger.warning(
                    f"Batch processing returned no results for {analyses_count} analyses"
                )

            # Clean up processed analyses
            delattr(self, "_neo4j_analyses")
            if hasattr(self, "_neo4j_repo_name"):
                delattr(self, "_neo4j_repo_name")

            return processing_result

        except Exception as e:
            logger.error(f"Error in batch Neo4j processing: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

            # Clean up anyway
            if hasattr(self, "_neo4j_analyses"):
                delattr(self, "_neo4j_analyses")
            if hasattr(self, "_neo4j_repo_name"):
                delattr(self, "_neo4j_repo_name")

            # RE-RAISE the exception so we can see what's actually failing
            raise

    def _chunk_content(self, content: str, chunk_size: int) -> List[str]:
        """
        Chunk content into smaller pieces for processing.

        Args:
            content: Text content to chunk
            chunk_size: Maximum size of each chunk

        Returns:
            List of content chunks
        """
        if len(content) <= chunk_size:
            return [content] if content.strip() else []

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # Try to break at sentence boundaries
            if end < len(content):
                # Look for sentence break within last 100 chars
                sentence_break = content.rfind(".", start, end)
                if sentence_break > start + chunk_size * 0.8:
                    end = sentence_break + 1

            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end

        return chunks

    def _detect_file_language(self, file_path: Path) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to file

        Returns:
            Detected language string
        """
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".go": "go",
            ".rs": "rust",
            ".kt": "kotlin",
            ".swift": "swift",
            ".scala": "scala",
            ".md": "markdown",
            ".txt": "text",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",
        }

        return extension_map.get(file_path.suffix.lower(), "text")

    def _is_code_file(self, file_path: Path) -> bool:
        """
        Check if file is a code file suitable for Neo4j processing.

        Args:
            file_path: Path to file

        Returns:
            True if file is a code file, False otherwise
        """
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".go",
            ".rs",
            ".kt",
            ".swift",
            ".scala",
        }

        return file_path.suffix.lower() in code_extensions

    async def _estimate_kg_entities(self, content: str, language: str) -> int:
        """
        Estimate number of entities that would be created in knowledge graph.

        Args:
            content: File content
            language: Programming language

        Returns:
            Estimated number of entities
        """
        # Simple heuristic based on common patterns
        lines = content.split("\n")
        entity_count = 0

        for line in lines:
            line = line.strip()

            # Count function/method definitions
            if language == "python":
                if line.startswith("def ") or line.startswith("class "):
                    entity_count += 1
            elif language in ["javascript", "typescript"]:
                if "function" in line or "const " in line or "let " in line:
                    entity_count += 1
            elif language == "java":
                if "public " in line or "private " in line or "protected " in line:
                    entity_count += 1

        return entity_count

    async def cleanup(self):
        """Clean up service resources."""
        cleanup_errors = []

        # Clean up legacy executor if it exists
        if hasattr(self, "executor") and self.executor:
            try:
                self.executor.shutdown(wait=True)
                logger.debug("Legacy ThreadPoolExecutor shutdown completed")
            except Exception as e:
                cleanup_errors.append(f"Legacy executor cleanup error: {e}")
                logger.error(f"Error shutting down legacy executor: {e}")

        # Note: Executors from context are managed by the context itself
        # We don't shut them down here to avoid interfering with other services
        if self.context:
            logger.debug("Executors managed by context, skipping direct cleanup")
        elif self.io_executor or self.cpu_executor:
            logger.warning(
                "Executors present but no context - this may indicate a resource leak"
            )

        if cleanup_errors:
            logger.warning(
                f"Service cleanup completed with {len(cleanup_errors)} errors"
            )
        else:
            logger.info("Unified indexing service cleanup completed successfully")


# Module-level convenience functions
async def process_repository_unified(
    repo_url: str,
    destination: IndexingDestination = IndexingDestination.BOTH,
    file_types: List[str] = None,
    max_files: int = 50,
    chunk_size: int = 5000,
    max_size_mb: int = 500,
) -> UnifiedIndexingResponse:
    """
    Convenience function for unified repository processing.

    Args:
        repo_url: GitHub repository URL
        destination: Where to store processed data (qdrant, neo4j, or both)
        file_types: File extensions to process
        max_files: Maximum number of files to process
        chunk_size: Chunk size for RAG processing
        max_size_mb: Maximum repository size limit

    Returns:
        UnifiedIndexingResponse with processing results
    """
    if file_types is None:
        file_types = [".md", ".py", ".js", ".ts", ".java"]

    request = UnifiedIndexingRequest(
        repo_url=repo_url,
        destination=destination,
        file_types=file_types,
        max_files=max_files,
        chunk_size=chunk_size,
        max_size_mb=max_size_mb,
    )

    service = UnifiedIndexingService()
    try:
        return await service.process_repository_unified(request)
    finally:
        await service.cleanup()


# Initialize event loop configuration on import
setup_event_loop()
