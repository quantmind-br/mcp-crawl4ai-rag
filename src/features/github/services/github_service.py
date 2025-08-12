"""
Main GitHub repository processing service.

This module provides the primary orchestration service for GitHub repository
processing, coordinating between cloning, discovery, processing, and cleanup operations.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from ..repository.git_operations import GitRepository
from ..repository.metadata_extractor import MetadataExtractor
from ..discovery.multi_file_discovery import MultiFileDiscovery
from ..processors.processor_factory import ProcessorFactory
from ..core.exceptions import DiscoveryError
from ..core.models import (
    FileInfo,
    ProcessingResult,
    CloneResult,
    DiscoveryResult,
    ServiceStatistics,
    ProcessingConfig,
)
from ..config.settings import GitHubProcessorConfig, get_default_config


class GitHubService:
    """
    Main GitHub repository processing service.

    This service orchestrates the complete processing pipeline:
    1. Repository cloning and validation
    2. Metadata extraction
    3. File discovery and filtering
    4. Content processing using specialized processors
    5. Cleanup and resource management
    """

    def __init__(self, config: GitHubProcessorConfig = None):
        """
        Initialize GitHub processing service.

        Args:
            config: Service configuration (uses defaults if None)
        """
        self.config = config or get_default_config()

        # Initialize core components
        self.git_repo = GitRepository(self.config.git)
        self.metadata_extractor = MetadataExtractor()
        self.file_discovery = MultiFileDiscovery(self.config.discovery)
        self.processor_factory = ProcessorFactory()

        # Statistics tracking
        self.stats = {
            "repositories_processed": 0,
            "files_processed": 0,
            "total_processing_time": 0.0,
            "successful_operations": 0,
            "failed_operations": 0,
        }

        self.logger = logging.getLogger(__name__)

    async def process_repository(
        self, repo_url: str, processing_config: ProcessingConfig = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Process a complete GitHub repository.

        Args:
            repo_url: GitHub repository URL
            processing_config: Processing configuration
            **kwargs: Additional processing parameters

        Returns:
            Dictionary with comprehensive processing results
        """
        start_time = time.time()
        config = processing_config or ProcessingConfig()

        self.logger.info(f"Starting repository processing: {repo_url}")

        try:
            # Step 1: Clone repository
            clone_result = await self._clone_repository(repo_url, config.max_size_mb)
            if not clone_result.success:
                return self._create_error_result(
                    "clone_failed", clone_result.error, repo_url
                )

            repo_path = clone_result.temp_directory
            metadata = clone_result.metadata

            # Step 2: Discover files
            discovery_result = await self._discover_files(
                repo_path, config.file_types, config.max_files, config.filter_criteria
            )

            # Step 3: Process files
            processing_results = await self._process_files(
                discovery_result.discovered_files, repo_path, config.chunk_size
            )

            # Step 4: Compile results
            processing_time = time.time() - start_time
            final_result = self._compile_results(
                clone_result=clone_result,
                discovery_result=discovery_result,
                processing_results=processing_results,
                processing_time=processing_time,
                config=config,
            )

            # Update statistics
            self._update_statistics(True, processing_time, len(processing_results))

            self.logger.info(
                f"Repository processing completed: {repo_url} "
                f"({len(processing_results)} files in {processing_time:.2f}s)"
            )

            return final_result

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_statistics(False, processing_time, 0)

            self.logger.error(f"Repository processing failed for {repo_url}: {e}")
            return self._create_error_result("processing_failed", str(e), repo_url)

    async def _clone_repository(self, repo_url: str, max_size_mb: int) -> CloneResult:
        """Clone repository and extract metadata."""
        try:
            # Clone the repository
            temp_directory = self.git_repo.clone_repository(repo_url, max_size_mb)

            # Extract metadata
            metadata = self.metadata_extractor.extract_repo_metadata(
                repo_url, temp_directory
            )

            # Calculate size
            size_mb = self.git_repo._get_directory_size_mb(temp_directory)

            return self.git_repo.create_clone_result(
                success=True,
                repo_url=repo_url,
                temp_directory=temp_directory,
                metadata=metadata,
                size_mb=size_mb,
            )

        except Exception as e:
            return self.git_repo.create_clone_result(
                success=False, repo_url=repo_url, error=str(e)
            )

    async def _discover_files(
        self,
        repo_path: str,
        file_types: List[str],
        max_files: int,
        filter_criteria: Optional[Any] = None,
    ) -> DiscoveryResult:
        """Discover and filter files in the repository."""
        try:
            start_time = time.time()

            # Discover files
            discovered_files = self.file_discovery.discover_files(
                repo_path=repo_path, file_types=file_types, max_files=max_files
            )

            discovery_time = time.time() - start_time

            return self.file_discovery.create_discovery_result(
                repo_path=repo_path,
                discovered_files=discovered_files,
                file_types=file_types,
                discovery_time=discovery_time,
            )

        except Exception as e:
            raise DiscoveryError(f"File discovery failed: {e}", repo_path)

    async def _process_files(
        self, file_infos: List[FileInfo], repo_path: str, chunk_size: int
    ) -> List[ProcessingResult]:
        """Process discovered files using appropriate processors."""
        processing_results = []

        for file_info in file_infos:
            try:
                start_time = time.time()

                # Get appropriate processor
                processor = self.processor_factory.get_processor_for_file(
                    file_info.path
                )
                if not processor:
                    self.logger.debug(f"No processor available for {file_info.path}")
                    continue

                # Process the file
                processed_chunks = processor.process_file(
                    file_path=file_info.path,
                    relative_path=file_info.relative_path,
                    chunk_size=chunk_size,
                )

                processing_time = time.time() - start_time

                # Create processing result
                result = processor.create_processing_result(
                    file_path=file_info.path,
                    relative_path=file_info.relative_path,
                    processed_chunks=processed_chunks,
                    success=True,
                    processing_time=processing_time,
                )

                processing_results.append(result)

            except Exception as e:
                self.logger.warning(f"Processing failed for {file_info.path}: {e}")

                # Create error result
                error_result = ProcessingResult(
                    file_path=file_info.path,
                    relative_path=file_info.relative_path,
                    success=False,
                    processed_chunks=[],
                    error=str(e),
                )
                processing_results.append(error_result)

        return processing_results

    def _compile_results(
        self,
        clone_result: CloneResult,
        discovery_result: DiscoveryResult,
        processing_results: List[ProcessingResult],
        processing_time: float,
        config: ProcessingConfig,
    ) -> Dict[str, Any]:
        """Compile comprehensive processing results."""
        successful_files = [r for r in processing_results if r.success]
        failed_files = [r for r in processing_results if not r.success]

        # Calculate total chunks processed
        total_chunks = sum(len(r.processed_chunks) for r in successful_files)

        return {
            "success": True,
            "repo_url": clone_result.repo_url,
            "metadata": clone_result.metadata.__dict__ if clone_result.metadata else {},
            "repository_info": {
                "size_mb": clone_result.size_mb,
                "temp_directory": clone_result.temp_directory,
            },
            "discovery_summary": {
                "files_discovered": len(discovery_result.discovered_files),
                "file_types": discovery_result.file_types,
                "max_files_limit": discovery_result.max_files,
                "discovery_time": discovery_result.discovery_time,
            },
            "processing_summary": {
                "files_processed": len(processing_results),
                "files_successful": len(successful_files),
                "files_failed": len(failed_files),
                "total_chunks_processed": total_chunks,
                "processing_time": processing_time,
                "average_time_per_file": processing_time / len(processing_results)
                if processing_results
                else 0,
            },
            "configuration": {
                "file_types": config.file_types,
                "max_files": config.max_files,
                "chunk_size": config.chunk_size,
                "max_size_mb": config.max_size_mb,
            },
            "detailed_results": {
                "successful_files": [
                    {
                        "file_path": r.relative_path,
                        "chunks_count": len(r.processed_chunks),
                        "processing_time": r.processing_time,
                    }
                    for r in successful_files
                ],
                "failed_files": [
                    {
                        "file_path": r.relative_path,
                        "error": r.error,
                    }
                    for r in failed_files
                ]
                if failed_files
                else [],
            },
        }

    def _create_error_result(
        self, error_type: str, error_message: str, repo_url: str
    ) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "success": False,
            "error_type": error_type,
            "error": error_message,
            "repo_url": repo_url,
            "timestamp": time.time(),
        }

    def _update_statistics(
        self, success: bool, processing_time: float, files_processed: int
    ) -> None:
        """Update service statistics."""
        self.stats["repositories_processed"] += 1
        self.stats["files_processed"] += files_processed
        self.stats["total_processing_time"] += processing_time

        if success:
            self.stats["successful_operations"] += 1
        else:
            self.stats["failed_operations"] += 1

    def get_service_statistics(self) -> ServiceStatistics:
        """Get comprehensive service statistics."""
        supported_extensions = self.processor_factory.get_supported_extensions()
        processors = self.processor_factory.list_processors()

        return ServiceStatistics(
            supported_file_types=len(supported_extensions),
            file_extensions=supported_extensions,
            size_limits=self.config.processing.file_size_limits,
            processing_modes=["single_repository"],
            destinations=["local_processing"],
            temp_directories_active=len(self.git_repo.get_temp_dirs()),
            processors_registered=len(processors),
            total_files_processed=self.stats["files_processed"],
            total_repositories_processed=self.stats["repositories_processed"],
            average_processing_time=self._calculate_average_processing_time(),
        )

    def _calculate_average_processing_time(self) -> Optional[float]:
        """Calculate average processing time per repository."""
        if self.stats["repositories_processed"] == 0:
            return None
        return (
            self.stats["total_processing_time"] / self.stats["repositories_processed"]
        )

    def cleanup(self) -> None:
        """Clean up all temporary resources."""
        try:
            self.git_repo.cleanup()
            self.logger.info("Service cleanup completed")
        except Exception as e:
            self.logger.error(f"Service cleanup failed: {e}")

    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file extensions."""
        return self.processor_factory.get_supported_extensions()

    def get_processors(self) -> List[str]:
        """Get list of available processors."""
        return self.processor_factory.list_processors()

    # No backward-compatibility shims are kept in this service.
