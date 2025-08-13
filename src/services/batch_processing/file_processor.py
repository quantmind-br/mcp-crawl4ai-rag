"""
Module-level file processing functions for ProcessPoolExecutor compatibility.

This module provides functions that can be pickled and sent to worker processes
for CPU-bound operations like tree-sitter parsing. All functions are module-level
to ensure they can be serialized for multiprocessing.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


def parse_file_for_kg(
    file_path: str, content: str, language: str, repo_name: str
) -> Optional[Dict[str, Any]]:
    """
    Parse a file for knowledge graph extraction using tree-sitter.

    This function recreates parsers in each worker process to handle
    the fact that tree-sitter objects are not pickle-serializable.

    Args:
        file_path: Path to the file being parsed
        content: File content as string
        language: Programming language for parsing
        repo_name: Repository name for context

    Returns:
        Dictionary containing parsed analysis data or None if parsing fails
    """
    try:
        # Import tree-sitter modules in the worker process
        from ...k_graph.parsing.parser_factory import get_global_factory

        # Create parser factory in this process
        factory = get_global_factory()
        parser = factory.get_parser(language)

        if not parser:
            logger.debug(f"No parser available for language: {language}")
            return None

        # Parse the file
        result = parser.parse(content, file_path)

        if not result:
            logger.debug(f"No parse result for file: {file_path}")
            return None

        # Convert to serializable dictionary format
        analysis_data = {
            "file_path": file_path,
            "language": language,
            "repo_name": repo_name,
            "classes": [],
            "functions": [],
            "imports": result.imports or [],
            "line_count": result.line_count or 0,
        }

        # Convert classes to dictionaries
        if result.classes:
            for class_name, class_obj in result.classes.items():
                class_dict = {
                    "name": class_obj.name,
                    "full_name": class_obj.full_name,
                    "line_start": class_obj.line_start,
                    "line_end": class_obj.line_end,
                    "docstring": class_obj.docstring,
                    "methods": [],
                }

                # Convert methods to dictionaries
                for method in class_obj.methods:
                    method_dict = {
                        "name": method.name,
                        "params": method.params,
                        "return_type": method.return_type,
                        "line_start": method.line_start,
                        "line_end": method.line_end,
                        "docstring": method.docstring,
                    }
                    class_dict["methods"].append(method_dict)

                analysis_data["classes"].append(class_dict)

        # Convert functions to dictionaries
        if result.functions:
            for func_name, func_obj in result.functions.items():
                func_dict = {
                    "name": func_obj.name,
                    "full_name": func_obj.full_name,
                    "params": func_obj.params,
                    "return_type": func_obj.return_type,
                    "line_start": func_obj.line_start,
                    "line_end": func_obj.line_end,
                    "docstring": func_obj.docstring,
                }
                analysis_data["functions"].append(func_dict)

        logger.debug(
            f"Successfully parsed {file_path}: "
            f"{len(analysis_data['classes'])} classes, "
            f"{len(analysis_data['functions'])} functions"
        )

        return analysis_data

    except Exception as e:
        logger.error(f"Error parsing {file_path} in worker process: {e}")
        return None


def read_file_async_sync(file_path: str) -> Tuple[str, str]:
    """
    Synchronous file reading function for use in ProcessPoolExecutor.

    This function is named with '_sync' suffix to distinguish it from the
    async version and to indicate it's designed for synchronous execution
    in worker processes.

    Args:
        file_path: Path to file to read

    Returns:
        Tuple of (file_path, content) or (file_path, "") if reading fails
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return file_path, content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return file_path, ""


def process_file_batch_sync(file_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Process a batch of files synchronously for reading in ProcessPoolExecutor.

    Args:
        file_paths: List of file paths to read

    Returns:
        List of (file_path, content) tuples
    """
    results = []
    for file_path in file_paths:
        file_path_str, content = read_file_async_sync(file_path)
        results.append((file_path_str, content))
    return results


def detect_file_language(file_path: str) -> str:
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

    path = Path(file_path)
    return extension_map.get(path.suffix.lower(), "text")


def is_code_file(file_path: str) -> bool:
    """
    Check if file is a code file suitable for knowledge graph processing.

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

    path = Path(file_path)
    return path.suffix.lower() in code_extensions


async def read_file_async(file_path: str) -> Tuple[str, str]:
    """
    Read file asynchronously using aiofiles.

    Args:
        file_path: Path to file to read

    Returns:
        Tuple of (file_path, content)
    """
    try:
        import aiofiles

        async with aiofiles.open(
            file_path, "r", encoding="utf-8", errors="ignore"
        ) as f:
            content = await f.read()
        return file_path, content
    except Exception as e:
        logger.error(f"Error reading file {file_path} async: {e}")
        return file_path, ""


async def process_file_batch(
    file_paths: List[str],
    cpu_executor: ProcessPoolExecutor,
    should_process_kg: bool = True,
    repo_name: str = "unknown",
) -> List[Dict[str, Any]]:
    """
    Process a batch of files using both I/O and CPU operations.

    This function coordinates:
    1. Async file reading (I/O-bound)
    2. CPU-bound parsing using ProcessPoolExecutor

    Args:
        file_paths: List of file paths to process
        cpu_executor: ProcessPoolExecutor for CPU-bound operations
        should_process_kg: Whether to perform knowledge graph processing
        repo_name: Repository name for context

    Returns:
        List of processing results for each file
    """
    results = []

    # Stage 1: Read all files asynchronously
    logger.debug(f"Reading {len(file_paths)} files asynchronously")
    read_tasks = [read_file_async(fp) for fp in file_paths]
    file_contents = await asyncio.gather(*read_tasks, return_exceptions=True)

    # Stage 2: Process files for knowledge graph if requested
    if should_process_kg:
        logger.debug(f"Processing {len(file_paths)} files for knowledge graph")

        # Prepare arguments for CPU-bound processing
        parse_tasks = []
        for i, (file_path, content) in enumerate(file_contents):
            if isinstance(content, Exception):
                logger.error(f"Failed to read {file_path}: {content}")
                continue

            if not content.strip():
                continue

            if not is_code_file(file_path):
                continue

            language = detect_file_language(file_path)

            # Submit to ProcessPoolExecutor
            future = cpu_executor.submit(
                parse_file_for_kg, file_path, content, language, repo_name
            )
            parse_tasks.append((future, file_path, content))

        # Collect results as they complete
        for future, file_path, content in parse_tasks:
            try:
                # Wait for the future to complete
                kg_result = future.result(timeout=60)  # 60 second timeout per file

                result = {
                    "file_path": file_path,
                    "content": content,
                    "language": detect_file_language(file_path),
                    "kg_analysis": kg_result,
                    "processed_for_kg": kg_result is not None,
                    "processing_errors": [],
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {file_path} for KG: {e}")
                result = {
                    "file_path": file_path,
                    "content": content,
                    "language": detect_file_language(file_path),
                    "kg_analysis": None,
                    "processed_for_kg": False,
                    "processing_errors": [f"KG processing error: {str(e)}"],
                }
                results.append(result)
    else:
        # Just return file contents without KG processing
        for file_path, content in file_contents:
            if isinstance(content, Exception):
                logger.error(f"Failed to read {file_path}: {content}")
                continue

            result = {
                "file_path": file_path,
                "content": content,
                "language": detect_file_language(file_path),
                "kg_analysis": None,
                "processed_for_kg": False,
                "processing_errors": [],
            }
            results.append(result)

    logger.debug(f"Completed processing {len(results)} files in batch")
    return results
