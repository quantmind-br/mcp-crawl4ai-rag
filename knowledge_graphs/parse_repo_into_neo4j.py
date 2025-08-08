"""
Direct Neo4j GitHub Code Repository Extractor

Creates nodes and relationships directly in Neo4j without Graphiti:
- File nodes
- Class nodes
- Method nodes
- Function nodes
- Import relationships

Bypasses all LLM processing for maximum speed.
"""

import asyncio
import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Set
# ast import removed - using Tree-sitter exclusively

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

# Import Tree-sitter components
from knowledge_graphs.parser_factory import get_global_factory

# Load environment variables first
load_dotenv()

# Global semaphore to prevent concurrent Neo4j initialization
_neo4j_init_semaphore = asyncio.Semaphore(1)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_value = getattr(logging, log_level, logging.INFO)

# Configure basic logging format
logging.basicConfig(
    level=log_level_value,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # Force reconfiguration even if basicConfig was called before
)

# Ensure root logger level is set correctly
logging.getLogger().setLevel(log_level_value)
logger = logging.getLogger(__name__)


class Neo4jCodeAnalyzer:
    """Multi-language code analyzer for direct Neo4j insertion using Tree-sitter"""

    def __init__(self):
        # Initialize parser factory
        self.parser_factory = get_global_factory()

        # External modules to ignore
        self.external_modules = {
            # Python standard library
            "os",
            "sys",
            "json",
            "logging",
            "datetime",
            "pathlib",
            "typing",
            "collections",
            "asyncio",
            "subprocess",
            "ast",
            "re",
            "string",
            "urllib",
            "http",
            "email",
            "time",
            "uuid",
            "hashlib",
            "base64",
            "itertools",
            "functools",
            "operator",
            "contextlib",
            "copy",
            "pickle",
            "tempfile",
            "shutil",
            "glob",
            "fnmatch",
            "io",
            "codecs",
            "locale",
            "platform",
            "socket",
            "ssl",
            "threading",
            "queue",
            "multiprocessing",
            "concurrent",
            "warnings",
            "traceback",
            "inspect",
            "importlib",
            "pkgutil",
            "types",
            "weakref",
            "gc",
            "dataclasses",
            "enum",
            "abc",
            "numbers",
            "decimal",
            "fractions",
            "math",
            "cmath",
            "random",
            "statistics",
            # Common third-party libraries
            "requests",
            "urllib3",
            "httpx",
            "aiohttp",
            "flask",
            "django",
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "alembic",
            "psycopg2",
            "pymongo",
            "redis",
            "celery",
            "pytest",
            "unittest",
            "mock",
            "faker",
            "factory",
            "hypothesis",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scipy",
            "sklearn",
            "torch",
            "tensorflow",
            "keras",
            "opencv",
            "pillow",
            "boto3",
            "botocore",
            "azure",
            "google",
            "openai",
            "anthropic",
            "langchain",
            "transformers",
            "huggingface_hub",
            "click",
            "typer",
            "rich",
            "colorama",
            "tqdm",
            "python-dotenv",
            "pyyaml",
            "toml",
            "configargparse",
            "marshmallow",
            "attrs",
            "dataclasses-json",
            "jsonschema",
            "cerberus",
            "voluptuous",
            "schema",
            "jinja2",
            "mako",
            "cryptography",
            "bcrypt",
            "passlib",
            "jwt",
            "authlib",
            "oauthlib",
        }

    def analyze_file(
        self, file_path: Path, repo_root: Path, project_modules: Set[str]
    ) -> Dict[str, Any]:
        """Multi-language file analysis using Tree-sitter parsers"""
        try:
            # Get appropriate parser for this file
            parser = self.parser_factory.get_parser_for_file(str(file_path))
            if not parser:
                logger.debug(f"No parser available for {file_path}")
                return None

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse using Tree-sitter
            result = parser.parse(content, str(file_path))

            if result.errors:
                logger.warning(f"Parse errors in {file_path}: {result.errors}")
                if len(result.errors) > 3:  # Too many errors, skip this file
                    return None

            # Filter imports for internal modules (language-specific logic)
            filtered_imports = []
            if parser.get_language_name() == "python":
                # Use Python-specific import filtering
                filtered_imports = [
                    imp
                    for imp in result.imports
                    if self._is_likely_internal_python(imp, project_modules)
                ]
            else:
                # For other languages, use simpler filtering for now
                filtered_imports = [
                    imp
                    for imp in result.imports
                    if self._is_likely_internal_generic(imp, project_modules)
                ]

            # Tree-sitter already returns dictionary format, just ensure 'args' field for compatibility
            processed_classes = []
            for cls in result.classes:
                # Ensure methods have 'args' field for Neo4j compatibility
                processed_methods = []
                for method in cls.get("methods", []):
                    if "args" not in method:
                        method["args"] = method.get("params", [])
                    processed_methods.append(method)

                cls["methods"] = processed_methods
                processed_classes.append(cls)

            # Ensure functions have 'args' field for Neo4j compatibility
            processed_functions = []
            for func in result.functions:
                if "args" not in func:
                    func["args"] = func.get("params", [])
                processed_functions.append(func)

            return {
                "module_name": result.module_name,
                "file_path": str(file_path.relative_to(repo_root)),
                "classes": processed_classes,
                "functions": processed_functions,
                "imports": filtered_imports,
                "line_count": result.line_count,
                "language": result.language,
                "errors": result.errors,
            }

        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
            return None

    # AST-based Python parser removed - using Tree-sitter exclusively

    def _is_likely_internal(self, import_name: str, project_modules: Set[str]) -> bool:
        """Check if an import is likely internal to the project"""
        if not import_name:
            return False

        # Relative imports are definitely internal
        if import_name.startswith("."):
            return True

        # Check if it's a known external module
        base_module = import_name.split(".")[0]
        if base_module in self.external_modules:
            return False

        # Check if it matches any project module
        for project_module in project_modules:
            if import_name.startswith(project_module):
                return True

        # If it's not obviously external, consider it internal
        if (
            not any(ext in base_module.lower() for ext in ["test", "mock", "fake"])
            and not base_module.startswith("_")
            and len(base_module) > 2
        ):
            return True

        return False

    def _is_likely_internal_python(
        self, import_name: str, project_modules: Set[str]
    ) -> bool:
        """Python-specific import filtering (reuses existing logic)"""
        return self._is_likely_internal(import_name, project_modules)

    def _is_likely_internal_generic(
        self, import_name: str, project_modules: Set[str]
    ) -> bool:
        """Generic import filtering for non-Python languages"""
        if not import_name:
            return False

        # Remove common prefixes/suffixes for different languages
        clean_import = import_name.strip().strip("\"';")

        # Check if it's a relative import (language-agnostic patterns)
        if clean_import.startswith("./") or clean_import.startswith("../"):
            return True

        # Check against project modules
        base_module = clean_import.split(".")[0].split("/")[0]
        for project_module in project_modules:
            if clean_import.startswith(project_module) or base_module == project_module:
                return True

        # Heuristic: if it doesn't look like a standard library, consider it internal
        common_external = {
            "react",
            "vue",
            "angular",
            "lodash",
            "axios",
            "express",
            "fs",
            "path",
            "java.util",
            "java.io",
            "javax",
            "org.springframework",
            "com.google",
            "std",
            "core",
            "alloc",
            "collections",
            "serde",
            "tokio",
            "System",
            "Microsoft",
            "Newtonsoft",
            "Entity",
            "fmt",
            "net/http",
            "encoding/json",
            "os",
            "io",
            "jquery",
            "bootstrap",
            "moment",
            "uuid",
        }

        if any(ext in clean_import.lower() for ext in common_external):
            return False

        # If unsure and it looks reasonable, consider it internal
        return len(base_module) > 2 and not base_module.startswith("_")

    def _get_importable_module_name(
        self, file_path: Path, repo_root: Path, relative_path: str
    ) -> str:
        """Determine the actual importable module name for a Python file"""
        # Start with the default: convert file path to module path
        default_module = (
            relative_path.replace("/", ".").replace("\\", ".").replace(".py", "")
        )

        # Common patterns to detect the actual package root
        path_parts = Path(relative_path).parts

        # Look for common package indicators
        package_roots = []

        # Check each directory level for __init__.py to find package boundaries
        current_path = repo_root
        for i, part in enumerate(path_parts[:-1]):  # Exclude the .py file itself
            current_path = current_path / part
            if (current_path / "__init__.py").exists():
                # This is a package directory, mark it as a potential root
                package_roots.append(i)

        if package_roots:
            # Use the first (outermost) package as the root
            package_start = package_roots[0]
            module_parts = path_parts[package_start:]
            module_name = ".".join(module_parts).replace(".py", "")
            return module_name

        # Fallback: look for common Python project structures
        # Skip common non-package directories
        skip_dirs = {"src", "lib", "source", "python", "pkg", "packages"}

        # Find the first directory that's not in skip_dirs
        filtered_parts = []
        for part in path_parts:
            if (
                part.lower() not in skip_dirs or filtered_parts
            ):  # Once we start including, include everything
                filtered_parts.append(part)

        if filtered_parts:
            module_name = ".".join(filtered_parts).replace(".py", "")
            return module_name

        # Final fallback: use the default
        return default_module

        # AST helper methods removed - using Tree-sitter exclusively

        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                if hasattr(node, "value"):
                    return f"{self._get_name(node.value)}.{node.attr}"
                else:
                    return node.attr
            elif isinstance(node, ast.Subscript):
                # Handle List[Type], Dict[K,V], etc.
                base = self._get_name(node.value)
                if hasattr(node, "slice"):
                    if isinstance(node.slice, ast.Name):
                        return f"{base}[{node.slice.id}]"
                    elif isinstance(node.slice, ast.Tuple):
                        elts = [self._get_name(elt) for elt in node.slice.elts]
                        return f"{base}[{', '.join(elts)}]"
                    elif isinstance(node.slice, ast.Constant):
                        return f"{base}[{repr(node.slice.value)}]"
                    elif isinstance(node.slice, ast.Attribute):
                        return f"{base}[{self._get_name(node.slice)}]"
                    elif isinstance(node.slice, ast.Subscript):
                        return f"{base}[{self._get_name(node.slice)}]"
                    else:
                        # Try to get the name of the slice, fallback to Any if it fails
                        try:
                            slice_name = self._get_name(node.slice)
                            return f"{base}[{slice_name}]"
                        except Exception:
                            return f"{base}[Any]"
                return base
            elif isinstance(node, ast.Constant):
                return str(node.value)
            elif isinstance(node, ast.Str):  # Python < 3.8
                return f'"{node.s}"'
            elif isinstance(node, ast.Tuple):
                elts = [self._get_name(elt) for elt in node.elts]
                return f"({', '.join(elts)})"
            elif isinstance(node, ast.List):
                elts = [self._get_name(elt) for elt in node.elts]
                return f"[{', '.join(elts)}]"
            else:
                # Fallback for complex types - return a simple string representation
                return "Any"
        except Exception:
            # If anything goes wrong, return a safe default
            return "Any"


class DirectNeo4jExtractor:
    """Creates nodes and relationships directly in Neo4j"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        self.analyzer = Neo4jCodeAnalyzer()

    async def initialize(self):
        """Initialize Neo4j connection with deadlock prevention"""
        # Use semaphore to prevent concurrent initialization causing deadlocks
        async with _neo4j_init_semaphore:
            logger.info("Initializing Neo4j connection...")

            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
                )

                # Test connection first
                async with self.driver.session() as session:
                    await session.run("RETURN 1")

                # Create constraints and indexes (protected by semaphore)
                logger.info("Creating constraints and indexes...")
                async with self.driver.session() as session:
                    # Create constraints - using MERGE-friendly approach
                    await session.run(
                        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE"
                    )
                    await session.run(
                        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.full_name IS UNIQUE"
                    )
                    # Remove unique constraints for methods/attributes since they can be duplicated across classes
                    # await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.full_name IS UNIQUE")
                    # await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Function) REQUIRE f.full_name IS UNIQUE")
                    # await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Attribute) REQUIRE a.full_name IS UNIQUE")

                    # Create indexes for performance
                    await session.run(
                        "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.name)"
                    )
                    await session.run(
                        "CREATE INDEX IF NOT EXISTS FOR (c:Class) ON (c.name)"
                    )
                    await session.run(
                        "CREATE INDEX IF NOT EXISTS FOR (m:Method) ON (m.name)"
                    )

                logger.info("Neo4j initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Neo4j: {e}")
                if self.driver:
                    await self.driver.close()
                    self.driver = None
                raise

    async def clear_repository_data(self, repo_name: str):
        """Clear all data for a specific repository"""
        logger.info(f"Clearing existing data for repository: {repo_name}")
        async with self.driver.session() as session:
            # Delete in specific order to avoid constraint issues

            # 1. Delete methods and attributes (they depend on classes)
            await session.run(
                """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
                DETACH DELETE m
            """,
                repo_name=repo_name,
            )

            await session.run(
                """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
                DETACH DELETE a
            """,
                repo_name=repo_name,
            )

            # 2. Delete functions (they depend on files)
            await session.run(
                """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
                DETACH DELETE func
            """,
                repo_name=repo_name,
            )

            # 3. Delete classes (they depend on files)
            await session.run(
                """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
                DETACH DELETE c
            """,
                repo_name=repo_name,
            )

            # 4. Delete files (they depend on repository)
            await session.run(
                """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
                DETACH DELETE f
            """,
                repo_name=repo_name,
            )

            # 5. Finally delete the repository
            await session.run(
                """
                MATCH (r:Repository {name: $repo_name})
                DETACH DELETE r
            """,
                repo_name=repo_name,
            )

        logger.info(f"Cleared data for repository: {repo_name}")

    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()

    def clone_repo(self, repo_url: str, target_dir: str) -> str:
        """Clone repository with Windows long path support and selective file exclusion"""
        logger.info(f"Cloning repository to: {target_dir}")

        if os.path.exists(target_dir):
            logger.info(f"Removing existing directory: {target_dir}")
            try:

                def handle_remove_readonly(func, path, exc):
                    try:
                        if os.path.exists(path):
                            os.chmod(path, 0o777)
                            func(path)
                    except PermissionError:
                        logger.warning(
                            f"Could not remove {path} - file in use, skipping"
                        )
                        pass

                shutil.rmtree(target_dir, onerror=handle_remove_readonly)
            except Exception as e:
                logger.warning(
                    f"Could not fully remove {target_dir}: {e}. Proceeding anyway..."
                )

        try:
            logger.info(f"Running git clone from {repo_url}")

            # First, try normal clone with Windows long path support
            clone_cmd = ["git", "clone", "--depth", "1", repo_url, target_dir]

            # On Windows, enable long path support
            if os.name == "nt":  # Windows
                # Configure git for long paths before cloning
                try:
                    subprocess.run(
                        ["git", "config", "--global", "core.longpaths", "true"],
                        check=False,
                        capture_output=True,
                    )
                    logger.info("Enabled git long paths support")
                except Exception as e:
                    logger.warning(f"Could not enable git long paths: {e}")

            # Attempt to clone
            result = subprocess.run(
                clone_cmd, check=False, capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.warning(f"Standard clone failed: {result.stderr}")

                # If clone failed, try with sparse-checkout to exclude problematic directories
                logger.info(
                    "Attempting clone with sparse-checkout to exclude test files..."
                )

                # Remove partial clone if it exists
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir, onerror=handle_remove_readonly)

                # Clone with no-checkout first
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--no-checkout",
                        repo_url,
                        target_dir,
                    ],
                    check=True,
                )

                # Navigate to repo directory and setup sparse-checkout
                original_cwd = os.getcwd()
                try:
                    os.chdir(target_dir)

                    # Enable sparse-checkout
                    subprocess.run(
                        ["git", "config", "core.sparseCheckout", "true"], check=True
                    )

                    # Create sparse-checkout file to exclude problematic test directories
                    sparse_file = os.path.join(".git", "info", "sparse-checkout")
                    os.makedirs(os.path.dirname(sparse_file), exist_ok=True)

                    with open(sparse_file, "w") as f:
                        f.write("/*\n")  # Include everything by default
                        f.write(
                            "!tests/baselines/reference/tsserver/configuredProjects/\n"
                        )  # Exclude problematic dirs
                        f.write(
                            "!tests/baselines/reference/tsserver/projectReferences/\n"
                        )
                        f.write(
                            "!tests/baselines/reference/*/\n"
                        )  # Exclude other long test paths

                    # Checkout with sparse-checkout rules
                    subprocess.run(["git", "checkout", "HEAD"], check=True)
                    logger.info("Repository cloned successfully with sparse-checkout")

                finally:
                    os.chdir(original_cwd)
            else:
                logger.info("Repository cloned successfully")

        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e}")
            raise RuntimeError(f"Failed to clone repository: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during clone: {e}")
            raise RuntimeError(f"Failed to clone repository: {e}")

        return target_dir

    def get_source_files(self, repo_path: str) -> List[Path]:
        """Get source files for all supported languages, focusing on main source directories"""
        source_files = []
        exclude_dirs = {
            "tests",
            "test",
            "__pycache__",
            ".git",
            "venv",
            "env",
            "node_modules",
            "build",
            "dist",
            ".pytest_cache",
            "docs",
            "documentation",
            "examples",
            "example",
            "demo",
            "benchmark",
            "target",  # Rust build dir
            "bin",  # Go/C build dir
            ".gradle",  # Java build system
            ".maven",  # Java build system
        }

        # Extensions are checked via parser factory's is_supported_file method

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [
                d for d in dirs if d not in exclude_dirs and not d.startswith(".")
            ]

            for file in files:
                file_path = Path(root) / file

                # Check if file is supported by any parser
                if self.analyzer.parser_factory.is_supported_file(str(file_path)):
                    # Skip test files (multiple patterns for different languages)
                    if any(
                        pattern in file.lower()
                        for pattern in [
                            "test_",
                            "_test",
                            ".test.",
                            "spec_",
                            "_spec",
                            ".spec.",
                            "setup.py",
                            "conftest.py",
                            "__pycache__",
                        ]
                    ):
                        continue

                    # Size limit (avoid very large files)
                    if file_path.stat().st_size < 500_000:
                        source_files.append(file_path)

        return source_files

    async def analyze_repository(self, repo_url: str, temp_dir: str = None):
        """Analyze repository and create nodes/relationships in Neo4j"""
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        logger.info(f"Analyzing repository: {repo_name}")

        # Clear existing data for this repository before re-processing
        await self.clear_repository_data(repo_name)

        # Set default temp_dir to repos folder at script level
        if temp_dir is None:
            script_dir = Path(__file__).parent
            temp_dir = str(script_dir / "repos" / repo_name)

        # Clone and analyze
        repo_path = Path(self.clone_repo(repo_url, temp_dir))

        try:
            logger.info("Getting source files...")
            source_files = self.get_source_files(str(repo_path))
            logger.info(f"Found {len(source_files)} source files to analyze")

            # Group files by language for better logging
            files_by_language = {}
            for file_path in source_files:
                language = self.analyzer.parser_factory.detect_language(str(file_path))
                if language not in files_by_language:
                    files_by_language[language] = []
                files_by_language[language].append(file_path)

            logger.info("Files by language:")
            for lang, files in files_by_language.items():
                logger.info(f"  {lang}: {len(files)} files")

            # First pass: identify project modules (language-agnostic)
            logger.info("Identifying project modules...")
            project_modules = set()
            for file_path in source_files:
                relative_path = str(file_path.relative_to(repo_path))
                # Extract potential module names from path structure
                path_parts = relative_path.replace("\\", "/").split("/")
                if len(path_parts) > 0 and not path_parts[0].startswith("."):
                    project_modules.add(path_parts[0])
                    # Also add subdirectories as potential modules
                    if len(path_parts) > 1:
                        project_modules.add(
                            f"{path_parts[0]}.{path_parts[1].split('.')[0]}"
                        )

            logger.info(f"Identified project modules: {sorted(project_modules)}")

            # Second pass: analyze files and collect data
            logger.info("Analyzing source files...")
            modules_data = []
            for i, file_path in enumerate(source_files):
                if i % 20 == 0:
                    language = self.analyzer.parser_factory.detect_language(
                        str(file_path)
                    )
                    logger.info(
                        f"Analyzing file {i + 1}/{len(source_files)} [{language}]: {file_path.name}"
                    )

                # Try new multi-language analysis first
                analysis = self.analyzer.analyze_file(
                    file_path, repo_path, project_modules
                )

                # Skip files that Tree-sitter cannot parse
                if not analysis:
                    logger.debug(f"Tree-sitter parser could not analyze: {file_path}")

                if analysis:
                    modules_data.append(analysis)

            logger.info(f"Found {len(modules_data)} files with content")

            # Create nodes and relationships in Neo4j
            logger.info("Creating nodes and relationships in Neo4j...")
            await self._create_graph(repo_name, modules_data)

            # Print summary with language breakdown
            total_classes = sum(len(mod["classes"]) for mod in modules_data)
            total_methods = sum(
                len(cls["methods"]) for mod in modules_data for cls in mod["classes"]
            )
            total_functions = sum(len(mod["functions"]) for mod in modules_data)
            total_imports = sum(len(mod["imports"]) for mod in modules_data)

            # Language breakdown
            lang_stats = {}
            for mod in modules_data:
                lang = mod.get("language", "unknown")
                if lang not in lang_stats:
                    lang_stats[lang] = {"files": 0, "classes": 0, "functions": 0}
                lang_stats[lang]["files"] += 1
                lang_stats[lang]["classes"] += len(mod["classes"])
                lang_stats[lang]["functions"] += len(mod["functions"])

            print(
                f"\\n=== Multi-Language Neo4j Repository Analysis for {repo_name} ==="
            )
            print(f"Files processed: {len(modules_data)}")
            print(f"Classes created: {total_classes}")
            print(f"Methods created: {total_methods}")
            print(f"Functions created: {total_functions}")
            print(f"Import relationships: {total_imports}")
            print("\\nLanguage breakdown:")
            for lang, stats in sorted(lang_stats.items()):
                print(
                    f"  {lang}: {stats['files']} files, {stats['classes']} classes, {stats['functions']} functions"
                )

            logger.info(f"Successfully created Neo4j graph for {repo_name}")

        finally:
            if os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                try:

                    def handle_remove_readonly(func, path, exc):
                        try:
                            if os.path.exists(path):
                                os.chmod(path, 0o777)
                                func(path)
                        except PermissionError:
                            logger.warning(
                                f"Could not remove {path} - file in use, skipping"
                            )
                            pass

                    shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
                    logger.info("Cleanup completed")
                except Exception as e:
                    logger.warning(
                        f"Cleanup failed: {e}. Directory may remain at {temp_dir}"
                    )
                    # Don't fail the whole process due to cleanup issues

    async def _create_graph(self, repo_name: str, modules_data: List[Dict]):
        """Create all nodes and relationships in Neo4j"""

        async with self.driver.session() as session:
            # Create Repository node
            await session.run(
                "CREATE (r:Repository {name: $repo_name, created_at: datetime()})",
                repo_name=repo_name,
            )

            nodes_created = 0
            relationships_created = 0

            for i, mod in enumerate(modules_data):
                # Skip if mod is not a dictionary (safety check)
                if not isinstance(mod, dict):
                    logger.warning(
                        f"Skipping invalid module data at index {i}: {type(mod)}"
                    )
                    continue

                # Ensure required fields exist
                if not all(
                    key in mod
                    for key in [
                        "file_path",
                        "module_name",
                        "classes",
                        "functions",
                        "imports",
                    ]
                ):
                    logger.warning(
                        f"Skipping module with missing required fields: {mod.get('file_path', 'unknown')}"
                    )
                    continue

                # 1. Create File node
                await session.run(
                    """
                    CREATE (f:File {
                        name: $name,
                        path: $path,
                        module_name: $module_name,
                        line_count: $line_count,
                        language: $language,
                        created_at: datetime()
                    })
                """,
                    name=mod["file_path"].split("/")[-1],
                    path=mod["file_path"],
                    module_name=mod["module_name"],
                    line_count=mod.get("line_count", 0),
                    language=mod.get("language", "unknown"),
                )
                nodes_created += 1

                # 2. Connect File to Repository
                await session.run(
                    """
                    MATCH (r:Repository {name: $repo_name})
                    MATCH (f:File {path: $file_path})
                    CREATE (r)-[:CONTAINS]->(f)
                """,
                    repo_name=repo_name,
                    file_path=mod["file_path"],
                )
                relationships_created += 1

                # 3. Create Class nodes and relationships
                for cls in mod.get("classes", []):
                    if (
                        not isinstance(cls, dict)
                        or "name" not in cls
                        or "full_name" not in cls
                    ):
                        logger.warning(f"Skipping invalid class data: {cls}")
                        continue

                    # Create Class node using MERGE to avoid duplicates
                    await session.run(
                        """
                        MERGE (c:Class {full_name: $full_name})
                        ON CREATE SET c.name = $name, c.created_at = datetime()
                    """,
                        name=cls["name"],
                        full_name=cls["full_name"],
                    )
                    nodes_created += 1

                    # Connect File to Class
                    await session.run(
                        """
                        MATCH (f:File {path: $file_path})
                        MATCH (c:Class {full_name: $class_full_name})
                        MERGE (f)-[:DEFINES]->(c)
                    """,
                        file_path=mod["file_path"],
                        class_full_name=cls["full_name"],
                    )
                    relationships_created += 1

                    # 4. Create Method nodes - use MERGE to avoid duplicates
                    for method in cls.get("methods", []):
                        if not isinstance(method, dict) or "name" not in method:
                            logger.warning(f"Skipping invalid method data: {method}")
                            continue

                        method_full_name = f"{cls['full_name']}.{method['name']}"
                        # Create method with unique ID to avoid conflicts
                        method_id = f"{cls['full_name']}::{method['name']}"

                        await session.run(
                            """
                            MERGE (m:Method {method_id: $method_id})
                            ON CREATE SET m.name = $name, 
                                         m.full_name = $full_name,
                                         m.args = $args,
                                         m.params_list = $params_list,
                                         m.params_detailed = $params_detailed,
                                         m.return_type = $return_type,
                                         m.created_at = datetime()
                        """,
                            name=method["name"],
                            full_name=method_full_name,
                            method_id=method_id,
                            args=method.get("args", []),
                            params_list=[
                                f"{p.get('name', '')}:{p.get('type', '')}"
                                for p in method.get("params", [])
                                if isinstance(p, dict)
                            ],  # Simple format
                            params_detailed=method.get(
                                "params_detailed", []
                            ),  # Detailed format
                            return_type=method.get("return_type", "Any"),
                        )
                        nodes_created += 1

                        # Connect Class to Method
                        await session.run(
                            """
                            MATCH (c:Class {full_name: $class_full_name})
                            MATCH (m:Method {method_id: $method_id})
                            MERGE (c)-[:HAS_METHOD]->(m)
                        """,
                            class_full_name=cls["full_name"],
                            method_id=method_id,
                        )
                        relationships_created += 1

                    # 5. Create Attribute nodes - use MERGE to avoid duplicates
                    for attr in cls.get("attributes", []):
                        if not isinstance(attr, dict) or "name" not in attr:
                            logger.warning(f"Skipping invalid attribute data: {attr}")
                            continue

                        attr_full_name = f"{cls['full_name']}.{attr['name']}"
                        # Create attribute with unique ID to avoid conflicts
                        attr_id = f"{cls['full_name']}::{attr['name']}"
                        await session.run(
                            """
                            MERGE (a:Attribute {attr_id: $attr_id})
                            ON CREATE SET a.name = $name,
                                         a.full_name = $full_name,
                                         a.type = $type,
                                         a.created_at = datetime()
                        """,
                            name=attr["name"],
                            full_name=attr_full_name,
                            attr_id=attr_id,
                            type=attr.get("type", "Any"),
                        )
                        nodes_created += 1

                        # Connect Class to Attribute
                        await session.run(
                            """
                            MATCH (c:Class {full_name: $class_full_name})
                            MATCH (a:Attribute {attr_id: $attr_id})
                            MERGE (c)-[:HAS_ATTRIBUTE]->(a)
                        """,
                            class_full_name=cls["full_name"],
                            attr_id=attr_id,
                        )
                        relationships_created += 1

                # 6. Create Function nodes (top-level) - use MERGE to avoid duplicates
                for func in mod.get("functions", []):
                    if not isinstance(func, dict) or "name" not in func:
                        logger.warning(f"Skipping invalid function data: {func}")
                        continue

                    func_id = f"{mod['file_path']}::{func['name']}"
                    await session.run(
                        """
                        MERGE (f:Function {func_id: $func_id})
                        ON CREATE SET f.name = $name,
                                     f.full_name = $full_name,
                                     f.args = $args,
                                     f.params_list = $params_list,
                                     f.params_detailed = $params_detailed,
                                     f.return_type = $return_type,
                                     f.created_at = datetime()
                    """,
                        name=func["name"],
                        full_name=func.get("full_name", func["name"]),
                        func_id=func_id,
                        args=func.get("args", []),
                        params_list=func.get(
                            "params_list", []
                        ),  # Simple format for backwards compatibility
                        params_detailed=func.get(
                            "params_detailed", []
                        ),  # Detailed format
                        return_type=func.get("return_type", "Any"),
                    )
                    nodes_created += 1

                    # Connect File to Function
                    await session.run(
                        """
                        MATCH (file:File {path: $file_path})
                        MATCH (func:Function {func_id: $func_id})
                        MERGE (file)-[:DEFINES]->(func)
                    """,
                        file_path=mod["file_path"],
                        func_id=func_id,
                    )
                    relationships_created += 1

                # 7. Create Import relationships
                imports_list = mod.get("imports", [])
                if isinstance(imports_list, list):
                    for import_name in imports_list:
                        if isinstance(import_name, str):
                            # Try to find the target file
                            await session.run(
                                """
                                MATCH (source:File {path: $source_path})
                                OPTIONAL MATCH (target:File) 
                                WHERE target.module_name = $import_name OR target.module_name STARTS WITH $import_name
                                WITH source, target
                                WHERE target IS NOT NULL
                                MERGE (source)-[:IMPORTS]->(target)
                            """,
                                source_path=mod["file_path"],
                                import_name=import_name,
                            )
                            relationships_created += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(modules_data)} files...")

            logger.info(
                f"Created {nodes_created} nodes and {relationships_created} relationships"
            )

    async def search_graph(self, query_type: str, **kwargs):
        """Search the Neo4j graph directly"""
        async with self.driver.session() as session:
            if query_type == "files_importing":
                target = kwargs.get("target")
                result = await session.run(
                    """
                    MATCH (source:File)-[:IMPORTS]->(target:File)
                    WHERE target.module_name CONTAINS $target
                    RETURN source.path as file, target.module_name as imports
                """,
                    target=target,
                )
                return [
                    {"file": record["file"], "imports": record["imports"]}
                    async for record in result
                ]

            elif query_type == "classes_in_file":
                file_path = kwargs.get("file_path")
                result = await session.run(
                    """
                    MATCH (f:File {path: $file_path})-[:DEFINES]->(c:Class)
                    RETURN c.name as class_name, c.full_name as full_name
                """,
                    file_path=file_path,
                )
                return [
                    {
                        "class_name": record["class_name"],
                        "full_name": record["full_name"],
                    }
                    async for record in result
                ]

            elif query_type == "methods_of_class":
                class_name = kwargs.get("class_name")
                result = await session.run(
                    """
                    MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
                    WHERE c.name CONTAINS $class_name OR c.full_name CONTAINS $class_name
                    RETURN m.name as method_name, m.args as args
                """,
                    class_name=class_name,
                )
                return [
                    {"method_name": record["method_name"], "args": record["args"]}
                    async for record in result
                ]


async def main():
    """Example usage"""
    load_dotenv()

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

    extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)

    try:
        await extractor.initialize()

        # Analyze repository - direct Neo4j, no LLM processing!
        # repo_url = "https://github.com/pydantic/pydantic-ai.git"
        repo_url = "https://github.com/getzep/graphiti.git"
        await extractor.analyze_repository(repo_url)

        # Direct graph queries
        print("\\n=== Direct Neo4j Queries ===")

        # Which files import from models?
        results = await extractor.search_graph("files_importing", target="models")
        print(f"\\nFiles importing from 'models': {len(results)}")
        for result in results[:3]:
            print(f"- {result['file']} imports {result['imports']}")

        # What classes are in a specific file?
        results = await extractor.search_graph(
            "classes_in_file", file_path="pydantic_ai/models/openai.py"
        )
        print(f"\\nClasses in openai.py: {len(results)}")
        for result in results:
            print(f"- {result['class_name']}")

        # What methods does OpenAIModel have?
        results = await extractor.search_graph(
            "methods_of_class", class_name="OpenAIModel"
        )
        print(f"\\nMethods of OpenAIModel: {len(results)}")
        for result in results[:5]:
            print(f"- {result['method_name']}({', '.join(result['args'])})")

    finally:
        await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())
