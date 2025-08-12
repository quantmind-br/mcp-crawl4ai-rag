#!/usr/bin/env python3
"""
Tree-sitter Grammar Build Script

Clones and compiles Tree-sitter language grammars for multi-language code analysis.
Creates a shared library that can be used by the Tree-sitter Python bindings.

Usage:
    python scripts/build_grammars.py

This script must be run once before using Tree-sitter parsing functionality.
"""

import subprocess
import shutil
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GrammarBuilder:
    """Builds Tree-sitter language grammars into a shared library."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Default to script directory
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)

        # Point to the new grammars location in src/k_graph/parsing/grammars
        project_root = self.base_dir.parent
        self.grammars_dir = project_root / "src" / "k_graph" / "parsing" / "grammars"
        self.build_dir = project_root / "build"

        # Grammar repositories to clone
        self.grammar_repos = {
            "python": "https://github.com/tree-sitter/tree-sitter-python",
            "javascript": "https://github.com/tree-sitter/tree-sitter-javascript",
            "typescript": "https://github.com/tree-sitter/tree-sitter-typescript",
            "java": "https://github.com/tree-sitter/tree-sitter-java",
            "go": "https://github.com/tree-sitter/tree-sitter-go",
            "rust": "https://github.com/tree-sitter/tree-sitter-rust",
            "c": "https://github.com/tree-sitter/tree-sitter-c",
            "cpp": "https://github.com/tree-sitter/tree-sitter-cpp",
            "c_sharp": "https://github.com/tree-sitter/tree-sitter-c-sharp",
            "php": "https://github.com/tree-sitter/tree-sitter-php",
            "ruby": "https://github.com/tree-sitter/tree-sitter-ruby",
            "kotlin": "https://github.com/tree-sitter/tree-sitter-kotlin",
        }

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.grammars_dir.mkdir(exist_ok=True)
        self.build_dir.mkdir(exist_ok=True)
        logger.info(f"Created directories: {self.grammars_dir}, {self.build_dir}")

    def check_git_available(self) -> bool:
        """Check if git is available on the system."""
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Git is not available. Please install git to clone grammars.")
            return False

    def clone_grammar(self, name: str, url: str) -> bool:
        """Clone a single grammar repository."""
        repo_path = self.grammars_dir / f"tree-sitter-{name}"

        if repo_path.exists():
            logger.info(f"Grammar {name} already exists at {repo_path}")
            return True

        try:
            logger.info(f"Cloning {name} grammar from {url}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(repo_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Successfully cloned {name} grammar")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {name} grammar: {e.stderr}")
            return False

    def clone_all_grammars(self) -> bool:
        """Clone all required grammar repositories."""
        logger.info("Starting grammar cloning process...")
        success = True

        for name, url in self.grammar_repos.items():
            if not self.clone_grammar(name, url):
                success = False

        if success:
            logger.info("All grammars cloned successfully")
        else:
            logger.error("Some grammars failed to clone")

        return success

    def get_grammar_paths(self) -> List[str]:
        """Get list of grammar paths for compilation."""
        grammar_paths = []

        # Regular grammars
        for name in [
            "python",
            "javascript",
            "java",
            "go",
            "rust",
            "c",
            "cpp",
            "c_sharp",
            "php",
            "ruby",
            "kotlin",
        ]:
            grammar_path = self.grammars_dir / f"tree-sitter-{name}"
            if grammar_path.exists():
                grammar_paths.append(str(grammar_path))
            else:
                logger.warning(f"Grammar path not found: {grammar_path}")

        # TypeScript special case - has typescript and tsx subdirectories
        typescript_base = self.grammars_dir / "tree-sitter-typescript"
        if typescript_base.exists():
            for subdir in ["typescript", "tsx"]:
                typescript_subdir = typescript_base / subdir
                if typescript_subdir.exists():
                    grammar_paths.append(str(typescript_subdir))
                else:
                    logger.warning(
                        f"TypeScript subdirectory not found: {typescript_subdir}"
                    )
        else:
            logger.warning(f"TypeScript grammar not found: {typescript_base}")

        return grammar_paths

    def verify_language_packages(self) -> bool:
        """Verify that language packages are installed and working."""
        logger.info("Verifying installed language packages...")

        languages_to_test = {
            "python": ("tree_sitter_python", "language"),
            "javascript": ("tree_sitter_javascript", "language"),
            "typescript": ("tree_sitter_typescript", "language_typescript"),
            "java": ("tree_sitter_java", "language"),
            "go": ("tree_sitter_go", "language"),
            "rust": ("tree_sitter_rust", "language"),
            "c": ("tree_sitter_c", "language"),
            "cpp": ("tree_sitter_cpp", "language"),
            "c_sharp": ("tree_sitter_c_sharp", "language"),
            "php": ("tree_sitter_php", "language_php"),
            "ruby": ("tree_sitter_ruby", "language"),
            "kotlin": ("tree_sitter_kotlin", "language"),
        }

        all_working = True

        for lang_name, (module_name, func_name) in languages_to_test.items():
            try:
                logger.info(f"Testing {lang_name} language package...")

                # Import the language module
                from tree_sitter import Language, Parser

                module = __import__(module_name)
                language_func = getattr(module, func_name)
                language_capsule = language_func()

                # Create Language object from PyCapsule
                language = Language(language_capsule)

                # Basic test - create a parser
                parser = Parser()
                parser.language = language

                # Test parsing simple code
                test_code = self._get_test_code(lang_name)
                tree = parser.parse(test_code)

                if tree.root_node.child_count > 0:
                    logger.info(f"✓ {lang_name} language package working correctly")
                else:
                    logger.warning(f"⚠ {lang_name} parsed but produced empty tree")

            except ImportError as e:
                logger.error(f"✗ {lang_name} language package not found: {e}")
                all_working = False
            except Exception as e:
                logger.error(f"✗ {lang_name} language package failed: {e}")
                all_working = False

        return all_working

    def _get_test_code(self, language: str) -> bytes:
        """Get simple test code for each language."""
        test_codes = {
            "python": b"def hello(): pass",
            "javascript": b"function hello() {}",
            "typescript": b"function hello(): void {}",
            "java": b"class Hello { public void main() {} }",
            "go": b"package main\nfunc hello() {}",
            "rust": b"fn hello() {}",
            "c": b"#include <stdio.h>\nint main() { return 0; }",
            "cpp": b"#include <iostream>\nint main() { return 0; }",
            "c_sharp": b"class Hello { static void Main() {} }",
            "php": b"<?php function hello() {} ?>",
            "ruby": b"def hello\nend",
            "kotlin": b"fun hello() {}",
        }
        return test_codes.get(language, b"")

    def build_shared_library(self) -> bool:
        """Build verification - now just verifies language packages work."""
        logger.info("Modern tree-sitter uses pre-built language packages")
        logger.info("Verifying language packages instead of building from source...")

        return self.verify_language_packages()

    def _verify_library(self, library_path: str):
        """Verify that the built library can load languages."""
        try:
            from tree_sitter import Language

            # Test loading each expected language
            test_languages = ["python", "javascript", "typescript", "tsx", "java"]

            for lang_name in test_languages:
                try:
                    Language(library_path, lang_name)
                    logger.info(f"✓ Language '{lang_name}' loads successfully")
                except Exception as e:
                    logger.warning(f"✗ Language '{lang_name}' failed to load: {e}")

        except Exception as e:
            logger.error(f"Library verification failed: {e}")

    def clean_grammars(self):
        """Clean up cloned grammar repositories."""
        if self.grammars_dir.exists():
            logger.info(f"Cleaning grammars directory: {self.grammars_dir}")
            shutil.rmtree(self.grammars_dir)
            logger.info("Grammars directory cleaned")

    def build(self, clean_first: bool = False) -> bool:
        """
        Main verification process for modern tree-sitter.

        Args:
            clean_first: Whether to clean existing grammars before building

        Returns:
            True if verification was successful, False otherwise
        """
        logger.info("Starting Tree-sitter language verification...")

        try:
            # Verify language packages are installed and working
            if not self.verify_language_packages():
                logger.error("Language package verification failed")
                logger.error("Make sure all required packages are installed:")
                logger.error(
                    "  uv add tree-sitter-python tree-sitter-javascript tree-sitter-typescript tree-sitter-java tree-sitter-go tree-sitter-rust tree-sitter-c tree-sitter-cpp tree-sitter-c-sharp tree-sitter-php tree-sitter-ruby tree-sitter-kotlin"
                )
                return False

            logger.info("Grammar verification completed successfully!")
            logger.info("All language packages are working correctly.")

            return True

        except KeyboardInterrupt:
            logger.info("Verification process interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during verification: {e}")
            return False


def build_grammars_if_needed() -> bool:
    """
    Build Tree-sitter grammars if they are needed.

    This function can be called programmatically to ensure grammars are available.
    It's designed to be safe to call multiple times.

    Returns:
        True if grammars are available or were successfully built
    """
    try:
        builder = GrammarBuilder()

        # For modern tree-sitter, we mainly verify language packages are available
        logger.info("Verifying Tree-sitter language packages...")

        if builder.verify_language_packages():
            logger.info("All Tree-sitter language packages are available")
            return True
        else:
            logger.warning("Some Tree-sitter language packages are not available")
            logger.warning(
                "Install with: uv add tree-sitter-python tree-sitter-javascript tree-sitter-typescript tree-sitter-java tree-sitter-go"
            )
            return False

    except Exception as e:
        logger.error(f"Error during grammar verification: {e}")
        return False


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Tree-sitter language grammars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_grammars.py                 # Build grammars
    python build_grammars.py --clean         # Clean and rebuild
    python build_grammars.py --base-dir /path/to/dir  # Use custom directory
        """,
    )

    parser.add_argument(
        "--clean", action="store_true", help="Clean existing grammars before building"
    )

    parser.add_argument(
        "--base-dir", type=str, help="Base directory for grammars and build output"
    )

    args = parser.parse_args()

    # Create builder and run
    builder = GrammarBuilder(base_dir=args.base_dir)
    success = builder.build(clean_first=args.clean)

    if success:
        print("\nTree-sitter grammars built successfully!")
        print("You can now use multi-language parsing in the code analysis tools.")
    else:
        print("\nFailed to build Tree-sitter grammars.")
        print("Check the logs above for error details.")
        exit(1)


if __name__ == "__main__":
    main()
