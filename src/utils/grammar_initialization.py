"""
Grammar initialization utilities for Tree-sitter multi-language parsing.

This module provides functions to automatically verify and build Tree-sitter
grammars when they are missing, ensuring smooth application startup.
"""

import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def get_grammars_directory() -> Path:
    """Get the path to the Tree-sitter grammars directory."""
    # Get project root (assuming we're in src/utils/)
    project_root = Path(__file__).parent.parent.parent
    return project_root / "knowledge_graphs" / "grammars"


def check_essential_grammars() -> Dict[str, bool]:
    """
    Check if essential Tree-sitter language packages are available.

    Returns:
        Dict mapping language names to availability status
    """
    essential_languages = {
        "python": ("tree_sitter_python", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "java": ("tree_sitter_java", "language"),
        "go": ("tree_sitter_go", "language"),
    }

    availability = {}

    for lang_name, (module_name, func_name) in essential_languages.items():
        try:
            # Try to import the language module and access the language function
            module = __import__(module_name)
            language_func = getattr(module, func_name)

            # Test that we can actually get a language capsule
            language_capsule = language_func()
            availability[lang_name] = True
            logger.debug(f"✓ {lang_name} Tree-sitter language available")

        except (ImportError, AttributeError, Exception) as e:
            availability[lang_name] = False
            logger.debug(f"✗ {lang_name} Tree-sitter language not available: {e}")

    return availability


def check_grammars_directory() -> bool:
    """
    Check if the grammars directory exists and contains expected grammar repositories.

    Returns:
        True if grammars directory appears to be properly set up
    """
    grammars_dir = get_grammars_directory()

    if not grammars_dir.exists():
        logger.debug(f"Grammars directory does not exist: {grammars_dir}")
        return False

    # Check for essential grammar repositories
    essential_grammars = [
        "tree-sitter-python",
        "tree-sitter-javascript",
        "tree-sitter-java",
    ]
    missing_grammars = []

    for grammar in essential_grammars:
        grammar_path = grammars_dir / grammar
        if not grammar_path.exists():
            missing_grammars.append(grammar)

    if missing_grammars:
        logger.debug(f"Missing grammar directories: {missing_grammars}")
        return False

    logger.debug("Grammars directory appears to be properly set up")
    return True


def run_grammar_builder() -> bool:
    """
    Run the grammar builder to verify Tree-sitter language packages.

    Returns:
        True if grammar verification completed successfully
    """
    try:
        # Import and use the build_grammars_if_needed function directly
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / "knowledge_graphs"))

        from build_grammars import build_grammars_if_needed

        return build_grammars_if_needed()

    except ImportError as e:
        logger.error(f"Could not import grammar builder: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running grammar builder: {e}")
        return False
    finally:
        # Clean up sys.path
        if str(project_root / "knowledge_graphs") in sys.path:
            sys.path.remove(str(project_root / "knowledge_graphs"))


def initialize_grammars_if_needed() -> bool:
    """
    Initialize Tree-sitter grammars if they are missing or incomplete.

    This function checks if Tree-sitter language packages are available and
    if the grammars directory is properly set up. If not, it attempts to
    build the grammars automatically.

    Returns:
        True if grammars are available or were successfully initialized
    """
    logger.info("Checking Tree-sitter grammar availability...")

    # First check if language packages are already available
    availability = check_essential_grammars()
    available_count = sum(availability.values())
    total_count = len(availability)

    if available_count == total_count:
        logger.info(f"✓ All {total_count} essential Tree-sitter languages available")
        return True

    logger.info(f"Only {available_count}/{total_count} Tree-sitter languages available")

    # Check if we need to build grammars
    if not check_grammars_directory():
        logger.info("Tree-sitter grammars directory missing or incomplete")

        # Check if we can build grammars (requires git)
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(
                "Git not available - cannot auto-build Tree-sitter grammars. "
                "Some knowledge graph features may be limited."
            )
            return False

        # Attempt to build grammars
        if run_grammar_builder():
            logger.info("✓ Tree-sitter grammars initialized successfully")
            return True
        else:
            logger.warning(
                "Failed to build Tree-sitter grammars automatically. "
                "Some knowledge graph features may be limited. "
                "Try running manually: python knowledge_graphs/build_grammars.py"
            )
            return False

    else:
        # Grammars directory exists but packages aren't available
        # This might happen if packages need to be rebuilt
        logger.info("Grammars directory exists but language packages unavailable")
        logger.info("This may require manual intervention or package rebuilding")
        return False
