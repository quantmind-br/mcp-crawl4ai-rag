"""
Tests for Tree-sitter grammar initialization.
"""

import pytest
import unittest.mock as mock

from src.utils.grammar_initialization import (
    check_essential_grammars,
    check_grammars_directory,
    initialize_grammars_if_needed,
    get_grammars_directory,
)


class TestGrammarInitialization:
    """Test grammar initialization functions."""

    def test_get_grammars_directory(self):
        """Test that grammars directory path is correctly calculated."""
        grammars_dir = get_grammars_directory()
        assert grammars_dir.name == "grammars"
        assert grammars_dir.parent.name == "knowledge_graphs"

    def test_check_essential_grammars_available(self):
        """Test checking essential grammars when all are available."""
        with mock.patch("builtins.__import__") as mock_import:
            # Mock successful imports
            mock_module = mock.Mock()
            mock_module.language.return_value = mock.Mock()  # Mock language capsule
            mock_import.return_value = mock_module

            availability = check_essential_grammars()

            assert len(availability) == 5  # python, javascript, typescript, java, go
            assert all(availability.values())  # All should be True

    def test_check_essential_grammars_missing(self):
        """Test checking essential grammars when some are missing."""
        def mock_import_side_effect(module_name):
            if module_name == "tree_sitter_python":
                raise ImportError("Module not found")
            mock_module = mock.Mock()
            mock_module.language.return_value = mock.Mock()
            mock_module.language_typescript.return_value = mock.Mock()
            return mock_module

        with mock.patch("builtins.__import__", side_effect=mock_import_side_effect):
            availability = check_essential_grammars()

            assert availability["python"] is False
            assert availability["javascript"] is True
            assert availability["typescript"] is True

    def test_check_grammars_directory_missing(self):
        """Test checking grammars directory when it doesn't exist."""
        with mock.patch("src.utils.grammar_initialization.get_grammars_directory") as mock_get_dir:
            mock_path = mock.Mock()
            mock_path.exists.return_value = False
            mock_get_dir.return_value = mock_path

            result = check_grammars_directory()
            assert result is False

    def test_check_grammars_directory_incomplete(self):
        """Test checking grammars directory when it exists but is incomplete."""
        with mock.patch("src.utils.grammar_initialization.get_grammars_directory") as mock_get_dir:
            mock_path = mock.Mock()
            mock_path.exists.return_value = True

            # Mock grammar subdirectories
            def mock_truediv(self, grammar_name):
                mock_subpath = mock.Mock()
                # tree-sitter-python is missing, others exist
                mock_subpath.exists.return_value = grammar_name != "tree-sitter-python"
                return mock_subpath

            mock_path.__truediv__ = mock_truediv
            mock_get_dir.return_value = mock_path

            result = check_grammars_directory()
            assert result is False

    def test_initialize_grammars_already_available(self):
        """Test initialization when grammars are already available."""
        with mock.patch("src.utils.grammar_initialization.check_essential_grammars") as mock_check:
            mock_check.return_value = {
                "python": True,
                "javascript": True,
                "typescript": True,
                "java": True,
                "go": True,
            }

            result = initialize_grammars_if_needed()
            assert result is True

    def test_initialize_grammars_need_building(self):
        """Test initialization when grammars need to be built."""
        with mock.patch("src.utils.grammar_initialization.check_essential_grammars") as mock_check, \
             mock.patch("src.utils.grammar_initialization.check_grammars_directory") as mock_check_dir, \
             mock.patch("src.utils.grammar_initialization.run_grammar_builder") as mock_run_builder, \
             mock.patch("subprocess.run") as mock_subprocess:

            # Mock that grammars are not available
            mock_check.return_value = {
                "python": False,
                "javascript": False,
                "typescript": False,
                "java": False,
                "go": False,
            }

            # Mock that directory doesn't exist
            mock_check_dir.return_value = False

            # Mock that git is available
            mock_subprocess.return_value = mock.Mock(returncode=0)

            # Mock successful grammar building
            mock_run_builder.return_value = True

            result = initialize_grammars_if_needed()
            assert result is True
            mock_run_builder.assert_called_once()

    def test_initialize_grammars_git_unavailable(self):
        """Test initialization when git is not available."""
        with mock.patch("src.utils.grammar_initialization.check_essential_grammars") as mock_check, \
             mock.patch("src.utils.grammar_initialization.check_grammars_directory") as mock_check_dir, \
             mock.patch("subprocess.run") as mock_subprocess:

            # Mock that grammars are not available
            mock_check.return_value = {
                "python": False,
                "javascript": False,
                "typescript": False,
                "java": False,
                "go": False,
            }

            # Mock that directory doesn't exist
            mock_check_dir.return_value = False

            # Mock that git is not available
            mock_subprocess.side_effect = FileNotFoundError()

            result = initialize_grammars_if_needed()
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__])