from src.utils.grammar_initialization import (
    initialize_grammars_if_needed,
    check_essential_grammars,
    check_grammars_directory,
)
from unittest.mock import patch


class TestGrammarInitialization:
    def test_check_essential_grammars_returns_dict(self):
        """Test that check_essential_grammars returns a dictionary"""
        result = check_essential_grammars()
        assert isinstance(result, dict)

    def test_check_grammars_directory_returns_bool(self):
        """Test that check_grammars_directory returns a boolean"""
        result = check_grammars_directory()
        assert isinstance(result, bool)

    @patch("src.utils.grammar_initialization.check_essential_grammars")
    def test_initialize_grammars_if_needed_all_available(self, mock_check_essential):
        """Test initialize_grammars_if_needed when all grammars are available"""
        mock_check_essential.return_value = {
            "python": True,
            "javascript": True,
            "java": True,
        }
        result = initialize_grammars_if_needed()
        assert result is True

    @patch("src.utils.grammar_initialization.check_grammars_directory")
    @patch("src.utils.grammar_initialization.check_essential_grammars")
    def test_initialize_grammars_if_needed_needs_building(
        self, mock_check_essential, mock_check_directory
    ):
        """Test initialize_grammars_if_needed when grammars need to be built"""
        mock_check_essential.return_value = {
            "python": False,
            "javascript": False,
            "java": False,
        }
        mock_check_directory.return_value = True
        result = initialize_grammars_if_needed()
        # Should return False when directory exists but packages aren't available
        assert result is False
