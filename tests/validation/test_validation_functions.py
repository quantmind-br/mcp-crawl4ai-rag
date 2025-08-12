"""
Testes para as funcionalidades de validação em validation.py.

Este módulo contém testes para as funções de validação, incluindo:
- Validação de URLs do GitHub
- Normalização de URLs
- Tratamento de casos extremos
- Validação de formatos de URL
"""

import pytest

# Importa as funções de validação
try:
    from src.utils.validation import validate_github_url, normalize_github_url
except ImportError:
    from utils.validation import validate_github_url, normalize_github_url


class TestGitHubURLValidation:
    """Testes para validação de URLs do GitHub."""

    def test_validate_github_url_valid_https(self):
        """Testa validação de URL HTTPS válida do GitHub."""
        url = "https://github.com/user/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_http(self):
        """Testa validação de URL HTTP válida do GitHub."""
        url = "http://github.com/user/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_with_www(self):
        """Testa validação de URL com www válida do GitHub."""
        url = "https://www.github.com/user/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_with_dot_git(self):
        """Testa validação de URL com .git válida do GitHub."""
        url = "https://github.com/user/repo.git"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_with_subpath(self):
        """Testa validação de URL com subcaminho válida do GitHub."""
        url = "https://github.com/user/repo/tree/main/docs"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_with_blob(self):
        """Testa validação de URL com blob válida do GitHub."""
        url = "https://github.com/user/repo/blob/main/README.md"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_with_issues(self):
        """Testa validação de URL com issues válida do GitHub."""
        url = "https://github.com/user/repo/issues/123"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_with_pull(self):
        """Testa validação de URL com pull request válida do GitHub."""
        url = "https://github.com/user/repo/pull/456"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_with_releases(self):
        """Testa validação de URL com releases válida do GitHub."""
        url = "https://github.com/user/repo/releases/tag/v1.0.0"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_with_wiki(self):
        """Testa validação de URL com wiki válida do GitHub."""
        url = "https://github.com/user/repo/wiki/Home"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_complex_username(self):
        """Testa validação de URL com nome de usuário complexo."""
        url = "https://github.com/user-name123/repo-name"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_valid_complex_repo_name(self):
        """Testa validação de URL com nome de repositório complexo."""
        url = "https://github.com/user/repo-name-123"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_empty_string(self):
        """Testa validação de string vazia."""
        url = ""
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_github_url_none(self):
        """Testa validação de None."""
        url = None
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "string" in error.lower()

    def test_validate_github_url_not_string(self):
        """Testa validação de tipo não string."""
        url = 123
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "string" in error.lower()

    def test_validate_github_url_invalid_scheme(self):
        """Testa validação de URL com esquema inválido."""
        url = "ftp://github.com/user/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "http" in error.lower()

    def test_validate_github_url_not_github_domain(self):
        """Testa validação de URL que não é do GitHub."""
        url = "https://gitlab.com/user/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "github.com" in error.lower()

    def test_validate_github_url_missing_path(self):
        """Testa validação de URL sem caminho."""
        url = "https://github.com"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "path" in error.lower()

    def test_validate_github_url_missing_owner(self):
        """Testa validação de URL sem proprietário."""
        url = "https://github.com/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "owner" in error.lower()

    def test_validate_github_url_missing_repo(self):
        """Testa validação de URL sem repositório."""
        url = "https://github.com/user/"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "repository" in error.lower()

    def test_validate_github_url_invalid_owner_format(self):
        """Testa validação de URL com formato de proprietário inválido."""
        url = "https://github.com/-user/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "owner" in error.lower()

    def test_validate_github_url_invalid_repo_format(self):
        """Testa validação de URL com formato de repositório inválido."""
        url = "https://github.com/user/-repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "repository" in error.lower()

    def test_validate_github_url_invalid_path_patterns(self):
        """Testa validação de URL com padrões de caminho inválidos."""
        url = "https://github.com/user/repo/../other"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "invalid path" in error.lower()

    def test_validate_github_url_invalid_subpath(self):
        """Testa validação de URL com subcaminho inválido."""
        url = "https://github.com/user/repo/invalid/path"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "unsupported" in error.lower()

    def test_validate_github_url_malformed_url(self):
        """Testa validação de URL malformada."""
        url = "not-a-url"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "scheme" in error.lower()  # A mensagem real menciona "scheme"


class TestGitHubURLNormalization:
    """Testes para normalização de URLs do GitHub."""

    def test_normalize_github_url_valid_https(self):
        """Testa normalização de URL HTTPS válida."""
        url = "https://github.com/user/repo"
        normalized = normalize_github_url(url)

        assert normalized == "https://github.com/user/repo.git"

    def test_normalize_github_url_valid_http(self):
        """Testa normalização de URL HTTP válida."""
        url = "http://github.com/user/repo"
        normalized = normalize_github_url(url)

        assert normalized == "https://github.com/user/repo.git"

    def test_normalize_github_url_with_www(self):
        """Testa normalização de URL com www."""
        url = "https://www.github.com/user/repo"
        normalized = normalize_github_url(url)

        assert normalized == "https://github.com/user/repo.git"

    def test_normalize_github_url_with_dot_git(self):
        """Testa normalização de URL que já tem .git."""
        url = "https://github.com/user/repo.git"
        normalized = normalize_github_url(url)

        assert normalized == "https://github.com/user/repo.git"

    def test_normalize_github_url_with_subpath(self):
        """Testa normalização de URL com subcaminho."""
        url = "https://github.com/user/repo/tree/main/docs"
        normalized = normalize_github_url(url)

        assert normalized == "https://github.com/user/repo.git"

    def test_normalize_github_url_complex_username(self):
        """Testa normalização de URL com nome de usuário complexo."""
        url = "https://github.com/user-name-123/repo-name"
        normalized = normalize_github_url(url)

        assert normalized == "https://github.com/user-name-123/repo-name.git"

    def test_normalize_github_url_invalid_url(self):
        """Testa normalização de URL inválida."""
        url = "https://gitlab.com/user/repo"

        with pytest.raises(ValueError) as exc_info:
            normalize_github_url(url)

        assert "Invalid GitHub URL" in str(exc_info.value)

    def test_normalize_github_url_empty_url(self):
        """Testa normalização de URL vazia."""
        url = ""

        with pytest.raises(ValueError) as exc_info:
            normalize_github_url(url)

        assert "Invalid GitHub URL" in str(exc_info.value)


class TestGitHubURLValidationEdgeCases:
    """Testes para casos extremos de validação de URLs do GitHub."""

    def test_validate_github_url_very_long_username(self):
        """Testa validação de URL com nome de usuário muito longo."""
        long_username = "a" * 39  # Máximo permitido pelo GitHub
        url = f"https://github.com/{long_username}/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_very_long_repo_name(self):
        """Testa validação de URL com nome de repositório muito longo."""
        long_repo = "a" * 100  # Nome muito longo
        url = f"https://github.com/user/{long_repo}"
        is_valid, error = validate_github_url(url)

        assert is_valid is True  # A validação não verifica comprimento máximo
        assert error == ""

    def test_validate_github_url_special_characters_in_username(self):
        """Testa validação de URL com caracteres especiais no nome de usuário."""
        url = "https://github.com/user-123/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_special_characters_in_repo(self):
        """Testa validação de URL com caracteres especiais no nome do repositório."""
        url = "https://github.com/user/repo-123"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_leading_hyphen_username(self):
        """Testa validação de URL com hífen no início do nome de usuário."""
        url = "https://github.com/-user/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "owner" in error.lower()

    def test_validate_github_url_trailing_hyphen_username(self):
        """Testa validação de URL com hífen no final do nome de usuário."""
        url = "https://github.com/user-/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "owner" in error.lower()

    def test_validate_github_url_leading_hyphen_repo(self):
        """Testa validação de URL com hífen no início do nome do repositório."""
        url = "https://github.com/user/-repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "repository" in error.lower()

    def test_validate_github_url_trailing_hyphen_repo(self):
        """Testa validação de URL com hífen no final do nome do repositório."""
        url = "https://github.com/user/repo-"
        is_valid, error = validate_github_url(url)

        assert is_valid is False
        assert "repository" in error.lower()

    def test_validate_github_url_consecutive_hyphens(self):
        """Testa validação de URL com hífens consecutivos."""
        url = "https://github.com/user--name/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True  # Hífens consecutivos são permitidos
        assert error == ""

    def test_validate_github_url_uppercase_username(self):
        """Testa validação de URL com nome de usuário em maiúsculas."""
        url = "https://github.com/USER/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_uppercase_repo(self):
        """Testa validação de URL com nome de repositório em maiúsculas."""
        url = "https://github.com/user/REPO"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_numbers_only_username(self):
        """Testa validação de URL com nome de usuário apenas números."""
        url = "https://github.com/123/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

    def test_validate_github_url_numbers_only_repo(self):
        """Testa validação de URL com nome de repositório apenas números."""
        url = "https://github.com/user/123"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""


class TestGitHubURLValidationIntegration:
    """Testes de integração para validação de URLs do GitHub."""

    def test_validate_and_normalize_workflow(self):
        """Testa workflow completo de validação e normalização."""
        # Testa URL válida
        url = "https://github.com/user/repo"
        is_valid, error = validate_github_url(url)

        assert is_valid is True
        assert error == ""

        # Testa normalização
        normalized = normalize_github_url(url)
        assert normalized == "https://github.com/user/repo.git"

        # Testa que a URL normalizada também é válida
        is_valid_normalized, error_normalized = validate_github_url(normalized)
        assert is_valid_normalized is True
        assert error_normalized == ""

    def test_validate_multiple_url_formats(self):
        """Testa validação de múltiplos formatos de URL."""
        valid_urls = [
            "https://github.com/user/repo",
            "http://github.com/user/repo",
            "https://www.github.com/user/repo",
            "https://github.com/user/repo.git",
            "https://github.com/user-name/repo-name",
            "https://github.com/user123/repo123",
            "https://github.com/USER/REPO",
        ]

        for url in valid_urls:
            is_valid, error = validate_github_url(url)
            assert is_valid is True, f"URL {url} deveria ser válida"
            assert error == "", f"URL {url} não deveria ter erro"

    def test_validate_multiple_invalid_urls(self):
        """Testa validação de múltiplas URLs inválidas."""
        invalid_urls = [
            ("", "URL vazia"),
            ("https://gitlab.com/user/repo", "Domínio errado"),
            ("ftp://github.com/user/repo", "Esquema inválido"),
            ("https://github.com", "Sem caminho"),
            ("https://github.com/user", "Sem repositório"),
            ("https://github.com/-user/repo", "Hífen no início do usuário"),
            ("https://github.com/user/-repo", "Hífen no início do repositório"),
            ("https://github.com/user/repo/invalid/path", "Subcaminho inválido"),
        ]

        for url, description in invalid_urls:
            is_valid, error = validate_github_url(url)
            assert is_valid is False, f"URL {url} deveria ser inválida: {description}"
            assert error != "", f"URL {url} deveria ter erro: {description}"

    def test_normalize_multiple_url_formats(self):
        """Testa normalização de múltiplos formatos de URL."""
        url_formats = [
            ("https://github.com/user/repo", "https://github.com/user/repo.git"),
            ("http://github.com/user/repo", "https://github.com/user/repo.git"),
            ("https://www.github.com/user/repo", "https://github.com/user/repo.git"),
            ("https://github.com/user/repo.git", "https://github.com/user/repo.git"),
            (
                "https://github.com/user/repo/tree/main",
                "https://github.com/user/repo.git",
            ),
            (
                "https://github.com/user/repo/blob/main/file.md",
                "https://github.com/user/repo.git",
            ),
        ]

        for input_url, expected_output in url_formats:
            normalized = normalize_github_url(input_url)
            assert normalized == expected_output, (
                f"URL {input_url} não foi normalizada corretamente"
            )

    def test_error_messages_consistency(self):
        """Testa consistência das mensagens de erro."""
        test_cases = [
            ("", "empty"),
            ("https://gitlab.com/user/repo", "github.com"),
            ("ftp://github.com/user/repo", "http"),
            ("https://github.com", "path"),
            ("https://github.com/user", "owner"),
            ("https://github.com/user/", "repository"),
            ("https://github.com/-user/repo", "owner"),
            ("https://github.com/user/-repo", "repository"),
        ]

        for url, expected_error_keyword in test_cases:
            is_valid, error = validate_github_url(url)
            assert is_valid is False
            assert expected_error_keyword.lower() in error.lower(), (
                f"URL {url} deveria conter '{expected_error_keyword}' no erro"
            )
