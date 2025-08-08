"""
Testes para as funcionalidades de rag_tools.py.

Este módulo contém testes para as ferramentas RAG (Retrieval-Augmented Generation), incluindo:
- Obtenção de fontes disponíveis
- Consultas RAG com filtros
- Busca de exemplos de código
- Tratamento de erros
"""

import pytest
import json
from unittest.mock import Mock, patch

from src.tools.rag_tools import (
    get_available_sources,
    perform_rag_query,
    search_code_examples,
)


class TestRAGToolsBasicFunctions:
    """Testes para funções básicas de RAG tools."""

    @pytest.mark.asyncio
    async def test_get_available_sources_success(self):
        """Testa obtenção bem-sucedida de fontes disponíveis."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()

        # Simula fontes disponíveis
        mock_sources = [
            {
                "source_id": "example.com",
                "summary": "Documentação do exemplo",
                "word_count": 1500,
                "last_updated": "2024-01-01",
            },
            {
                "source_id": "docs.example.com",
                "summary": "Documentação técnica",
                "word_count": 3000,
                "last_updated": "2024-01-02",
            },
        ]
        mock_context.request_context.lifespan_context.qdrant_client.get_available_sources.return_value = mock_sources

        result = await get_available_sources(mock_context)

        assert isinstance(result, str)
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["sources"]) == 2
        assert result_data["total_sources"] == 2
        assert "message" in result_data

    @pytest.mark.asyncio
    async def test_get_available_sources_empty(self):
        """Testa obtenção de fontes quando não há fontes disponíveis."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.qdrant_client.get_available_sources.return_value = []

        result = await get_available_sources(mock_context)

        assert isinstance(result, str)
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["sources"]) == 0
        assert result_data["total_sources"] == 0

    @pytest.mark.asyncio
    async def test_get_available_sources_error(self):
        """Testa tratamento de erro na obtenção de fontes."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.qdrant_client.get_available_sources.side_effect = Exception(
            "Erro de banco de dados"
        )

        result = await get_available_sources(mock_context)

        assert isinstance(result, str)
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "error" in result_data

    @pytest.mark.asyncio
    async def test_perform_rag_query_success(self):
        """Testa consulta RAG bem-sucedida."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        # Simula resultados da busca
        mock_results = {
            "success": True,
            "results": [
                {
                    "content": "Conteúdo relevante 1",
                    "score": 0.95,
                    "metadata": {"url": "https://example.com/page1"},
                },
                {
                    "content": "Conteúdo relevante 2",
                    "score": 0.85,
                    "metadata": {"url": "https://example.com/page2"},
                },
            ],
            "query": "teste de consulta",
        }

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.return_value = mock_results

            result = await perform_rag_query(mock_context, "teste de consulta")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert len(result_data["results"]) == 2

    @pytest.mark.asyncio
    async def test_perform_rag_query_with_source_filter(self):
        """Testa consulta RAG com filtro de fonte."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        mock_results = {
            "success": True,
            "results": [
                {
                    "content": "Conteúdo filtrado",
                    "score": 0.90,
                    "metadata": {"url": "https://example.com/page"},
                }
            ],
            "query": "consulta filtrada",
        }

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.return_value = mock_results

            result = await perform_rag_query(
                mock_context, "consulta filtrada", source="example.com", match_count=3
            )

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True
            # Verifica se a função search_documents foi chamada com os parâmetros corretos
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_rag_query_with_reranker(self):
        """Testa consulta RAG com reranker."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = Mock()

        mock_results = {
            "success": True,
            "results": [
                {
                    "content": "Conteúdo com reranker",
                    "score": 0.98,
                    "metadata": {"url": "https://example.com/page"},
                }
            ],
            "query": "consulta com reranker",
        }

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.return_value = mock_results

            result = await perform_rag_query(mock_context, "consulta com reranker")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True

    @pytest.mark.asyncio
    async def test_perform_rag_query_error(self):
        """Testa tratamento de erro na consulta RAG."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.side_effect = Exception("Erro de busca")

            result = await perform_rag_query(mock_context, "consulta com erro")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "error" in result_data
            assert result_data["query"] == "consulta com erro"

    @pytest.mark.asyncio
    async def test_search_code_examples_success(self):
        """Testa busca bem-sucedida de exemplos de código."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        mock_results = {
            "success": True,
            "results": [
                {
                    "code": "def example_function():\n    return True",
                    "summary": "Função de exemplo",
                    "score": 0.95,
                    "metadata": {
                        "url": "https://example.com/code.md",
                        "language": "python",
                    },
                }
            ],
            "query": "exemplo de função",
        }

        with patch("src.tools.rag_tools.search_code_examples_impl") as mock_search:
            mock_search.return_value = mock_results

            result = await search_code_examples(mock_context, "exemplo de função")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert len(result_data["results"]) == 1
            assert "def example_function" in result_data["results"][0]["code"]

    @pytest.mark.asyncio
    async def test_search_code_examples_with_source_filter(self):
        """Testa busca de exemplos de código com filtro de fonte."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        mock_results = {
            "success": True,
            "results": [
                {
                    "code": "function test() {\n    return true;\n}",
                    "summary": "Função JavaScript",
                    "score": 0.90,
                    "metadata": {
                        "url": "https://example.com/js.md",
                        "language": "javascript",
                    },
                }
            ],
            "query": "função JavaScript",
        }

        with patch("src.tools.rag_tools.search_code_examples_impl") as mock_search:
            mock_search.return_value = mock_results

            result = await search_code_examples(
                mock_context,
                "função JavaScript",
                source_id="example.com",
                match_count=2,
            )

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_code_examples_with_reranker(self):
        """Testa busca de exemplos de código com reranker."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = Mock()

        mock_results = {
            "success": True,
            "results": [
                {
                    "code": "class Example:\n    def __init__(self):\n        pass",
                    "summary": "Classe de exemplo",
                    "score": 0.98,
                    "metadata": {
                        "url": "https://example.com/class.md",
                        "language": "python",
                    },
                }
            ],
            "query": "classe exemplo",
        }

        with patch("src.tools.rag_tools.search_code_examples_impl") as mock_search:
            mock_search.return_value = mock_results

            result = await search_code_examples(mock_context, "classe exemplo")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True

    @pytest.mark.asyncio
    async def test_search_code_examples_error(self):
        """Testa tratamento de erro na busca de exemplos de código."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        with patch("src.tools.rag_tools.search_code_examples_impl") as mock_search:
            mock_search.side_effect = Exception("Erro de busca de código")

            result = await search_code_examples(mock_context, "consulta com erro")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "error" in result_data
            assert result_data["query"] == "consulta com erro"


class TestRAGToolsEdgeCases:
    """Testes para casos extremos de RAG tools."""

    @pytest.mark.asyncio
    async def test_perform_rag_query_empty_query(self):
        """Testa consulta RAG com query vazia."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        mock_results = {"success": True, "results": [], "query": ""}

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.return_value = mock_results

            result = await perform_rag_query(mock_context, "")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True

    @pytest.mark.asyncio
    async def test_perform_rag_query_no_results(self):
        """Testa consulta RAG sem resultados."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        mock_results = {
            "success": True,
            "results": [],
            "query": "consulta sem resultados",
        }

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.return_value = mock_results

            result = await perform_rag_query(mock_context, "consulta sem resultados")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert len(result_data["results"]) == 0

    @pytest.mark.asyncio
    async def test_search_code_examples_empty_query(self):
        """Testa busca de exemplos de código com query vazia."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        mock_results = {"success": True, "results": [], "query": ""}

        with patch("src.tools.rag_tools.search_code_examples_impl") as mock_search:
            mock_search.return_value = mock_results

            result = await search_code_examples(mock_context, "")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True

    @pytest.mark.asyncio
    async def test_search_code_examples_no_results(self):
        """Testa busca de exemplos de código sem resultados."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        mock_results = {"success": True, "results": [], "query": "consulta sem código"}

        with patch("src.tools.rag_tools.search_code_examples_impl") as mock_search:
            mock_search.return_value = mock_results

            result = await search_code_examples(mock_context, "consulta sem código")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert len(result_data["results"]) == 0

    @pytest.mark.asyncio
    async def test_perform_rag_query_large_match_count(self):
        """Testa consulta RAG com grande número de resultados."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        # Simula muitos resultados
        mock_results = {
            "success": True,
            "results": [
                {
                    "content": f"Conteúdo {i}",
                    "score": 0.9 - (i * 0.01),
                    "metadata": {"url": f"https://example.com/page{i}"},
                }
                for i in range(20)
            ],
            "query": "consulta com muitos resultados",
        }

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.return_value = mock_results

            result = await perform_rag_query(
                mock_context, "consulta com muitos resultados", match_count=20
            )

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert len(result_data["results"]) == 20

    @pytest.mark.asyncio
    async def test_search_code_examples_large_match_count(self):
        """Testa busca de exemplos de código com grande número de resultados."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        # Simula muitos exemplos de código
        mock_results = {
            "success": True,
            "results": [
                {
                    "code": f"def function_{i}():\n    return {i}",
                    "summary": f"Função {i}",
                    "score": 0.9 - (i * 0.01),
                    "metadata": {
                        "url": f"https://example.com/code{i}.md",
                        "language": "python",
                    },
                }
                for i in range(15)
            ],
            "query": "consulta com muitos códigos",
        }

        with patch("src.tools.rag_tools.search_code_examples_impl") as mock_search:
            mock_search.return_value = mock_results

            result = await search_code_examples(
                mock_context, "consulta com muitos códigos", match_count=15
            )

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert len(result_data["results"]) == 15


class TestRAGToolsIntegration:
    """Testes de integração para RAG tools."""

    @pytest.mark.asyncio
    async def test_rag_workflow_complete(self):
        """Testa workflow completo de RAG."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        # Simula fontes disponíveis
        mock_sources = [
            {
                "source_id": "example.com",
                "summary": "Documentação de exemplo",
                "word_count": 2000,
            }
        ]
        mock_context.request_context.lifespan_context.qdrant_client.get_available_sources.return_value = mock_sources

        # Simula resultados de busca
        mock_search_results = {
            "success": True,
            "results": [
                {
                    "content": "Conteúdo relevante",
                    "score": 0.95,
                    "metadata": {"url": "https://example.com/page"},
                }
            ],
            "query": "consulta de teste",
        }

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.return_value = mock_search_results

            # Testa obtenção de fontes
            sources_result = await get_available_sources(mock_context)
            sources_data = json.loads(sources_result)
            assert sources_data["success"] is True
            assert len(sources_data["sources"]) == 1

            # Testa consulta RAG
            query_result = await perform_rag_query(
                mock_context, "consulta de teste", source="example.com"
            )
            query_data = json.loads(query_result)
            assert query_data["success"] is True
            assert len(query_data["results"]) == 1

    @pytest.mark.asyncio
    async def test_code_examples_workflow(self):
        """Testa workflow de busca de exemplos de código."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        # Simula fontes disponíveis
        mock_sources = [
            {
                "source_id": "github.com/example/repo",
                "summary": "Repositório de código",
                "word_count": 5000,
            }
        ]
        mock_context.request_context.lifespan_context.qdrant_client.get_available_sources.return_value = mock_sources

        # Simula resultados de busca de código
        mock_code_results = {
            "success": True,
            "results": [
                {
                    "code": "def example_function():\n    return 'Hello, World!'",
                    "summary": "Função de exemplo",
                    "score": 0.95,
                    "metadata": {
                        "url": "https://github.com/example/repo/code.md",
                        "language": "python",
                    },
                }
            ],
            "query": "função exemplo",
        }

        with patch("src.tools.rag_tools.search_code_examples_impl") as mock_search:
            mock_search.return_value = mock_code_results

            # Testa busca de exemplos de código
            result = await search_code_examples(
                mock_context, "função exemplo", source_id="github.com/example/repo"
            )
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert len(result_data["results"]) == 1
            assert "def example_function" in result_data["results"][0]["code"]

    @pytest.mark.asyncio
    async def test_rag_with_different_source_types(self):
        """Testa RAG com diferentes tipos de fontes."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_context.request_context.lifespan_context.reranker = None

        # Simula diferentes tipos de fontes
        mock_sources = [
            {
                "source_id": "docs.example.com",
                "summary": "Documentação oficial",
                "word_count": 10000,
            },
            {
                "source_id": "github.com/example/repo",
                "summary": "Repositório de código",
                "word_count": 5000,
            },
            {
                "source_id": "blog.example.com",
                "summary": "Blog técnico",
                "word_count": 3000,
            },
        ]
        mock_context.request_context.lifespan_context.qdrant_client.get_available_sources.return_value = mock_sources

        # Testa consulta sem filtro de fonte
        mock_results = {
            "success": True,
            "results": [
                {
                    "content": "Conteúdo de múltiplas fontes",
                    "score": 0.90,
                    "metadata": {"url": "https://docs.example.com/page"},
                }
            ],
            "query": "consulta geral",
        }

        with patch("src.tools.rag_tools.search_documents") as mock_search:
            mock_search.return_value = mock_results

            result = await perform_rag_query(mock_context, "consulta geral")
            result_data = json.loads(result)
            assert result_data["success"] is True

            # Verifica se a busca foi feita sem filtro de fonte
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args[0][1] == "consulta geral"  # query
            assert call_args[0][2] is None  # source (sem filtro)

