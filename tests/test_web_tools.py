"""
Testes para as funcionalidades de web_tools.py.

Este módulo contém testes para as ferramentas de crawling web, incluindo:
- Crawling de páginas únicas
- Crawling inteligente de URLs
- Extração de blocos de código
- Processamento de sitemaps
- Chunking inteligente de markdown
- Extração de informações de seção
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock

from src.tools.web_tools import (
    extract_source_summary,
    extract_code_blocks,
    generate_code_example_summary,
    is_sitemap,
    is_txt,
    parse_sitemap,
    smart_chunk_markdown,
    extract_section_info,
    process_code_example,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links,
    crawl_single_page,
    smart_crawl_url,
    create_base_prefix,
)


class TestWebToolsUtilityFunctions:
    """Testes para funções utilitárias de web_tools."""

    def test_extract_source_summary_empty_content(self):
        """Testa extração de resumo para conteúdo vazio."""
        result = extract_source_summary("test.com", "")
        assert "Empty source: test.com" in result

    def test_extract_source_summary_short_content(self):
        """Testa extração de resumo para conteúdo curto."""
        content = "Este é um conteúdo de teste para extração de resumo."
        result = extract_source_summary("test.com", content)
        # A função retorna o conteúdo direto se for curto
        assert "Este é um conteúdo de teste" in result

    def test_extract_source_summary_long_content(self):
        """Testa extração de resumo para conteúdo longo."""
        content = "A" * 1000
        result = extract_source_summary("test.com", content, max_length=100)
        # A função retorna "Content from test.com" quando o conteúdo é muito longo
        assert "Content from test.com" in result

    def test_extract_code_blocks_empty_content(self):
        """Testa extração de blocos de código de conteúdo vazio."""
        result = extract_code_blocks("")
        assert result == []

    def test_extract_code_blocks_no_code_blocks(self):
        """Testa extração quando não há blocos de código."""
        content = "Este é um texto simples sem blocos de código."
        result = extract_code_blocks(content)
        assert result == []

    def test_extract_code_blocks_single_block(self):
        """Testa extração de um único bloco de código."""
        content = """
        # Título
        
        Aqui está um exemplo de código:
        
        ```python
        def hello_world():
            print("Hello, World!")
        ```
        
        Mais texto aqui.
        """
        result = extract_code_blocks(
            content, min_length=10
        )  # Reduzir min_length para teste
        assert len(result) == 1
        assert result[0]["language"] == "python"
        assert "def hello_world():" in result[0]["code"]

    def test_extract_code_blocks_multiple_blocks(self):
        """Testa extração de múltiplos blocos de código."""
        content = """
        # Documentação
        
        Exemplo 1:
        ```javascript
        function test() {
            return true;
        }
        ```
        
        Exemplo 2:
        ```python
        def test():
            return True
        ```
        """
        result = extract_code_blocks(
            content, min_length=10
        )  # Reduzir min_length para teste
        assert len(result) == 2
        assert result[0]["language"] == "javascript"
        assert result[1]["language"] == "python"

    def test_extract_code_blocks_with_context(self):
        """Testa extração de blocos de código com contexto."""
        content = """
        Antes do código:
        
        ```python
        def example():
            pass
        ```
        
        Depois do código.
        """
        result = extract_code_blocks(
            content, min_length=10
        )  # Reduzir min_length para teste
        assert len(result) == 1
        assert "Antes do código:" in result[0]["context_before"]
        assert "Depois do código." in result[0]["context_after"]

    def test_generate_code_example_summary(self):
        """Testa geração de resumo para exemplo de código."""
        code = "def hello(): print('Hello')"
        context_before = "Antes do código"
        context_after = "Depois do código"

        result = generate_code_example_summary(code, context_before, context_after)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_is_sitemap_valid_sitemap(self):
        """Testa detecção de sitemap válido."""
        assert is_sitemap("https://example.com/sitemap.xml")
        assert is_sitemap("https://example.com/sitemap_index.xml")

    def test_is_sitemap_invalid_sitemap(self):
        """Testa detecção de URLs que não são sitemaps."""
        assert not is_sitemap("https://example.com/page.html")
        assert not is_sitemap("https://example.com/api/endpoint")

    def test_is_txt_valid_txt(self):
        """Testa detecção de arquivos .txt válidos."""
        assert is_txt("https://example.com/file.txt")
        assert is_txt("https://example.com/llms.txt")

    def test_is_txt_invalid_txt(self):
        """Testa detecção de URLs que não são arquivos .txt."""
        assert not is_txt("https://example.com/file.html")
        assert not is_txt("https://example.com/file.pdf")

    def test_smart_chunk_markdown_empty_text(self):
        """Testa chunking de texto vazio."""
        result = smart_chunk_markdown("")
        assert result == []

    def test_smart_chunk_markdown_short_text(self):
        """Testa chunking de texto curto."""
        text = "Texto curto que não precisa ser dividido."
        result = smart_chunk_markdown(text, chunk_size=1000)
        assert len(result) == 1
        assert result[0] == text

    def test_smart_chunk_markdown_long_text(self):
        """Testa chunking de texto longo."""
        text = "Parágrafo 1.\n\n" * 10 + "Parágrafo final."
        result = smart_chunk_markdown(text, chunk_size=50)
        assert len(result) > 1
        assert all(len(chunk) <= 50 for chunk in result)

    def test_smart_chunk_markdown_with_code_blocks(self):
        """Testa chunking respeitando blocos de código."""
        text = """
        Introdução.
        
        ```python
        def long_function():
            # Código muito longo que não deve ser quebrado
            pass
        ```
        
        Conclusão.
        """
        result = smart_chunk_markdown(text, chunk_size=30)
        assert len(result) > 1
        # Verifica se os blocos de código não foram quebrados
        code_blocks = [chunk for chunk in result if "```python" in chunk]
        assert len(code_blocks) > 0

    def test_extract_section_info(self):
        """Testa extração de informações de seção."""
        chunk = """
        # Título Principal
        
        ## Subtítulo
        
        Conteúdo do parágrafo com **formatação**.
        
        - Item 1
        - Item 2
        """
        result = extract_section_info(chunk)

        assert "word_count" in result
        assert "char_count" in result
        assert (
            "headers" in result
        )  # Corrigido: a função retorna "headers" não "has_headers"
        assert result["word_count"] > 0
        assert result["char_count"] > 0

    def test_process_code_example(self):
        """Testa processamento de exemplo de código."""
        args = ("def test(): pass", "Antes", "Depois")
        result = process_code_example(args)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_base_prefix_root_url(self):
        """Testa criação de prefixo base para URL raiz."""
        # Casos com trailing slash
        assert create_base_prefix("https://opencode.ai/") == "https://opencode.ai/"
        assert create_base_prefix("http://example.com/") == "http://example.com/"

        # Casos sem trailing slash
        assert create_base_prefix("https://opencode.ai") == "https://opencode.ai/"
        assert create_base_prefix("http://example.com") == "http://example.com/"

    def test_create_base_prefix_with_path(self):
        """Testa criação de prefixo base para URLs com path."""
        # Casos com trailing slash
        assert (
            create_base_prefix("https://docs.anthropic.com/en/")
            == "https://docs.anthropic.com/en/"
        )
        assert (
            create_base_prefix("https://example.com/docs/")
            == "https://example.com/docs/"
        )

        # Casos sem trailing slash
        assert (
            create_base_prefix("https://docs.anthropic.com/en")
            == "https://docs.anthropic.com/en/"
        )
        assert (
            create_base_prefix("https://example.com/docs")
            == "https://example.com/docs/"
        )

    def test_create_base_prefix_deep_path(self):
        """Testa criação de prefixo base para URLs com paths profundos."""
        assert (
            create_base_prefix("https://docs.anthropic.com/en/docs/claude-code/")
            == "https://docs.anthropic.com/en/docs/claude-code/"
        )
        assert (
            create_base_prefix("https://docs.anthropic.com/en/docs/claude-code")
            == "https://docs.anthropic.com/en/docs/claude-code/"
        )

    def test_create_base_prefix_url_filtering_examples(self):
        """Testa exemplos específicos mencionados no problema."""
        # Exemplo 1: opencode.ai deve incluir opencode.ai/docs/, opencode.ai/docs/cli/
        base_prefix = create_base_prefix("https://opencode.ai/")
        assert "https://opencode.ai/docs/".startswith(base_prefix)
        assert "https://opencode.ai/docs/cli/".startswith(base_prefix)
        assert "https://opencode.ai/anything".startswith(base_prefix)

        # Exemplo 2: docs.anthropic.com/en deve excluir docs.anthropic.com/pt/*
        base_prefix = create_base_prefix("https://docs.anthropic.com/en/")
        assert "https://docs.anthropic.com/en/docs/".startswith(base_prefix)
        assert "https://docs.anthropic.com/en/docs/claude/".startswith(base_prefix)
        assert not "https://docs.anthropic.com/pt/docs/".startswith(base_prefix)
        assert not "https://docs.anthropic.com/fr/docs/".startswith(base_prefix)


class TestWebToolsCrawlingFunctions:
    """Testes para funções de crawling."""

    @pytest.mark.asyncio
    async def test_crawl_markdown_file(self):
        """Testa crawling de arquivo markdown."""
        mock_crawler = AsyncMock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Título\n\nConteúdo do arquivo."
        mock_crawler.arun.return_value = mock_result

        result = await crawl_markdown_file(mock_crawler, "https://example.com/file.md")

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/file.md"
        assert "Título" in result[0]["markdown"]

    @pytest.mark.asyncio
    async def test_crawl_batch(self):
        """Testa crawling em lote."""
        mock_crawler = AsyncMock()
        mock_result1 = Mock()
        mock_result1.success = True
        mock_result1.markdown = "Conteúdo 1"
        mock_result1.url = "https://example.com/1"

        mock_result2 = Mock()
        mock_result2.success = True
        mock_result2.markdown = "Conteúdo 2"
        mock_result2.url = "https://example.com/2"

        mock_crawler.arun_many.return_value = [mock_result1, mock_result2]

        urls = ["https://example.com/1", "https://example.com/2"]
        result = await crawl_batch(mock_crawler, urls, max_concurrent=2)

        assert len(result) == 2
        assert all("url" in item for item in result)
        assert all("markdown" in item for item in result)

    @pytest.mark.asyncio
    async def test_crawl_recursive_internal_links(self):
        """Testa crawling recursivo de links internos."""
        mock_crawler = AsyncMock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.url = "https://example.com"  # Proper string URL
        mock_result.markdown = "Conteúdo da página"
        mock_result.links = {
            "internal": []
        }  # No internal links to avoid infinite recursion in test
        mock_crawler.arun_many.return_value = [mock_result]

        start_urls = ["https://example.com"]
        base_prefix = "https://example.com"
        result = await crawl_recursive_internal_links(
            mock_crawler,
            start_urls,
            base_prefix=base_prefix,
            max_depth=2,
            max_concurrent=5,
        )

        assert isinstance(result, list)
        assert len(result) == 1  # Should have one result
        assert result[0]["url"] == "https://example.com"
        assert result[0]["markdown"] == "Conteúdo da página"

    @pytest.mark.asyncio
    async def test_crawl_recursive_internal_links_respects_base_prefix(self):
        """Test that crawl_recursive_internal_links respects base_prefix filtering."""
        mock_crawler = AsyncMock()

        # Mock results with mixed URL prefixes
        mock_result_1 = Mock()
        mock_result_1.success = True
        mock_result_1.url = "https://docs.anthropic.com/en/docs/introduction"
        mock_result_1.markdown = "Introduction content"
        mock_result_1.links = {
            "internal": [
                {
                    "href": "https://docs.anthropic.com/en/docs/quickstart"
                },  # Should be included
                {
                    "href": "https://docs.anthropic.com/pt/docs/introducao"
                },  # Should be filtered out
                {
                    "href": "https://docs.anthropic.com/en/api/reference"
                },  # Should be included
                {
                    "href": "https://docs.anthropic.com/fr/docs/introduction"
                },  # Should be filtered out
            ]
        }

        mock_result_2 = Mock()
        mock_result_2.success = True
        mock_result_2.url = "https://docs.anthropic.com/en/docs/quickstart"
        mock_result_2.markdown = "Quickstart content"
        mock_result_2.links = {
            "internal": [
                {
                    "href": "https://docs.anthropic.com/en/api/reference"
                },  # Should be included (but already visited)
                {
                    "href": "https://docs.anthropic.com/es/docs/inicio"
                },  # Should be filtered out
            ]
        }

        mock_result_3 = Mock()
        mock_result_3.success = True
        mock_result_3.url = "https://docs.anthropic.com/en/api/reference"
        mock_result_3.markdown = "API reference content"
        mock_result_3.links = {"internal": []}  # No more links

        # Configure mock to return different results for different calls
        mock_crawler.arun_many.side_effect = [
            [mock_result_1],  # First depth: start URL
            [mock_result_2, mock_result_3],  # Second depth: only /en URLs
        ]

        start_urls = ["https://docs.anthropic.com/en/docs/introduction"]
        base_prefix = "https://docs.anthropic.com/en"

        result = await crawl_recursive_internal_links(
            mock_crawler,
            start_urls,
            base_prefix=base_prefix,
            max_depth=3,
            max_concurrent=5,
        )

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 3  # Should have 3 results (all from /en prefix)

        # Verify all results are from the correct prefix
        for doc in result:
            assert doc["url"].startswith(base_prefix), (
                f"URL {doc['url']} doesn't match prefix {base_prefix}"
            )

        # Verify specific URLs are included
        crawled_urls = [doc["url"] for doc in result]
        assert "https://docs.anthropic.com/en/docs/introduction" in crawled_urls
        assert "https://docs.anthropic.com/en/docs/quickstart" in crawled_urls
        assert "https://docs.anthropic.com/en/api/reference" in crawled_urls

        # Verify crawler was called the expected number of times (2 depths - function stops when no new URLs)
        assert mock_crawler.arun_many.call_count == 2

    @pytest.mark.asyncio
    async def test_crawl_single_page(self):
        """Testa crawling de página única."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.crawler = AsyncMock()
        # A implementação atual requer um qdrant_client no contexto
        mock_context.request_context.lifespan_context.qdrant_client = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "Conteúdo da página"
        mock_context.request_context.lifespan_context.crawler.arun.return_value = (
            mock_result
        )

        # Evita efeitos externos patchando funções de persistência
        with patch("src.tools.web_tools.update_source_info"):
            with patch("src.tools.web_tools.add_documents_to_vector_db"):
                result = await crawl_single_page(mock_context, "https://example.com")

        assert isinstance(result, str)
        result_data = json.loads(result)
        assert result_data["success"] is True

    @pytest.mark.asyncio
    async def test_smart_crawl_url_txt_file(self):
        """Testa crawling inteligente de arquivo .txt."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.crawler = AsyncMock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()

        # Mock para crawl_markdown_file
        with patch("src.tools.web_tools.is_txt", return_value=True):
            with patch("src.tools.web_tools.crawl_markdown_file") as mock_crawl:
                mock_crawl.return_value = [
                    {"url": "https://example.com/file.txt", "markdown": "Conteúdo"}
                ]

                # Mock para as funções de processamento
                with patch(
                    "src.tools.web_tools.extract_source_summary"
                ) as mock_extract:
                    with patch("src.tools.web_tools.update_source_info"):
                        with patch("src.tools.web_tools.add_documents_to_vector_db"):
                            mock_extract.return_value = "Resumo do conteúdo"

                            result = await smart_crawl_url(
                                mock_context, "https://example.com/file.txt"
                            )

                            assert isinstance(result, str)
                            result_data = json.loads(result)
                            assert result_data["success"] is True
                            assert result_data["crawl_type"] == "text_file"

    @pytest.mark.asyncio
    async def test_smart_crawl_url_sitemap(self):
        """Testa crawling inteligente de sitemap."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.crawler = AsyncMock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()

        with patch("src.tools.web_tools.is_sitemap", return_value=True):
            with patch("src.tools.web_tools.parse_sitemap") as mock_parse:
                with patch("src.tools.web_tools.crawl_batch") as mock_crawl:
                    with patch(
                        "src.tools.web_tools.extract_source_summary"
                    ) as mock_extract:
                        with patch("src.tools.web_tools.update_source_info"):
                            with patch(
                                "src.tools.web_tools.add_documents_to_vector_db"
                            ):
                                mock_parse.return_value = [
                                    "https://example.com/page1",
                                    "https://example.com/page2",
                                ]
                                mock_crawl.return_value = [
                                    {
                                        "url": "https://example.com/page1",
                                        "markdown": "Conteúdo 1",
                                    },
                                    {
                                        "url": "https://example.com/page2",
                                        "markdown": "Conteúdo 2",
                                    },
                                ]
                                mock_extract.return_value = "Resumo do conteúdo"

                                result = await smart_crawl_url(
                                    mock_context, "https://example.com/sitemap.xml"
                                )

                                assert isinstance(result, str)
                                result_data = json.loads(result)
                                assert result_data["success"] is True
                                assert result_data["crawl_type"] == "sitemap"

    @pytest.mark.asyncio
    async def test_smart_crawl_url_regular_webpage(self):
        """Testa crawling inteligente de página web regular."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.crawler = AsyncMock()
        mock_context.request_context.lifespan_context.qdrant_client = Mock()

        with patch("src.tools.web_tools.is_txt", return_value=False):
            with patch("src.tools.web_tools.is_sitemap", return_value=False):
                with patch(
                    "src.tools.web_tools.crawl_recursive_internal_links"
                ) as mock_crawl:
                    with patch(
                        "src.tools.web_tools.extract_source_summary"
                    ) as mock_extract:
                        with patch("src.tools.web_tools.update_source_info"):
                            with patch(
                                "src.tools.web_tools.add_documents_to_vector_db"
                            ):
                                mock_crawl.return_value = [
                                    {
                                        "url": "https://example.com",
                                        "markdown": "Conteúdo da página",
                                    }
                                ]
                                mock_extract.return_value = "Resumo do conteúdo"

                                result = await smart_crawl_url(
                                    mock_context, "https://example.com"
                                )

                                assert isinstance(result, str)
                                result_data = json.loads(result)
                                assert result_data["success"] is True
                                assert result_data["crawl_type"] == "webpage"


class TestWebToolsErrorHandling:
    """Testes para tratamento de erros em web_tools."""

    @pytest.mark.asyncio
    async def test_crawl_single_page_error(self):
        """Testa tratamento de erro no crawling de página única."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.crawler = AsyncMock()
        mock_context.request_context.lifespan_context.crawler.arun.side_effect = (
            Exception("Erro de rede")
        )

        result = await crawl_single_page(mock_context, "https://example.com")

        assert isinstance(result, str)
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "error" in result_data

    @pytest.mark.asyncio
    async def test_smart_crawl_url_error(self):
        """Testa tratamento de erro no crawling inteligente."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.crawler = AsyncMock()
        mock_context.request_context.lifespan_context.crawler.arun.side_effect = (
            Exception("Erro de crawling")
        )

        result = await smart_crawl_url(mock_context, "https://example.com")

        assert isinstance(result, str)
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "error" in result_data

    def test_parse_sitemap_error(self):
        """Testa tratamento de erro na análise de sitemap."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Erro de rede")

            result = parse_sitemap("https://example.com/sitemap.xml")
            assert result == []

    def test_extract_code_blocks_invalid_content(self):
        """Testa extração de blocos de código com conteúdo inválido."""
        # A função não trata None, então deve gerar erro
        with pytest.raises(AttributeError):
            extract_code_blocks(None)


class TestWebToolsIntegration:
    """Testes de integração para web_tools."""

    @pytest.mark.asyncio
    async def test_full_crawling_workflow(self):
        """Testa workflow completo de crawling."""
        mock_context = Mock()
        mock_context.request_context.lifespan_context.crawler = AsyncMock()
        mock_context.request_context.lifespan_context.crawler.arun.return_value = Mock(
            success=True, markdown="# Título\n\nConteúdo com **formatação**."
        )
        mock_context.request_context.lifespan_context.qdrant_client = Mock()

        # Simula o workflow completo
        with patch("src.tools.web_tools.is_txt", return_value=False):
            with patch("src.tools.web_tools.is_sitemap", return_value=False):
                with patch(
                    "src.tools.web_tools.crawl_recursive_internal_links"
                ) as mock_crawl:
                    with patch(
                        "src.tools.web_tools.extract_source_summary"
                    ) as mock_extract:
                        with patch("src.tools.web_tools.update_source_info"):
                            with patch(
                                "src.tools.web_tools.add_documents_to_vector_db"
                            ):
                                mock_crawl.return_value = [
                                    {
                                        "url": "https://example.com",
                                        "markdown": "# Título\n\nConteúdo.",
                                    }
                                ]
                                mock_extract.return_value = "Resumo do conteúdo"

                                result = await smart_crawl_url(
                                    mock_context, "https://example.com"
                                )

                                assert isinstance(result, str)
                                result_data = json.loads(result)
                                assert result_data["success"] is True
                                assert "chunks_stored" in result_data
                                assert "pages_crawled" in result_data

    def test_code_block_extraction_with_context(self):
        """Testa extração de blocos de código com contexto completo."""
        content = """
        # Documentação da API
        
        Aqui está um exemplo de como usar a função:
        
        ```python
        def example_function(param):
            \"\"\"
            Função de exemplo.
            
            Args:
                param: Parâmetro de entrada
            \"\"\"
            return param * 2
        ```
        
        Para usar esta função:
        
        ```python
        result = example_function(5)
        print(result)  # Saída: 10
        ```
        
        ## Conclusão
        
        Esta é a conclusão da documentação.
        """

        result = extract_code_blocks(
            content, min_length=10
        )  # Reduzir min_length para teste

        assert len(result) == 2
        assert result[0]["language"] == "python"
        assert result[1]["language"] == "python"
        assert "def example_function" in result[0]["code"]
        assert "result = example_function" in result[1]["code"]
        assert "Documentação da API" in result[0]["context_before"]
        assert "Conclusão" in result[1]["context_after"]
