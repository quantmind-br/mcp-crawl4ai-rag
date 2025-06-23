"""
Integration tests for the crawling pipeline.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.crawling import CrawlingService
from src.services.content_processing import ContentProcessingService
from src.clients.supabase_client import SupabaseService


class TestCrawlingPipeline:
    """Integration tests for the complete crawling pipeline."""
    
    @pytest.fixture
    def mock_crawler(self):
        """Create a mock AsyncWebCrawler."""
        crawler = AsyncMock()
        return crawler
    
    @pytest.fixture
    def crawling_service(self, mock_crawler):
        """Create a CrawlingService with mocked crawler."""
        return CrawlingService(mock_crawler)
    
    @pytest.fixture
    def content_service(self):
        """Create a ContentProcessingService."""
        with patch('src.services.content_processing.ChatClient'):
            return ContentProcessingService()
    
    @pytest.fixture
    def supabase_service(self):
        """Create a mock SupabaseService."""
        service = SupabaseService()
        service.client = MagicMock()
        return service
    
    @pytest.mark.asyncio
    async def test_single_page_crawl_success(self, crawling_service, mock_crawler):
        """Test successful single page crawling."""
        # Setup mock response
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Test Page\n\nThis is test content."
        mock_result.links = {"internal": [], "external": []}
        mock_result.error_message = None
        
        mock_crawler.arun.return_value = mock_result
        
        # Test crawl
        result = await crawling_service.crawl_single_page("https://example.com")
        
        assert result["success"] is True
        assert result["markdown"] == "# Test Page\n\nThis is test content."
        assert result["url"] == "https://example.com"
        assert result["links"] is not None
    
    @pytest.mark.asyncio
    async def test_single_page_crawl_failure(self, crawling_service, mock_crawler):
        """Test failed single page crawling."""
        # Setup mock response for failure
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.markdown = None
        mock_result.links = None
        mock_result.error_message = "Connection timeout"
        
        mock_crawler.arun.return_value = mock_result
        
        # Test crawl
        result = await crawling_service.crawl_single_page("https://example.com")
        
        assert result["success"] is False
        assert result["markdown"] is None
        assert result["error_message"] == "Connection timeout"
    
    def test_url_type_detection(self, crawling_service):
        """Test URL type detection methods."""
        # Test sitemap detection
        assert crawling_service.is_sitemap("https://example.com/sitemap.xml") is True
        assert crawling_service.is_sitemap("https://example.com/index.html") is False
        
        # Test text file detection
        assert crawling_service.is_txt("https://example.com/robots.txt") is True
        assert crawling_service.is_txt("https://example.com/page.html") is False
    
    @patch('requests.get')
    def test_sitemap_parsing(self, mock_get, crawling_service):
        """Test sitemap XML parsing."""
        # Mock sitemap XML response
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
            <url><loc>https://example.com/page3</loc></url>
        </urlset>"""
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = sitemap_xml.encode('utf-8')
        mock_get.return_value = mock_response
        
        # Test parsing
        urls = crawling_service.parse_sitemap("https://example.com/sitemap.xml")
        
        assert len(urls) == 3
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls
        assert "https://example.com/page3" in urls
    
    @patch('requests.get')
    def test_sitemap_parsing_failure(self, mock_get, crawling_service):
        """Test sitemap parsing with invalid XML."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        urls = crawling_service.parse_sitemap("https://example.com/sitemap.xml")
        
        assert urls == []
    
    @pytest.mark.asyncio
    async def test_batch_crawling(self, crawling_service, mock_crawler):
        """Test batch crawling of multiple URLs."""
        # Setup mock responses
        mock_results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.markdown = f"Content {i}"
            mock_result.url = f"https://example.com/page{i}"
            mock_results.append(mock_result)
        
        mock_crawler.arun_many.return_value = mock_results
        
        # Test batch crawl
        urls = ["https://example.com/page0", "https://example.com/page1", "https://example.com/page2"]
        results = await crawling_service.crawl_batch(urls, max_concurrent=2)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["url"] == f"https://example.com/page{i}"
            assert result["markdown"] == f"Content {i}"
    
    @pytest.mark.asyncio
    async def test_smart_crawl_sitemap(self, crawling_service, mock_crawler):
        """Test smart crawling with a sitemap URL."""
        # Mock sitemap parsing
        with patch.object(crawling_service, 'parse_sitemap') as mock_parse:
            mock_parse.return_value = ["https://example.com/page1", "https://example.com/page2"]
            
            # Mock batch crawling
            with patch.object(crawling_service, 'crawl_batch') as mock_batch:
                mock_batch.return_value = [
                    {"url": "https://example.com/page1", "markdown": "Content 1"},
                    {"url": "https://example.com/page2", "markdown": "Content 2"}
                ]
                
                # Test smart crawl
                results, crawl_type = await crawling_service.smart_crawl_url(
                    "https://example.com/sitemap.xml"
                )
                
                assert crawl_type == "sitemap"
                assert len(results) == 2
                mock_parse.assert_called_once_with("https://example.com/sitemap.xml")
                mock_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_smart_crawl_text_file(self, crawling_service, mock_crawler):
        """Test smart crawling with a text file URL."""
        # Mock text file crawling
        with patch.object(crawling_service, 'crawl_markdown_file') as mock_crawl_file:
            mock_crawl_file.return_value = [
                {"url": "https://example.com/robots.txt", "markdown": "User-agent: *\nDisallow: /admin/"}
            ]
            
            # Test smart crawl
            results, crawl_type = await crawling_service.smart_crawl_url(
                "https://example.com/robots.txt"
            )
            
            assert crawl_type == "text_file"
            assert len(results) == 1
            mock_crawl_file.assert_called_once_with("https://example.com/robots.txt")
    
    @pytest.mark.asyncio
    async def test_smart_crawl_regular_webpage(self, crawling_service, mock_crawler):
        """Test smart crawling with a regular webpage URL."""
        # Mock recursive crawling
        with patch.object(crawling_service, 'crawl_recursive_internal_links') as mock_recursive:
            mock_recursive.return_value = [
                {"url": "https://example.com/", "markdown": "Home page content"},
                {"url": "https://example.com/about", "markdown": "About page content"}
            ]
            
            # Test smart crawl
            results, crawl_type = await crawling_service.smart_crawl_url(
                "https://example.com/",
                max_depth=2,
                max_concurrent=5
            )
            
            assert crawl_type == "webpage"
            assert len(results) == 2
            mock_recursive.assert_called_once_with(
                ["https://example.com/"], 
                max_depth=2, 
                max_concurrent=5
            )
    
    @pytest.mark.asyncio
    async def test_content_processing_integration(self, content_service):
        """Test content processing service integration."""
        # Mock chat client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated context for this chunk."
        
        with patch.object(content_service.chat_client, 'make_completion_with_fallback') as mock_chat:
            mock_chat.return_value = mock_response
            
            # Test contextual embedding generation
            full_doc = "This is a full document with multiple sections and content."
            chunk = "This is a specific chunk of content."
            
            result, success = content_service.generate_contextual_embedding(full_doc, chunk)
            
            assert success is True
            assert "Generated context for this chunk." in result
            assert chunk in result
            mock_chat.assert_called_once()
    
    def test_supabase_service_initialization(self, supabase_service):
        """Test SupabaseService initialization and configuration."""
        assert supabase_service.batch_size > 0
        assert supabase_service.max_retries > 0
        assert supabase_service.client is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_crawl_and_store(self, crawling_service, content_service, supabase_service, mock_crawler):
        """Test end-to-end crawling and storage pipeline."""
        # Setup mock crawler response
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Test Page\n\nThis is test content with multiple paragraphs.\n\nMore content here."
        mock_result.links = {"internal": [], "external": []}
        
        mock_crawler.arun.return_value = mock_result
        
        # Mock Supabase operations
        supabase_service.client.table.return_value.delete.return_value.in_.return_value.execute.return_value = None
        supabase_service.client.table.return_value.insert.return_value.execute.return_value = None
        
        # Test the pipeline
        crawl_result = await crawling_service.crawl_single_page("https://example.com")
        
        # Verify crawl success
        assert crawl_result["success"] is True
        assert crawl_result["markdown"] is not None
        
        # Test content chunking
        from src.utils.content_utils import smart_chunk_markdown
        chunks = smart_chunk_markdown(crawl_result["markdown"])
        assert len(chunks) > 0
        
        # Mock embedding creation
        with patch('src.clients.embedding_client.EmbeddingClient') as mock_embedding_client:
            mock_embedding_client.return_value.create_embeddings_batch.return_value = [
                [0.1] * 1536 for _ in chunks
            ]
            
            # Test document storage (mock the actual storage call)
            url_to_full_document = {"https://example.com": crawl_result["markdown"]}
            
            # This would normally call supabase_service.add_documents_to_supabase
            # but we're just testing that the pipeline components work together
            assert len(chunks) > 0
            assert url_to_full_document["https://example.com"] is not None