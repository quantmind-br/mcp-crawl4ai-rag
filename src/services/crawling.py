"""
Web crawling service.

This module provides services for web crawling operations including
single page crawling, batch crawling, and recursive crawling.
"""

import requests
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from typing import List, Dict, Any

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher


class CrawlingService:
    """Service for web crawling operations."""
    
    def __init__(self, crawler: AsyncWebCrawler):
        self.crawler = crawler
    
    def is_sitemap(self, url: str) -> bool:
        """
        Check if a URL is a sitemap.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is a sitemap, False otherwise
        """
        return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path
    
    def is_txt(self, url: str) -> bool:
        """
        Check if a URL is a text file.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is a text file, False otherwise
        """
        return url.endswith('.txt')
    
    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap and extract URLs.
        
        Args:
            sitemap_url: URL of the sitemap
            
        Returns:
            List of URLs found in the sitemap
        """
        resp = requests.get(sitemap_url)
        urls = []

        if resp.status_code == 200:
            try:
                tree = ElementTree.fromstring(resp.content)
                urls = [loc.text for loc in tree.findall('.//{*}loc')]
            except Exception as e:
                print(f"Error parsing sitemap XML: {e}")

        return urls
    
    async def crawl_single_page(self, url: str) -> Dict[str, Any]:
        """
        Crawl a single web page.
        
        Args:
            url: URL of the web page to crawl
            
        Returns:
            Dictionary containing crawl result
        """
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await self.crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            return {
                'url': url,
                'markdown': result.markdown,
                'success': True,
                'links': result.links,
                'error_message': None
            }
        else:
            return {
                'url': url,
                'markdown': None,
                'success': False,
                'links': None,
                'error_message': result.error_message
            }
    
    async def crawl_markdown_file(self, url: str) -> List[Dict[str, Any]]:
        """
        Crawl a .txt or markdown file.
        
        Args:
            url: URL of the file
            
        Returns:
            List of dictionaries with URL and markdown content
        """
        crawl_config = CrawlerRunConfig()

        result = await self.crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{'url': url, 'markdown': result.markdown}]
        else:
            print(f"Failed to crawl {url}: {result.error_message}")
            return []
    
    async def crawl_batch(self, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """
        Batch crawl multiple URLs in parallel.
        
        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum number of concurrent browser sessions
            
        Returns:
            List of dictionaries with URL and markdown content
        """
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent
        )

        results = await self.crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
        return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]
    
    async def crawl_recursive_internal_links(
        self, 
        start_urls: List[str], 
        max_depth: int = 3, 
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recursively crawl internal links from start URLs up to a maximum depth.
        
        Args:
            start_urls: List of starting URLs
            max_depth: Maximum recursion depth
            max_concurrent: Maximum number of concurrent browser sessions
            
        Returns:
            List of dictionaries with URL and markdown content
        """
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent
        )

        visited = set()

        def normalize_url(url):
            return urldefrag(url)[0]

        current_urls = set([normalize_url(u) for u in start_urls])
        results_all = []

        for depth in range(max_depth):
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
            if not urls_to_crawl:
                break

            results = await self.crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)

                if result.success and result.markdown:
                    results_all.append({'url': result.url, 'markdown': result.markdown})
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)

            current_urls = next_level_urls

        return results_all
    
    async def smart_crawl_url(
        self, 
        url: str, 
        max_depth: int = 3, 
        max_concurrent: int = 10
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Intelligently crawl a URL based on its type.
        
        Args:
            url: URL to crawl
            max_depth: Maximum recursion depth for regular URLs
            max_concurrent: Maximum number of concurrent browser sessions
            
        Returns:
            Tuple of (crawl_results, crawl_type)
        """
        crawl_results = []
        crawl_type = None
        
        if self.is_txt(url):
            # For text files, use simple crawl
            crawl_results = await self.crawl_markdown_file(url)
            crawl_type = "text_file"
        elif self.is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = self.parse_sitemap(url)
            if sitemap_urls:
                crawl_results = await self.crawl_batch(sitemap_urls, max_concurrent=max_concurrent)
                crawl_type = "sitemap"
            else:
                crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await self.crawl_recursive_internal_links([url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        return crawl_results, crawl_type