# Project Overview: MCP Crawl4AI RAG Server

## Purpose
A powerful MCP (Model Context Protocol) server that integrates web crawling and RAG (Retrieval-Augmented Generation) capabilities for AI agents and coding assistants. The server provides tools to crawl websites, store content in vector databases, and perform semantic search over crawled content.

## Key Features
- **Smart URL Detection**: Automatically handles different URL types (webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size
- **Vector Search**: Performs RAG over crawled content with optional source filtering
- **Advanced RAG Strategies**: Contextual embeddings, hybrid search, agentic RAG, reranking
- **Knowledge Graph**: AI hallucination detection using Neo4j (optional)

## Target Integration
- Primary goal: Integration into [Archon](https://github.com/coleam00/Archon) as a knowledge engine
- Compatible with Claude Desktop, Windsurf, and other MCP clients
- Supports both SSE and stdio transport protocols

## Core Domains
- **Web Crawling**: Using Crawl4AI for intelligent content extraction
- **Vector Storage**: Qdrant for embeddings and semantic search
- **Knowledge Graphs**: Neo4j for code analysis and hallucination detection
- **AI Integration**: OpenAI-compatible APIs for embeddings and chat