# Project Overview

## Purpose
**Crawl4AI RAG MCP Server** is an advanced web crawling and RAG (Retrieval Augmented Generation) system that provides AI agents and coding assistants with intelligent capabilities through the Model Context Protocol (MCP).

## Core Functionality
- **Web Crawling**: Uses Crawl4AI for intelligent web scraping with automatic content extraction
- **GitHub Integration**: Clone and index repositories with multi-file type support
- **Vector Search**: Qdrant vector database for semantic search and document storage
- **Knowledge Graph**: Optional Neo4j integration for AI hallucination detection
- **RAG Strategies**: Multiple configurable strategies including hybrid search, reranking, and contextual embeddings

## Key Features
- **Smart URL Detection**: Automatically detects sitemaps, text files, or regular webpages
- **Multi-Provider API Support**: OpenAI, DeepInfra, Azure OpenAI with fallback configurations
- **GPU Acceleration**: Optional GPU support for reranking models
- **Caching**: Redis-based embedding cache for performance
- **Hybrid Search**: Combines semantic and keyword search using FastBM25
- **AI Hallucination Detection**: Validates AI-generated code against knowledge graphs

## Architecture
- **MCP Server**: FastMCP-based async server with SSE/stdio transport
- **Vector Database**: Qdrant for semantic search and storage
- **Web Crawler**: Crawl4AI with smart URL detection and GitHub integration
- **Knowledge Graph**: Neo4j for code structure analysis
- **Device Manager**: Automatic GPU/CPU detection and fallback
- **Event Loop Fix**: Windows compatibility layer for ConnectionResetError issues

## Technology Stack
- **Language**: Python 3.12+
- **Framework**: FastMCP (Model Context Protocol)
- **Databases**: Qdrant (vector), Neo4j (knowledge graph), Redis (cache)
- **AI APIs**: OpenAI, DeepInfra, Azure OpenAI (multi-provider support)
- **Dependencies**: crawl4ai, qdrant-client, fastembed, sentence-transformers, PyTorch