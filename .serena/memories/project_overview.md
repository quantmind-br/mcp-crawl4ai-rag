# Project Overview

## Purpose
Crawl4AI RAG MCP Server is an advanced web crawling and Retrieval-Augmented Generation (RAG) system designed for AI agents and coding assistants. It integrates:

- **Crawl4AI**: Intelligent web crawling with sitemap detection, recursive crawling, and GitHub repository processing
- **Qdrant**: Vector database for semantic search and embedding storage
- **Neo4j**: Knowledge graph for code structure analysis and AI hallucination detection
- **MCP (Model Context Protocol)**: Standardized interface for AI agent integration

## Key Features
1. **Web Crawling**: Smart URL detection, sitemap processing, recursive crawling
2. **GitHub Integration**: Repository cloning, documentation indexing, code structure analysis
3. **RAG Capabilities**: Hybrid search (semantic + keyword), contextual embeddings, reranking
4. **AI Hallucination Detection**: Code validation using knowledge graphs and AST analysis
5. **Multi-Language Support**: Tree-sitter parsers for Python, JavaScript, Java, Go, Rust, C/C++, TypeScript, etc.
6. **Advanced RAG Strategies**: Contextual embeddings, hybrid search, agentic RAG, GPU acceleration

## Architecture
- **MCP Server**: FastMCP-based async server with SSE/stdio transport
- **Vector Database**: Qdrant for semantic search and storage
- **Knowledge Graph**: Neo4j for code relationships and validation
- **Multi-Provider Support**: OpenAI, Azure OpenAI, DeepInfra with fallback configurations
- **Device Management**: Automatic GPU/CPU detection and optimization

## Target Users
- AI coding assistants requiring documentation and code knowledge
- Developers building AI agents with web crawling capabilities
- Teams needing intelligent code analysis and hallucination detection
- Projects requiring advanced RAG capabilities for technical content