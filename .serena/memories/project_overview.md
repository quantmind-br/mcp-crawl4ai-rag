# Project Overview - MCP Crawl4AI RAG Server

## Purpose
This is an advanced Model Context Protocol (MCP) server that integrates multiple technologies to provide AI agents and coding assistants with intelligent web crawling, GitHub repository indexing, vector search, and AI hallucination detection capabilities.

## Core Functionality
- **Web Crawling**: Smart web crawling with sitemap detection and recursive crawling using Crawl4AI
- **GitHub Integration**: Clone and index GitHub repositories for documentation and code analysis
- **Vector Search**: Semantic search capabilities using Qdrant vector database
- **Knowledge Graphs**: Code structure analysis and AI hallucination detection using Neo4j
- **RAG (Retrieval Augmented Generation)**: Advanced RAG strategies with reranking and contextual embeddings

## Key Features
- Multi-language code parsing with Tree-sitter grammars (Python, JavaScript, Java, Go, Rust, C/C++, etc.)
- GPU acceleration support for embedding models
- Fallback API configuration for high availability
- Hybrid search combining semantic and keyword search
- Agentic RAG for code example extraction
- Cross-system file linking between Qdrant and Neo4j

## Target Use Cases
1. **Documentation RAG**: Crawl and index documentation sites for intelligent Q&A
2. **Code Assistant**: Extract code examples and provide coding help
3. **Hallucination Detection**: Validate AI-generated code against repository structure
4. **Repository Analysis**: Understand codebase structure and relationships