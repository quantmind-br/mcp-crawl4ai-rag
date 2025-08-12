# Project Overview - Crawl4AI MCP RAG Server

## Purpose
This is an MCP (Model Context Protocol) server that integrates Crawl4AI, Qdrant vector database, and Neo4j knowledge graph to provide AI agents with advanced web crawling, GitHub repository indexing, vector search, and AI hallucination detection capabilities.

## Core Functionality
- **Web Crawling**: Intelligent crawling with sitemap detection and recursive crawling
- **GitHub Integration**: Clone and index GitHub repositories for documentation
- **RAG Search**: Semantic vector search with multiple strategies (hybrid, reranking, contextual)
- **Knowledge Graph**: Code structure analysis and AI hallucination detection
- **Multi-language Support**: Tree-sitter parsers for 10+ programming languages

## Key Components
- **MCP Server**: FastMCP-based async server with context management
- **Tools**: Organized MCP tools (web, github, rag, knowledge graph)
- **Services**: Core business logic for indexing and search
- **Clients**: Database integrations (Qdrant, Neo4j, OpenAI)
- **Knowledge Graphs**: Multi-language code analysis system

## Target Use Cases
- Documentation RAG for AI coding assistants
- GitHub repository analysis and search
- AI-generated code validation and hallucination detection
- Advanced semantic search with contextual embeddings