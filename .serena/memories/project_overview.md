# MCP Crawl4AI RAG Server - Project Overview

## Purpose
Advanced Model Context Protocol (MCP) server that integrates Crawl4AI, Qdrant vector database, and Neo4j knowledge graph to provide AI agents and coding assistants with:
- Intelligent web crawling capabilities
- GitHub repository indexing and analysis
- Vector search and retrieval (RAG)
- AI hallucination detection through knowledge graphs

## Core Architecture
- **FastMCP-based server** with async context management
- **Dual-storage system**: Qdrant for vector embeddings + Neo4j for code structure
- **Multi-language code parsing** using Tree-sitter grammars
- **Unified indexing service** for cross-system file linking
- **MCP tools** organized by functionality (web, GitHub, RAG, knowledge graph)

## Key Features
- Smart web crawling with sitemap detection and recursive link following
- GitHub repository cloning and comprehensive indexing
- Contextual embeddings and hybrid search (semantic + keyword)
- Code structure analysis for 10+ programming languages
- AI-generated code validation against knowledge graph
- Performance optimization for enterprise-scale processing

## Target Users
- AI agents requiring web data and repository knowledge
- Coding assistants needing accurate code context
- Development teams implementing RAG workflows
- Enterprise systems requiring hallucination detection

## Technology Focus
Production-ready MCP server with emphasis on accuracy, performance, and scalability for AI-powered development workflows.