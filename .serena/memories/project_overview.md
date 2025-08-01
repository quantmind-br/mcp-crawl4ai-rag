# Project Overview

## Purpose
The **Crawl4AI RAG MCP Server** is a powerful implementation of the Model Context Protocol (MCP) that integrates Crawl4AI web crawling capabilities with Supabase vector database for RAG (Retrieval-Augmented Generation). It enables AI agents and AI coding assistants to:

- Crawl websites and store content in vector databases
- Perform semantic search over crawled content
- Extract and search code examples from documentation
- Detect AI hallucinations using Neo4j knowledge graphs
- Analyze GitHub repositories for code structure

## Primary Goals
- Integrate into [Archon](https://github.com/coleam00/Archon) as a knowledge engine
- Support multiple embedding models (currently OpenAI-based)
- Implement advanced RAG strategies beyond basic lookups
- Enable local deployment with Ollama support

## Key Features
- **Smart URL Detection**: Handles webpages, sitemaps, text files
- **Recursive Crawling**: Follows internal links
- **Parallel Processing**: Efficient multi-page crawling
- **Advanced RAG Strategies**: Contextual embeddings, hybrid search, reranking
- **Knowledge Graph Integration**: AI hallucination detection via Neo4j
- **Code Example Extraction**: Specialized code search for AI coding assistants

## Target Use Cases
- AI coding assistants that need to reference documentation
- RAG systems requiring high-quality content retrieval
- Knowledge engines for AI agent development
- Code validation and hallucination detection systems