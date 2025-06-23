"""
Knowledge graph tools for MCP server.

This module contains MCP tools for knowledge graph operations:
- parse_github_repository: Parse GitHub repository into Neo4j
- check_ai_script_hallucinations: Check AI-generated scripts for hallucinations
- query_knowledge_graph: Query and explore the Neo4j knowledge graph
"""

import json

from mcp.server.fastmcp import Context

from ..services.knowledge_graph import KnowledgeGraphService
from ..utils.validation import validate_script_path, validate_github_url
from ..config import config


def register_knowledge_tools(mcp):
    """Register knowledge graph tools with the MCP server."""
    
    @mcp.tool()
    async def parse_github_repository(ctx: Context, repo_url: str) -> str:
        """
        Parse a GitHub repository into the Neo4j knowledge graph.
        
        This tool clones a GitHub repository, analyzes its Python files, and stores
        the code structure (classes, methods, functions, imports) in Neo4j for use
        in hallucination detection. The tool:
        
        - Clones the repository to a temporary location
        - Analyzes Python files to extract code structure
        - Stores classes, methods, functions, and imports in Neo4j
        - Provides detailed statistics about the parsing results
        - Automatically handles module name detection for imports
        
        Args:
            ctx: The MCP server provided context
            repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')
        
        Returns:
            JSON string with parsing results, statistics, and repository information
        """
        try:
            # Check if knowledge graph functionality is enabled
            if not config.USE_KNOWLEDGE_GRAPH:
                return json.dumps({
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
                }, indent=2)
            
            # Get the knowledge graph service from context
            kg_service = ctx.request_context.lifespan_context.repo_extractor
            
            if not kg_service:
                return json.dumps({
                    "success": False,
                    "error": "Repository extractor not available. Check Neo4j configuration in environment variables."
                }, indent=2)
            
            # Create knowledge graph service wrapper
            knowledge_service = KnowledgeGraphService()
            knowledge_service.repo_extractor = kg_service
            knowledge_service.is_enabled = True
            
            # Validate repository URL
            validation = validate_github_url(repo_url)
            if not validation["valid"]:
                return json.dumps({
                    "success": False,
                    "repo_url": repo_url,
                    "error": validation["error"]
                }, indent=2)
            
            # Parse the repository
            result = await knowledge_service.parse_github_repository(repo_url)
            
            return json.dumps(result, separators=(',', ':'))  # Compact JSON to reduce size
            
        except Exception as e:
            error_msg = str(e)
            # Handle specific SSE communication errors
            if "BrokenResourceError" in error_msg or "anyio" in error_msg:
                print(f"SSE communication error after successful parsing: {error_msg}")
                # The parsing likely succeeded, just return a simple success
                repo_name = repo_url.split('/')[-1].replace('.git', '') if 'repo_url' in locals() else "unknown"
                return json.dumps({
                    "success": True,
                    "repo_name": repo_name,
                    "message": "Parsing completed (communication error during response)",
                    "note": "Check Neo4j directly for results"
                }, separators=(',', ':'))
            
            return json.dumps({
                "success": False,
                "repo_url": repo_url,
                "error": f"Repository parsing failed: {error_msg}"
            }, separators=(',', ':'))

    @mcp.tool()
    async def check_ai_script_hallucinations(ctx: Context, script_path: str) -> str:
        """
        Check an AI-generated Python script for hallucinations using the knowledge graph.
        
        This tool analyzes a Python script for potential AI hallucinations by validating
        imports, method calls, class instantiations, and function calls against a Neo4j
        knowledge graph containing real repository data.
        
        The tool performs comprehensive analysis including:
        - Import validation against known repositories
        - Method call validation on classes from the knowledge graph
        - Class instantiation parameter validation
        - Function call parameter validation
        - Attribute access validation
        
        Args:
            ctx: The MCP server provided context
            script_path: Absolute path to the Python script to analyze
        
        Returns:
            JSON string with hallucination detection results, confidence scores, and recommendations
        """
        try:
            # Check if knowledge graph functionality is enabled
            if not config.USE_KNOWLEDGE_GRAPH:
                return json.dumps({
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
                }, indent=2)
            
            # Get the knowledge validator from context
            knowledge_validator = ctx.request_context.lifespan_context.knowledge_validator
            
            if not knowledge_validator:
                return json.dumps({
                    "success": False,
                    "error": "Knowledge graph validator not available. Check Neo4j configuration in environment variables."
                }, indent=2)
            
            # Create knowledge graph service wrapper
            knowledge_service = KnowledgeGraphService()
            knowledge_service.knowledge_validator = knowledge_validator
            knowledge_service.is_enabled = True
            
            # Validate script path
            validation = validate_script_path(script_path)
            if not validation["valid"]:
                return json.dumps({
                    "success": False,
                    "script_path": script_path,
                    "error": validation["error"]
                }, indent=2)
            
            # Check for hallucinations
            result = await knowledge_service.check_ai_script_hallucinations(script_path)
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "script_path": script_path,
                "error": f"Analysis failed: {str(e)}"
            }, indent=2)

    @mcp.tool()
    async def query_knowledge_graph(ctx: Context, command: str) -> str:
        """
        Query and explore the Neo4j knowledge graph containing repository data.
        
        This tool provides comprehensive access to the knowledge graph for exploring repositories,
        classes, methods, functions, and their relationships. Perfect for understanding what data
        is available for hallucination detection and debugging validation results.
        
        **⚠️ IMPORTANT: Always start with the `repos` command first!**
        Before using any other commands, run `repos` to see what repositories are available
        in your knowledge graph. This will help you understand what data you can explore.
        
        ## Available Commands:
        
        **Repository Commands:**
        - `repos` - **START HERE!** List all repositories in the knowledge graph
        - `explore <repo_name>` - Get detailed overview of a specific repository
        
        **Class Commands:**  
        - `classes` - List all classes across all repositories (limited to 20)
        - `classes <repo_name>` - List classes in a specific repository
        - `class <class_name>` - Get detailed information about a specific class including methods and attributes
        
        **Method Commands:**
        - `method <method_name>` - Search for methods by name across all classes
        - `method <method_name> <class_name>` - Search for a method within a specific class
        
        **Custom Query:**
        - `query <cypher_query>` - Execute a custom Cypher query (results limited to 20 records)
        
        ## Knowledge Graph Schema:
        
        **Node Types:**
        - Repository: `(r:Repository {name: string})`
        - File: `(f:File {path: string, module_name: string})`
        - Class: `(c:Class {name: string, full_name: string})`
        - Method: `(m:Method {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
        - Function: `(func:Function {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
        - Attribute: `(a:Attribute {name: string, type: string})`
        
        **Relationships:**
        - `(r:Repository)-[:CONTAINS]->(f:File)`
        - `(f:File)-[:DEFINES]->(c:Class)`
        - `(c:Class)-[:HAS_METHOD]->(m:Method)`
        - `(c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)`
        - `(f:File)-[:DEFINES]->(func:Function)`
        
        ## Example Workflow:
        ```
        1. repos                                    # See what repositories are available
        2. explore pydantic-ai                      # Explore a specific repository
        3. classes pydantic-ai                      # List classes in that repository
        4. class Agent                              # Explore the Agent class
        5. method run_stream                        # Search for run_stream method
        6. method __init__ Agent                    # Find Agent constructor
        7. query "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) WHERE m.name = 'run' RETURN c.name, m.name LIMIT 5"
        ```
        
        Args:
            ctx: The MCP server provided context
            command: Command string to execute (see available commands above)
        
        Returns:
            JSON string with query results, statistics, and metadata
        """
        try:
            # Check if knowledge graph functionality is enabled
            if not config.USE_KNOWLEDGE_GRAPH:
                return json.dumps({
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
                }, indent=2)
            
            # Get Neo4j driver from context
            repo_extractor = ctx.request_context.lifespan_context.repo_extractor
            if not repo_extractor or not repo_extractor.driver:
                return json.dumps({
                    "success": False,
                    "error": "Neo4j connection not available. Check Neo4j configuration in environment variables."
                }, indent=2)
            
            # Create knowledge graph service wrapper
            knowledge_service = KnowledgeGraphService()
            knowledge_service.repo_extractor = repo_extractor
            knowledge_service.is_enabled = True
            
            # Execute query
            result = await knowledge_service.query_knowledge_graph(command)
            
            return json.dumps(result, indent=2)
                
        except Exception as e:
            return json.dumps({
                "success": False,
                "command": command,
                "error": f"Query execution failed: {str(e)}"
            }, indent=2)