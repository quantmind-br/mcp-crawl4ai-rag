"""
Knowledge Graph MCP Tools.

This module contains MCP tools for interacting with Neo4j knowledge graphs,
including repository parsing, AI hallucination detection, and graph querying.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
from mcp.server.fastmcp import Context

# Add knowledge_graphs folder to path for importing knowledge graph modules
project_root = Path(__file__).resolve().parent.parent.parent
knowledge_graphs_path = project_root / "knowledge_graphs"
if str(knowledge_graphs_path) not in sys.path:
    sys.path.append(str(knowledge_graphs_path))

from ai_script_analyzer import AIScriptAnalyzer
from hallucination_reporter import HallucinationReporter

import logging

logger = logging.getLogger(__name__)


def validate_script_path(script_path: str) -> Dict[str, Any]:
    """Validate script path and return error info if invalid."""
    if not script_path or not isinstance(script_path, str):
        return {"valid": False, "error": "Script path is required"}

    if not os.path.exists(script_path):
        return {"valid": False, "error": f"Script not found: {script_path}"}

    if not script_path.endswith(".py"):
        return {"valid": False, "error": "Only Python (.py) files are supported"}

    try:
        # Check if file is readable
        with open(script_path, "r", encoding="utf-8") as f:
            f.read(1)  # Read first character to test
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Cannot read script file: {str(e)}"}


def validate_github_url(repo_url: str) -> Dict[str, Any]:
    """Validate GitHub repository URL."""
    if not repo_url or not isinstance(repo_url, str):
        return {"valid": False, "error": "Repository URL is required"}

    repo_url = repo_url.strip()

    # Basic GitHub URL validation
    if not ("github.com" in repo_url.lower() or repo_url.endswith(".git")):
        return {"valid": False, "error": "Please provide a valid GitHub repository URL"}

    # Check URL format
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        return {
            "valid": False,
            "error": "Repository URL must start with https:// or git@",
        }

    return {"valid": True, "repo_name": repo_url.split("/")[-1].replace(".git", "")}


async def _handle_repos_command(session, command: str) -> str:
    """Handle 'repos' command - list all repositories"""
    query = "MATCH (r:Repository) RETURN r.name as name ORDER BY r.name"
    result = await session.run(query)

    repos = []
    async for record in result:
        repos.append(record["name"])

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {"repositories": repos},
            "metadata": {"total_results": len(repos), "limited": False},
        },
        indent=2,
    )


async def _handle_explore_command(session, command: str, repo_name: str) -> str:
    """Handle 'explore <repo>' command - get repository overview"""
    # Check if repository exists
    repo_check_query = "MATCH (r:Repository {name: $repo_name}) RETURN r.name as name"
    result = await session.run(repo_check_query, repo_name=repo_name)
    repo_record = await result.single()

    if not repo_record:
        return json.dumps(
            {
                "success": False,
                "command": command,
                "error": f"Repository '{repo_name}' not found in knowledge graph",
            },
            indent=2,
        )

    # Get file count
    files_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
    RETURN count(f) as file_count
    """
    result = await session.run(files_query, repo_name=repo_name)
    file_count = (await result.single())["file_count"]

    # Get class count
    classes_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
    RETURN count(DISTINCT c) as class_count
    """
    result = await session.run(classes_query, repo_name=repo_name)
    class_count = (await result.single())["class_count"]

    # Get function count
    functions_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
    RETURN count(DISTINCT func) as function_count
    """
    result = await session.run(functions_query, repo_name=repo_name)
    function_count = (await result.single())["function_count"]

    # Get method count
    methods_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
    RETURN count(DISTINCT m) as method_count
    """
    result = await session.run(methods_query, repo_name=repo_name)
    method_count = (await result.single())["method_count"]

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {
                "repository": repo_name,
                "statistics": {
                    "files": file_count,
                    "classes": class_count,
                    "functions": function_count,
                    "methods": method_count,
                },
            },
            "metadata": {"total_results": 1, "limited": False},
        },
        indent=2,
    )


async def _handle_classes_command(session, command: str, repo_name: str = None) -> str:
    """Handle 'classes [repo]' command - list classes"""
    limit = 20

    if repo_name:
        query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, repo_name=repo_name, limit=limit)
    else:
        query = """
        MATCH (c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, limit=limit)

    classes = []
    async for record in result:
        classes.append({"name": record["name"], "full_name": record["full_name"]})

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {"classes": classes, "repository_filter": repo_name},
            "metadata": {
                "total_results": len(classes),
                "limited": len(classes) >= limit,
            },
        },
        indent=2,
    )


async def _handle_class_command(session, command: str, class_name: str) -> str:
    """Handle 'class <name>' command - explore specific class"""
    # Find the class
    class_query = """
    MATCH (c:Class)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN c.name as name, c.full_name as full_name
    LIMIT 1
    """
    result = await session.run(class_query, class_name=class_name)
    class_record = await result.single()

    if not class_record:
        return json.dumps(
            {
                "success": False,
                "command": command,
                "error": f"Class '{class_name}' not found in knowledge graph",
            },
            indent=2,
        )

    actual_name = class_record["name"]
    full_name = class_record["full_name"]

    # Get methods
    methods_query = """
    MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed, m.return_type as return_type
    ORDER BY m.name
    """
    result = await session.run(methods_query, class_name=class_name)

    methods = []
    async for record in result:
        # Use detailed params if available, fall back to simple params
        params_to_use = record["params_detailed"] or record["params_list"] or []
        methods.append(
            {
                "name": record["name"],
                "parameters": params_to_use,
                "return_type": record["return_type"] or "Any",
            }
        )

    # Get attributes
    attributes_query = """
    MATCH (c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN a.name as name, a.type as type
    ORDER BY a.name
    """
    result = await session.run(attributes_query, class_name=class_name)

    attributes = []
    async for record in result:
        attributes.append({"name": record["name"], "type": record["type"] or "Any"})

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {
                "class": {
                    "name": actual_name,
                    "full_name": full_name,
                    "methods": methods,
                    "attributes": attributes,
                }
            },
            "metadata": {
                "total_results": 1,
                "methods_count": len(methods),
                "attributes_count": len(attributes),
                "limited": False,
            },
        },
        indent=2,
    )


async def _handle_method_command(
    session, command: str, method_name: str, class_name: str = None
) -> str:
    """Handle 'method <name> [class]' command - search for methods"""
    if class_name:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE (c.name = $class_name OR c.full_name = $class_name)
          AND m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list, 
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        """
        result = await session.run(
            query, class_name=class_name, method_name=method_name
        )
    else:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list, 
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        ORDER BY c.name
        LIMIT 20
        """
        result = await session.run(query, method_name=method_name)

    methods = []
    async for record in result:
        # Use detailed params if available, fall back to simple params
        params_to_use = record["params_detailed"] or record["params_list"] or []
        methods.append(
            {
                "class_name": record["class_name"],
                "class_full_name": record["class_full_name"],
                "method_name": record["method_name"],
                "parameters": params_to_use,
                "return_type": record["return_type"] or "Any",
                "legacy_args": record["args"] or [],
            }
        )

    if not methods:
        return json.dumps(
            {
                "success": False,
                "command": command,
                "error": f"Method '{method_name}'"
                + (f" in class '{class_name}'" if class_name else "")
                + " not found",
            },
            indent=2,
        )

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {"methods": methods, "class_filter": class_name},
            "metadata": {
                "total_results": len(methods),
                "limited": len(methods) >= 20 and not class_name,
            },
        },
        indent=2,
    )


async def _handle_query_command(session, command: str, cypher_query: str) -> str:
    """Handle 'query <cypher>' command - execute custom Cypher query"""
    try:
        # Execute the query with a limit to prevent overwhelming responses
        result = await session.run(cypher_query)

        records = []
        count = 0
        async for record in result:
            records.append(dict(record))
            count += 1
            if count >= 20:  # Limit results to prevent overwhelming responses
                break

        return json.dumps(
            {
                "success": True,
                "command": command,
                "data": {"query": cypher_query, "results": records},
                "metadata": {
                    "total_results": len(records),
                    "limited": len(records) >= 20,
                },
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "command": command,
                "error": f"Cypher query error: {str(e)}",
                "data": {"query": cypher_query},
            },
            indent=2,
        )


# Import MCP decorator - this module will be imported by the main MCP server
# The tools will be registered by the core app module
def register_knowledge_graph_tools(mcp_instance):
    """Register knowledge graph tools with the MCP instance."""
    # This function will be called by the core app module to register tools
    mcp_instance.tool()(parse_github_repository)
    mcp_instance.tool()(check_ai_script_hallucinations)
    mcp_instance.tool()(query_knowledge_graph)


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
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps(
                {
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment.",
                },
                indent=2,
            )

        # Get the repository extractor from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor

        if not repo_extractor:
            return json.dumps(
                {
                    "success": False,
                    "error": "Repository extractor not available. Check Neo4j configuration in environment variables.",
                },
                indent=2,
            )

        # Validate repository URL
        validation = validate_github_url(repo_url)
        if not validation["valid"]:
            return json.dumps(
                {"success": False, "repo_url": repo_url, "error": validation["error"]},
                indent=2,
            )

        repo_name = validation["repo_name"]

        # Parse the repository (this includes cloning, analysis, and Neo4j storage)
        logger.info(f"Starting repository analysis for: {repo_name}")
        await repo_extractor.analyze_repository(repo_url)
        logger.info(f"Repository analysis completed for: {repo_name}")

        # Query Neo4j for statistics about the parsed repository
        async with repo_extractor.driver.session() as session:
            # Get comprehensive repository statistics
            stats_query = """
            MATCH (r:Repository {name: $repo_name})
            OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
            OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:Attribute)
            WITH r, 
                 count(DISTINCT f) as files_count,
                 count(DISTINCT c) as classes_count,
                 count(DISTINCT m) as methods_count,
                 count(DISTINCT func) as functions_count,
                 count(DISTINCT a) as attributes_count
            
            // Get some sample module names
            OPTIONAL MATCH (r)-[:CONTAINS]->(sample_f:File)
            WITH r, files_count, classes_count, methods_count, functions_count, attributes_count,
                 collect(DISTINCT sample_f.module_name)[0..5] as sample_modules
            
            RETURN 
                r.name as repo_name,
                files_count,
                classes_count, 
                methods_count,
                functions_count,
                attributes_count,
                sample_modules
            """

            result = await session.run(stats_query, repo_name=repo_name)
            record = await result.single()

            if record:
                stats = {
                    "repository": record["repo_name"],
                    "files_processed": record["files_count"],
                    "classes_created": record["classes_count"],
                    "methods_created": record["methods_count"],
                    "functions_created": record["functions_count"],
                    "attributes_created": record["attributes_count"],
                    "sample_modules": record["sample_modules"] or [],
                }
            else:
                return json.dumps(
                    {
                        "success": False,
                        "repo_url": repo_url,
                        "error": f"Repository '{repo_name}' not found in database after parsing",
                    },
                    indent=2,
                )

        return json.dumps(
            {
                "success": True,
                "repo_url": repo_url,
                "repo_name": repo_name,
                "message": f"Successfully parsed repository '{repo_name}' into knowledge graph",
                "statistics": stats,
                "ready_for_validation": True,
                "next_steps": [
                    "Repository is now available for hallucination detection",
                    f"Use check_ai_script_hallucinations to validate scripts against {repo_name}",
                    "The knowledge graph contains classes, methods, and functions from this repository",
                ],
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "repo_url": repo_url,
                "error": f"Repository parsing failed: {str(e)}",
            },
            indent=2,
        )


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
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps(
                {
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment.",
                },
                indent=2,
            )

        # Get the knowledge validator from context
        knowledge_validator = ctx.request_context.lifespan_context.knowledge_validator

        if not knowledge_validator:
            return json.dumps(
                {
                    "success": False,
                    "error": "Knowledge graph validator not available. Check Neo4j configuration in environment variables.",
                },
                indent=2,
            )

        # Validate script path
        validation = validate_script_path(script_path)
        if not validation["valid"]:
            return json.dumps(
                {
                    "success": False,
                    "script_path": script_path,
                    "error": validation["error"],
                },
                indent=2,
            )

        # Step 1: Analyze script structure using AST
        analyzer = AIScriptAnalyzer()
        analysis_result = analyzer.analyze_script(script_path)

        if analysis_result.errors:
            logger.warning(
                f"Analysis warnings for {script_path}: {analysis_result.errors}"
            )

        # Step 2: Validate against knowledge graph
        validation_result = await knowledge_validator.validate_script(analysis_result)

        # Step 3: Generate comprehensive report
        reporter = HallucinationReporter()
        report = reporter.generate_comprehensive_report(validation_result)

        # Format response with comprehensive information
        return json.dumps(
            {
                "success": True,
                "script_path": script_path,
                "overall_confidence": validation_result.overall_confidence,
                "validation_summary": {
                    "total_validations": report["validation_summary"][
                        "total_validations"
                    ],
                    "valid_count": report["validation_summary"]["valid_count"],
                    "invalid_count": report["validation_summary"]["invalid_count"],
                    "uncertain_count": report["validation_summary"]["uncertain_count"],
                    "not_found_count": report["validation_summary"]["not_found_count"],
                    "hallucination_rate": report["validation_summary"][
                        "hallucination_rate"
                    ],
                },
                "hallucinations_detected": report["hallucinations_detected"],
                "recommendations": report["recommendations"],
                "analysis_metadata": {
                    "total_imports": report["analysis_metadata"]["total_imports"],
                    "total_classes": report["analysis_metadata"]["total_classes"],
                    "total_methods": report["analysis_metadata"]["total_methods"],
                    "total_attributes": report["analysis_metadata"]["total_attributes"],
                    "total_functions": report["analysis_metadata"]["total_functions"],
                },
                "libraries_analyzed": report.get("libraries_analyzed", []),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "script_path": script_path,
                "error": f"Analysis failed: {str(e)}",
            },
            indent=2,
        )


async def query_knowledge_graph(ctx: Context, command: str) -> str:
    """
    Query and explore the Neo4j knowledge graph containing repository data.

    This tool provides comprehensive access to the knowledge graph for exploring repositories,
    classes, methods, functions, and their relationships. Perfect for understanding what data
    is available for hallucination detection and debugging validation results.

    ⚠️ IMPORTANT: Always start with the `repos` command first!
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
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps(
                {
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment.",
                },
                indent=2,
            )

        # Get Neo4j driver from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        if not repo_extractor or not repo_extractor.driver:
            return json.dumps(
                {
                    "success": False,
                    "error": "Neo4j connection not available. Check Neo4j configuration in environment variables.",
                },
                indent=2,
            )

        # Parse command
        command = command.strip()
        if not command:
            return json.dumps(
                {
                    "success": False,
                    "command": "",
                    "error": "Command cannot be empty. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>",
                },
                indent=2,
            )

        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        async with repo_extractor.driver.session() as session:
            # Route to appropriate handler
            if cmd == "repos":
                return await _handle_repos_command(session, command)
            elif cmd == "explore":
                if not args:
                    return json.dumps(
                        {
                            "success": False,
                            "command": command,
                            "error": "Repository name required. Usage: explore <repo_name>",
                        },
                        indent=2,
                    )
                return await _handle_explore_command(session, command, args[0])
            elif cmd == "classes":
                repo_name = args[0] if args else None
                return await _handle_classes_command(session, command, repo_name)
            elif cmd == "class":
                if not args:
                    return json.dumps(
                        {
                            "success": False,
                            "command": command,
                            "error": "Class name required. Usage: class <class_name>",
                        },
                        indent=2,
                    )
                return await _handle_class_command(session, command, args[0])
            elif cmd == "method":
                if not args:
                    return json.dumps(
                        {
                            "success": False,
                            "command": command,
                            "error": "Method name required. Usage: method <method_name> [class_name]",
                        },
                        indent=2,
                    )
                method_name = args[0]
                class_name = args[1] if len(args) > 1 else None
                return await _handle_method_command(
                    session, command, method_name, class_name
                )
            elif cmd == "query":
                if not args:
                    return json.dumps(
                        {
                            "success": False,
                            "command": command,
                            "error": "Cypher query required. Usage: query <cypher_query>",
                        },
                        indent=2,
                    )
                cypher_query = " ".join(args)
                return await _handle_query_command(session, command, cypher_query)
            else:
                return json.dumps(
                    {
                        "success": False,
                        "command": command,
                        "error": f"Unknown command '{cmd}'. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>",
                    },
                    indent=2,
                )

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "command": command,
                "error": f"Query execution failed: {str(e)}",
            },
            indent=2,
        )