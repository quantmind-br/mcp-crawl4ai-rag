"""
Knowledge graph service.

This module provides services for knowledge graph operations including
Neo4j connectivity, repository analysis, and AI hallucination detection.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

from ..config import config
from ..utils.validation import validate_neo4j_connection, format_neo4j_error

# Add knowledge_graphs folder to path for importing knowledge graph modules
knowledge_graphs_path = Path(__file__).resolve().parent.parent.parent / 'knowledge_graphs'
sys.path.append(str(knowledge_graphs_path))


class KnowledgeGraphService:
    """Service for knowledge graph operations."""
    
    def __init__(self):
        self.knowledge_validator = None
        self.repo_extractor = None
        self.is_enabled = config.USE_KNOWLEDGE_GRAPH
    
    async def initialize(self):
        """Initialize knowledge graph components if enabled."""
        if not self.is_enabled:
            print("Knowledge graph functionality disabled - set USE_KNOWLEDGE_GRAPH=true to enable")
            return
        
        if not validate_neo4j_connection():
            print("Neo4j credentials not configured - knowledge graph tools will be unavailable")
            return
        
        neo4j_uri = config.NEO4J_URI
        neo4j_user = config.NEO4J_USER
        neo4j_password = config.NEO4J_PASSWORD
        
        try:
            print("Initializing knowledge graph components...")
            
            # Import knowledge graph modules
            from knowledge_graph_validator import KnowledgeGraphValidator
            from parse_repo_into_neo4j import DirectNeo4jExtractor
            
            # Initialize knowledge graph validator
            self.knowledge_validator = KnowledgeGraphValidator(neo4j_uri, neo4j_user, neo4j_password)
            await self.knowledge_validator.initialize()
            print("✓ Knowledge graph validator initialized")
            
            # Initialize repository extractor
            self.repo_extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)
            await self.repo_extractor.initialize()
            print("✓ Repository extractor initialized")
            
        except Exception as e:
            print(f"Failed to initialize Neo4j components: {format_neo4j_error(e)}")
            self.knowledge_validator = None
            self.repo_extractor = None
    
    async def close(self):
        """Close knowledge graph components."""
        if self.knowledge_validator:
            try:
                await self.knowledge_validator.close()
                print("✓ Knowledge graph validator closed")
            except Exception as e:
                print(f"Error closing knowledge validator: {e}")
        if self.repo_extractor:
            try:
                await self.repo_extractor.close()
                print("✓ Repository extractor closed")
            except Exception as e:
                print(f"Error closing repository extractor: {e}")
    
    def is_available(self) -> bool:
        """Check if knowledge graph functionality is available."""
        return self.is_enabled and self.knowledge_validator is not None and self.repo_extractor is not None
    
    async def parse_github_repository(self, repo_url: str) -> Dict[str, Any]:
        """
        Parse a GitHub repository into the Neo4j knowledge graph.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Dictionary with parsing results and statistics
        """
        if not self.is_available():
            raise ValueError("Knowledge graph functionality not available")
        
        # Parse the repository (this includes cloning, analysis, and Neo4j storage)
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        print(f"Starting repository analysis for: {repo_name}")
        
        try:
            # Add timeout to prevent hanging
            await asyncio.wait_for(
                self.repo_extractor.analyze_repository(repo_url),
                timeout=config.REPO_ANALYSIS_TIMEOUT
            )
            print(f"Repository analysis completed for: {repo_name}")
        except asyncio.TimeoutError:
            print(f"Repository analysis timed out for: {repo_name}")
            return {
                "success": False,
                "repo_name": repo_name,
                "error": "Repository analysis timed out (30 min limit)"
            }
        
        # Query Neo4j for statistics about the parsed repository
        async with self.repo_extractor.driver.session() as session:
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
                    "repository": record['repo_name'],
                    "files_processed": record['files_count'],
                    "classes_created": record['classes_count'],
                    "methods_created": record['methods_count'], 
                    "functions_created": record['functions_count'],
                    "attributes_created": record['attributes_count'],
                    "sample_modules": record['sample_modules'] or []
                }
            else:
                return {
                    "success": False,
                    "repo_url": repo_url,
                    "error": f"Repository '{repo_name}' not found in database after parsing"
                }
        
        # Create a compact response to avoid SSE communication issues
        compact_stats = {
            "files_processed": stats.get("files_count", 0),
            "classes_created": stats.get("classes_count", 0),
            "methods_created": stats.get("methods_count", 0),
            "functions_created": stats.get("functions_count", 0),
            "total_nodes": (stats.get("classes_count", 0) + 
                          stats.get("methods_count", 0) + 
                          stats.get("functions_count", 0))
        }
        
        return {
            "success": True,
            "repo_name": repo_name,
            "message": f"Successfully parsed {repo_name}",
            "stats": compact_stats,
            "ready": True
        }
    
    async def check_ai_script_hallucinations(self, script_path: str) -> Dict[str, Any]:
        """
        Check an AI-generated Python script for hallucinations using the knowledge graph.
        
        Args:
            script_path: Absolute path to the Python script to analyze
            
        Returns:
            Dictionary with hallucination detection results
        """
        if not self.is_available():
            raise ValueError("Knowledge graph functionality not available")
        
        # Import analysis modules
        from ai_script_analyzer import AIScriptAnalyzer
        from hallucination_reporter import HallucinationReporter
        
        # Step 1: Analyze script structure using AST
        analyzer = AIScriptAnalyzer()
        analysis_result = analyzer.analyze_script(script_path)
        
        if analysis_result.errors:
            print(f"Analysis warnings for {script_path}: {analysis_result.errors}")
        
        # Step 2: Validate against knowledge graph
        validation_result = await self.knowledge_validator.validate_script(analysis_result)
        
        # Step 3: Generate comprehensive report
        reporter = HallucinationReporter()
        report = reporter.generate_comprehensive_report(validation_result)
        
        # Format response with comprehensive information
        return {
            "success": True,
            "script_path": script_path,
            "overall_confidence": validation_result.overall_confidence,
            "validation_summary": {
                "total_validations": report["validation_summary"]["total_validations"],
                "valid_count": report["validation_summary"]["valid_count"],
                "invalid_count": report["validation_summary"]["invalid_count"],
                "uncertain_count": report["validation_summary"]["uncertain_count"],
                "not_found_count": report["validation_summary"]["not_found_count"],
                "hallucination_rate": report["validation_summary"]["hallucination_rate"]
            },
            "hallucinations_detected": report["hallucinations_detected"],
            "recommendations": report["recommendations"],
            "analysis_metadata": {
                "total_imports": report["analysis_metadata"]["total_imports"],
                "total_classes": report["analysis_metadata"]["total_classes"],
                "total_methods": report["analysis_metadata"]["total_methods"],
                "total_attributes": report["analysis_metadata"]["total_attributes"],
                "total_functions": report["analysis_metadata"]["total_functions"]
            },
            "libraries_analyzed": report.get("libraries_analyzed", [])
        }
    
    async def query_knowledge_graph(self, command: str) -> Dict[str, Any]:
        """
        Query and explore the Neo4j knowledge graph containing repository data.
        
        Args:
            command: Command string to execute
            
        Returns:
            Dictionary with query results
        """
        if not self.is_available():
            raise ValueError("Knowledge graph functionality not available")
        
        # Parse command
        command = command.strip()
        if not command:
            return {
                "success": False,
                "command": "",
                "error": "Command cannot be empty. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>"
            }
        
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        async with self.repo_extractor.driver.session() as session:
            # Route to appropriate handler
            if cmd == "repos":
                return await self._handle_repos_command(session, command)
            elif cmd == "explore":
                if not args:
                    return {
                        "success": False,
                        "command": command,
                        "error": "Repository name required. Usage: explore <repo_name>"
                    }
                return await self._handle_explore_command(session, command, args[0])
            elif cmd == "classes":
                repo_name = args[0] if args else None
                return await self._handle_classes_command(session, command, repo_name)
            elif cmd == "class":
                if not args:
                    return {
                        "success": False,
                        "command": command,
                        "error": "Class name required. Usage: class <class_name>"
                    }
                return await self._handle_class_command(session, command, args[0])
            elif cmd == "method":
                if not args:
                    return {
                        "success": False,
                        "command": command,
                        "error": "Method name required. Usage: method <method_name> [class_name]"
                    }
                method_name = args[0]
                class_name = args[1] if len(args) > 1 else None
                return await self._handle_method_command(session, command, method_name, class_name)
            elif cmd == "query":
                if not args:
                    return {
                        "success": False,
                        "command": command,
                        "error": "Cypher query required. Usage: query <cypher_query>"
                    }
                cypher_query = " ".join(args)
                return await self._handle_query_command(session, command, cypher_query)
            else:
                return {
                    "success": False,
                    "command": command,
                    "error": f"Unknown command '{cmd}'. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>"
                }
    
    async def _handle_repos_command(self, session, command: str) -> Dict[str, Any]:
        """Handle 'repos' command - list all repositories"""
        query = "MATCH (r:Repository) RETURN r.name as name ORDER BY r.name"
        result = await session.run(query)
        
        repos = []
        async for record in result:
            repos.append(record['name'])
        
        return {
            "success": True,
            "command": command,
            "data": {
                "repositories": repos
            },
            "metadata": {
                "total_results": len(repos),
                "limited": False
            }
        }
    
    async def _handle_explore_command(self, session, command: str, repo_name: str) -> Dict[str, Any]:
        """Handle 'explore <repo>' command - get repository overview"""
        # Check if repository exists
        repo_check_query = "MATCH (r:Repository {name: $repo_name}) RETURN r.name as name"
        result = await session.run(repo_check_query, repo_name=repo_name)
        repo_record = await result.single()
        
        if not repo_record:
            return {
                "success": False,
                "command": command,
                "error": f"Repository '{repo_name}' not found in knowledge graph"
            }
        
        # Get file count
        files_query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
        RETURN count(f) as file_count
        """
        result = await session.run(files_query, repo_name=repo_name)
        file_count = (await result.single())['file_count']
        
        # Get class count
        classes_query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
        RETURN count(DISTINCT c) as class_count
        """
        result = await session.run(classes_query, repo_name=repo_name)
        class_count = (await result.single())['class_count']
        
        # Get function count
        functions_query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
        RETURN count(DISTINCT func) as function_count
        """
        result = await session.run(functions_query, repo_name=repo_name)
        function_count = (await result.single())['function_count']
        
        # Get method count
        methods_query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
        RETURN count(DISTINCT m) as method_count
        """
        result = await session.run(methods_query, repo_name=repo_name)
        method_count = (await result.single())['method_count']
        
        return {
            "success": True,
            "command": command,
            "data": {
                "repository": repo_name,
                "statistics": {
                    "files": file_count,
                    "classes": class_count,
                    "functions": function_count,
                    "methods": method_count
                }
            },
            "metadata": {
                "total_results": 1,
                "limited": False
            }
        }
    
    async def _handle_classes_command(self, session, command: str, repo_name: Optional[str] = None) -> Dict[str, Any]:
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
            classes.append({
                'name': record['name'],
                'full_name': record['full_name']
            })
        
        return {
            "success": True,
            "command": command,
            "data": {
                "classes": classes,
                "repository_filter": repo_name
            },
            "metadata": {
                "total_results": len(classes),
                "limited": len(classes) >= limit
            }
        }
    
    async def _handle_class_command(self, session, command: str, class_name: str) -> Dict[str, Any]:
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
            return {
                "success": False,
                "command": command,
                "error": f"Class '{class_name}' not found in knowledge graph"
            }
        
        actual_name = class_record['name']
        full_name = class_record['full_name']
        
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
            params_to_use = record['params_detailed'] or record['params_list'] or []
            methods.append({
                'name': record['name'],
                'parameters': params_to_use,
                'return_type': record['return_type'] or 'Any'
            })
        
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
            attributes.append({
                'name': record['name'],
                'type': record['type'] or 'Any'
            })
        
        return {
            "success": True,
            "command": command,
            "data": {
                "class": {
                    "name": actual_name,
                    "full_name": full_name,
                    "methods": methods,
                    "attributes": attributes
                }
            },
            "metadata": {
                "total_results": 1,
                "methods_count": len(methods),
                "attributes_count": len(attributes),
                "limited": False
            }
        }
    
    async def _handle_method_command(self, session, command: str, method_name: str, class_name: Optional[str] = None) -> Dict[str, Any]:
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
            result = await session.run(query, class_name=class_name, method_name=method_name)
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
            params_to_use = record['params_detailed'] or record['params_list'] or []
            methods.append({
                'class_name': record['class_name'],
                'class_full_name': record['class_full_name'],
                'method_name': record['method_name'],
                'parameters': params_to_use,
                'return_type': record['return_type'] or 'Any',
                'legacy_args': record['args'] or []
            })
        
        if not methods:
            return {
                "success": False,
                "command": command,
                "error": f"Method '{method_name}'" + (f" in class '{class_name}'" if class_name else "") + " not found"
            }
        
        return {
            "success": True,
            "command": command,
            "data": {
                "methods": methods,
                "class_filter": class_name
            },
            "metadata": {
                "total_results": len(methods),
                "limited": len(methods) >= 20 and not class_name
            }
        }
    
    async def _handle_query_command(self, session, command: str, cypher_query: str) -> Dict[str, Any]:
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
            
            return {
                "success": True,
                "command": command,
                "data": {
                    "query": cypher_query,
                    "results": records
                },
                "metadata": {
                    "total_results": len(records),
                    "limited": len(records) >= 20
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "command": command,
                "error": f"Cypher query error: {str(e)}",
                "data": {
                    "query": cypher_query
                }
            }