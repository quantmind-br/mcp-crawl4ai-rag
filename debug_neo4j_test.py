#!/usr/bin/env python3
"""
Debug script to diagnose Neo4j batch processing issues.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.unified_indexing_service import UnifiedIndexingService
from src.models.unified_indexing_models import UnifiedIndexingRequest, IndexingDestination

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_neo4j_parsing():
    """Test Neo4j parsing with detailed debugging."""
    
    # Create a simple test case with minimal Go code
    service = UnifiedIndexingService()
    
    try:
        # Initialize Neo4j connection
        await service.neo4j_parser.initialize()
        
        # Create a mock analysis result similar to what would be generated
        mock_analysis = {
            'file_path': 'test.go',
            'module_name': 'test',
            'language': 'go',
            'line_count': 10,
            'classes': [],
            'functions': [
                {
                    'name': 'TestFunction',
                    'full_name': 'TestFunction',
                    'args': [],
                    'params_list': [],
                    'params_detailed': [],
                    'return_type': 'void'
                }
            ],
            'imports': [],
            'file_id': 'test-repo:test.go'
        }
        
        # Simulate the batch processing
        service._neo4j_analyses = [mock_analysis]
        service._neo4j_repo_name = 'debug-test-repo'
        
        logger.info("Testing batch processing with mock data...")
        result = await service._batch_process_neo4j_analyses()
        
        if result:
            logger.info(f"Batch processing result: {result}")
        else:
            logger.error("Batch processing returned None")
            
        # Check if repository was created
        from src.tools.rag_tools import query_knowledge_graph_command
        repos_result = await query_knowledge_graph_command("repos")
        logger.info(f"Repositories in database: {repos_result}")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
    finally:
        await service.cleanup()

if __name__ == "__main__":
    asyncio.run(test_neo4j_parsing())