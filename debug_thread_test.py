#!/usr/bin/env python3
"""
Debug script to test thread-based analysis collection.
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.unified_indexing_service import UnifiedIndexingService  # noqa: E402
from src.models.unified_indexing_models import UnifiedIndexingRequest, IndexingDestination  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_thread_analysis():
    """Test if analysis data is collected properly when using threads."""
    
    service = UnifiedIndexingService()
    
    try:
        # Create a temporary Go file with known content
        go_content = '''package main

import "fmt"

func main() {
    fmt.Println("Hello World")
}

func TestFunction(x int, y string) bool {
    return x > 0
}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(go_content)
            test_file_path = Path(f.name)
        
        logger.info(f"Created test file: {test_file_path}")
        
        # Create a mock request like the unified indexing service uses
        request = UnifiedIndexingRequest(
            repo_url='https://github.com/test/repo',
            destination=IndexingDestination.NEO4J,
            file_types=['.go'],
            max_files=5,
            chunk_size=5000,
            max_size_mb=500
        )
        
        # Test using the same method path as the unified indexing service
        logger.info("Testing _process_file_for_neo4j (with threads)...")
        result = await service._process_file_for_neo4j(
            test_file_path, 
            go_content, 
            'test-repo:test.go',
            request
        )
        
        logger.info(f"Thread-based parse result: {result}")
        
        # Check if analyses were collected after threading
        if hasattr(service, '_neo4j_analyses'):
            logger.info(f"THREAD TEST - Number of analyses collected: {len(service._neo4j_analyses)}")
            for i, analysis in enumerate(service._neo4j_analyses):
                logger.info(f"THREAD TEST - Analysis {i}: Functions={len(analysis.get('functions', []))}")
        else:
            logger.warning("THREAD TEST - No _neo4j_analyses attribute found after threading!")
            
        # Now test the batch processing if we have data
        if hasattr(service, '_neo4j_analyses') and service._neo4j_analyses:
            logger.info("Testing batch processing...")
            batch_result = await service._batch_process_neo4j_analyses()
            logger.info(f"Batch processing result: {batch_result}")
        
        # Cleanup temp file
        test_file_path.unlink()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
    finally:
        await service.cleanup()

if __name__ == "__main__":
    asyncio.run(test_thread_analysis())