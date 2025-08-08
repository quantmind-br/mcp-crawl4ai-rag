#!/usr/bin/env python3
"""
Debug script to examine Tree-sitter analysis data.
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.unified_indexing_service import UnifiedIndexingService

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_analysis_data():
    """Test what analysis data is generated for a simple Go file."""
    
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

type TestStruct struct {
    Name string
    ID   int
}

func (t TestStruct) GetName() string {
    return t.Name
}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(go_content)
            test_file_path = Path(f.name)
        
        logger.info(f"Created test file: {test_file_path}")
        logger.info(f"File content preview: {go_content[:100]}...")
        
        # Test the _parse_file_for_neo4j method directly
        result = service._parse_file_for_neo4j(
            test_file_path, 
            go_content, 
            'test-repo:test.go',
            'https://github.com/test/repo'
        )
        
        logger.info(f"Parse result: {result}")
        
        # Check if analyses were collected
        if hasattr(service, '_neo4j_analyses'):
            logger.info(f"Number of analyses collected: {len(service._neo4j_analyses)}")
            for i, analysis in enumerate(service._neo4j_analyses):
                logger.info(f"Analysis {i}: {analysis}")
                logger.info(f"  Classes: {len(analysis.get('classes', []))}")
                logger.info(f"  Functions: {len(analysis.get('functions', []))}")
                for func in analysis.get('functions', []):
                    logger.info(f"    Function: {func}")
        else:
            logger.warning("No _neo4j_analyses attribute found")
            
        # Cleanup temp file
        test_file_path.unlink()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
    finally:
        await service.cleanup()

if __name__ == "__main__":
    asyncio.run(test_analysis_data())