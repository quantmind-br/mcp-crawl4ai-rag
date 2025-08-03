#!/usr/bin/env python3
"""
Fix Qdrant dimensions mismatch issue.

This script recreates Qdrant collections with the correct dimensions
based on the current embedding model configuration.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from embedding_config import get_embedding_dimensions, validate_embeddings_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Fix Qdrant dimension mismatch."""
    
    print("Fixing Qdrant dimension mismatch...")
    print("=" * 50)
    
    try:
        # Step 1: Validate embedding configuration
        print("[1/5] Validating embedding configuration...")
        validate_embeddings_config()
        current_dims = get_embedding_dimensions()
        print(f"OK Current embedding dimensions: {current_dims}")
        
        # Step 2: Connect to Qdrant
        print("\n[2/5] Connecting to Qdrant...")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        print(f"OK Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        
        # Step 3: Check existing collections
        print("\n[3/5] Checking existing collections...")
        collections = client.get_collections()
        
        target_collections = ["crawled_pages", "code_examples"]
        existing_collections = [col.name for col in collections.collections]
        
        for collection_name in target_collections:
            if collection_name in existing_collections:
                collection_info = client.get_collection(collection_name)
                # Handle different Qdrant client versions
                vectors_config = collection_info.config.params.vectors
                if hasattr(vectors_config, 'size'):
                    # Single vector configuration
                    current_collection_dims = vectors_config.size
                elif hasattr(vectors_config, '__getitem__'):
                    # Named vectors configuration (dict-like)
                    current_collection_dims = list(vectors_config.values())[0].size
                else:
                    # Fallback: assume it's a dict
                    current_collection_dims = vectors_config.size
                    
                print(f"INFO Collection '{collection_name}': {current_collection_dims} dimensions")
                
                if current_collection_dims != current_dims:
                    print(f"WARNING Dimension mismatch detected for '{collection_name}'!")
                    print(f"   Expected: {current_dims}, Found: {current_collection_dims}")
                else:
                    print(f"OK Collection '{collection_name}' has correct dimensions")
            else:
                print(f"INFO Collection '{collection_name}' not found (will be created)")
        
        # Step 4: Recreate collections with correct dimensions
        print(f"\n[4/5] Recreating collections with {current_dims} dimensions...")
        
        vector_config = VectorParams(size=current_dims, distance=Distance.COSINE)
        
        for collection_name in target_collections:
            print(f"\nProcessing collection '{collection_name}'...")
            
            # Delete existing collection if it exists
            if collection_name in existing_collections:
                print(f"   Deleting existing collection...")
                client.delete_collection(collection_name)
                print(f"   DELETED '{collection_name}'")
            
            # Create new collection with correct dimensions
            print(f"   Creating new collection with {current_dims} dimensions...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config
            )
            print(f"   CREATED '{collection_name}' with {current_dims} dimensions")
        
        # Step 5: Verify collections
        print(f"\n[5/5] Verifying collections...")
        for collection_name in target_collections:
            collection_info = client.get_collection(collection_name)
            # Handle different Qdrant client versions for verification
            vectors_config = collection_info.config.params.vectors
            if hasattr(vectors_config, 'size'):
                actual_dims = vectors_config.size
            elif hasattr(vectors_config, '__getitem__'):
                actual_dims = list(vectors_config.values())[0].size
            else:
                actual_dims = vectors_config.size
            
            if actual_dims == current_dims:
                print(f"OK '{collection_name}': {actual_dims} dimensions (correct)")
            else:
                print(f"ERROR '{collection_name}': {actual_dims} dimensions (expected {current_dims})")
                raise Exception(f"Collection '{collection_name}' still has wrong dimensions!")
        
        print("\n" + "=" * 50)
        print("SUCCESS: Qdrant dimension fix completed!")
        print(f"All collections now use {current_dims} dimensions")
        print("Server should now work without dimension errors")
        print("\nNext steps:")
        print("1. Restart the MCP server")
        print("2. Test with a crawling operation")
        print("3. Verify embeddings are working correctly")
        
    except Exception as e:
        print(f"\nERROR fixing Qdrant dimensions: {e}")
        logger.exception("Failed to fix Qdrant dimensions")
        sys.exit(1)

if __name__ == "__main__":
    main()