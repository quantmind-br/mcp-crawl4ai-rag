#!/usr/bin/env python3
"""
Debug script to verify environment variable loading.
"""
import os
from dotenv import load_dotenv

print("=== Debug Environment Variables ===")
print("Before load_dotenv():")
print(f"USE_HYBRID_SEARCH = {os.getenv('USE_HYBRID_SEARCH', 'NOT_SET')}")
print()

load_dotenv()

print("After load_dotenv():")
print(f"USE_HYBRID_SEARCH = {os.getenv('USE_HYBRID_SEARCH', 'NOT_SET')}")
print()

# Test the Qdrant client initialization
print("=== Testing Qdrant Client ===")
try:
    import sys
    sys.path.insert(0, 'src')
    
    from clients.qdrant_client import get_qdrant_client
    
    client = get_qdrant_client()
    print(f"Hybrid search enabled: {client.use_hybrid_search}")
    
    # Test collection config
    from clients.qdrant_client import get_hybrid_collections_config, get_active_collections_config
    
    hybrid_config = get_hybrid_collections_config()
    active_config = get_active_collections_config()
    
    print(f"Hybrid config keys: {list(hybrid_config.keys())}")
    print(f"Active config keys: {list(active_config.keys())}")
    
    # Check if active config is using hybrid
    if active_config == hybrid_config:
        print("[OK] Active configuration is using hybrid search")
    else:
        print("[ERROR] Active configuration is NOT using hybrid search")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()