#!/usr/bin/env python3
"""
Qdrant Schema Synchronization Script

This script synchronizes the Qdrant database schema with the application's
current configuration (from .env), recreating collections as needed.
It correctly handles both legacy and hybrid search schemas.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Imports after path setup for standalone script
from qdrant_client import QdrantClient  # noqa: E402
from clients.qdrant_client import get_active_collections_config  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def main():
    """Synchronize Qdrant schema with application configuration."""

    print("Qdrant Schema Synchronization Script")
    print("=" * 50)
    print("This script will delete and recreate your Qdrant collections")
    print("to match the current application configuration (.env file).")
    print("\nWARNING: THIS WILL DELETE ALL DATA IN THE COLLECTIONS.")

    confirm = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    if confirm not in ["yes", "y"]:
        print("Operation cancelled.")
        sys.exit(0)

    try:
        # Step 1: Get application's expected schema
        print("\n[1/4] Getting application's collection configuration...")
        try:
            # This will read .env and determine if hybrid is on, and get correct dimensions
            collections_config = get_active_collections_config()
            is_hybrid = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
            print(f"OK Configuration loaded. Hybrid search: {is_hybrid}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Please ensure your .env file is correctly configured.")
            sys.exit(1)

        # Step 2: Connect to Qdrant
        print("\n[2/4] Connecting to Qdrant...")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        print(f"OK Connected to Qdrant at {qdrant_host}:{qdrant_port}")

        # Step 3: Recreate collections
        print("\n[3/4] Recreating collections...")
        target_collections = ["crawled_pages", "code_examples", "sources"]

        for collection_name in target_collections:
            print(f"\nProcessing collection '{collection_name}'...")
            config = collections_config[collection_name]

            # Delete existing collection if it exists
            try:
                client.get_collection(collection_name)
                print("   Deleting existing collection...")
                client.delete_collection(collection_name)
                print(f"   DELETED '{collection_name}'")
            except Exception:
                print(f"   Collection '{collection_name}' does not exist. Skipping deletion.")

            # Create new collection with correct schema
            print("   Creating new collection with up-to-date schema...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=config["vectors_config"],
                sparse_vectors_config=config.get("sparse_vectors_config"),
            )
            print(f"   CREATED '{collection_name}'")

        # Step 4: Verify collections
        print("\n[4/4] Verifying collections...")
        for collection_name in target_collections:
            collection_info = client.get_collection(collection_name)

            is_qdrant_hybrid = (
                hasattr(collection_info.config.params, "sparse_vectors")
                and collection_info.config.params.sparse_vectors is not None
            )

            if is_qdrant_hybrid == is_hybrid:
                print(f"OK '{collection_name}': Schema is correct (Hybrid: {is_qdrant_hybrid})")
            else:
                print(f"ERROR '{collection_name}': Schema mismatch after creation!")
                raise Exception(f"Failed to correctly create '{collection_name}'")

        print("\n" + "=" * 50)
        print("SUCCESS: Qdrant schema synchronization completed!")
        print("All collections now match the application configuration.")
        print("\nYou can now start the MCP server.")

    except Exception as e:
        print(f"\nERROR during schema synchronization: {e}")
        logger.exception("Failed to synchronize Qdrant schema")
        sys.exit(1)


if __name__ == "__main__":
    main()
