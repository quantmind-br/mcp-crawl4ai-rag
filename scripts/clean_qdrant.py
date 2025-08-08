#!/usr/bin/env python3
"""
Interactive Qdrant Database Cleanup Script

This interactive script safely cleans all data from Qdrant collections with a menu-driven interface.
It provides options for:
- Cleaning all collections
- Cleaning specific collections
- Backing up data before cleaning
- Recreating collections with fresh configuration
- Listing collections and their information

Usage:
    python clean_qdrant.py    # Interactive menu interface
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_value = getattr(logging, log_level, logging.INFO)

# Configure basic logging format
logging.basicConfig(
    level=log_level_value,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # Force reconfiguration even if basicConfig was called before
)

# Ensure root logger level is set correctly
logging.getLogger().setLevel(log_level_value)
logger = logging.getLogger(__name__)


class QdrantCleaner:
    """Qdrant database cleanup utility"""

    def __init__(self):
        """Initialize the cleaner with Qdrant connection"""
        try:
            self.client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
            )
            logger.info("Connected to Qdrant successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            sys.exit(1)

    def list_collections(self):
        """List all existing collections"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            logger.info(f"Found collections: {collection_names}")
            return collection_names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def get_collection_info(self, collection_name):
        """Get detailed information about a collection"""
        try:
            info = self.client.get_collection(collection_name)
            count = self.client.count(collection_name)
            return {
                "name": collection_name,
                "vectors_count": count.count,
                "config": info.config,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get info for collection {collection_name}: {e}")
            return None

    def backup_collection(self, collection_name, backup_dir="backups"):
        """Create a backup of collection metadata and optionally data"""
        try:
            # Create backup directory
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True)

            # Create timestamped backup file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"{collection_name}_backup_{timestamp}.json"

            # Get collection info
            info = self.get_collection_info(collection_name)
            if not info:
                return False

            # Save backup
            with open(backup_file, "w") as f:
                json.dump(info, f, indent=2, default=str)

            logger.info(f"Backup created: {backup_file}")
            logger.info(
                f"Collection {collection_name}: {info['vectors_count']} vectors backed up"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to backup collection {collection_name}: {e}")
            return False

    def clean_collection(self, collection_name, backup=False):
        """Clean all data from a specific collection"""
        try:
            # Check if collection exists
            collections = self.list_collections()
            if collection_name not in collections:
                logger.warning(f"Collection {collection_name} does not exist")
                return False

            # Get collection info before cleaning
            info = self.get_collection_info(collection_name)
            if not info:
                return False

            original_count = info["vectors_count"]
            logger.info(f"Collection {collection_name} has {original_count} vectors")

            # Create backup if requested
            if backup:
                if not self.backup_collection(collection_name):
                    logger.error("Backup failed, aborting clean operation")
                    return False

            # Clean the collection by deleting all points
            if original_count > 0:
                # Delete all points using scroll and delete
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=10000,  # Process in batches
                    with_payload=False,
                    with_vectors=False,
                )

                while scroll_result[0]:  # While there are points
                    point_ids = [point.id for point in scroll_result[0]]
                    if point_ids:
                        self.client.delete(
                            collection_name=collection_name, points_selector=point_ids
                        )
                        logger.info(
                            f"Deleted {len(point_ids)} points from {collection_name}"
                        )

                    # Get next batch
                    if scroll_result[1]:  # If there's a next_page_offset
                        scroll_result = self.client.scroll(
                            collection_name=collection_name,
                            offset=scroll_result[1],
                            limit=10000,
                            with_payload=False,
                            with_vectors=False,
                        )
                    else:
                        break

            # Verify cleaning
            final_count = self.client.count(collection_name).count
            logger.info(
                f"Collection {collection_name} cleaned: {original_count} → {final_count} vectors"
            )

            return final_count == 0

        except Exception as e:
            logger.error(f"Failed to clean collection {collection_name}: {e}")
            return False

    def recreate_collection(self, collection_name):
        """Delete and recreate a collection with fresh configuration"""
        try:
            # Get current configuration before deletion
            info = self.get_collection_info(collection_name)
            if not info:
                logger.warning(
                    f"Collection {collection_name} does not exist, cannot recreate"
                )
                return False

            original_count = info["vectors_count"]
            vector_config = info["config"].params.vectors

            logger.info(
                f"Recreating collection {collection_name} (had {original_count} vectors)"
            )

            # Delete the collection
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name}")

            # Recreate with same configuration
            # Handle both single vector and named vectors configurations
            if hasattr(vector_config, "size"):
                # Single vector configuration
                vectors_config = VectorParams(
                    size=vector_config.size, distance=vector_config.distance
                )
            else:
                # Named vectors configuration - use default configuration
                vectors_config = VectorParams(
                    size=1024,  # Default embedding dimension
                    distance=Distance.COSINE,
                )

            self.client.create_collection(
                collection_name=collection_name, vectors_config=vectors_config
            )
            logger.info(f"Recreated collection {collection_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to recreate collection {collection_name}: {e}")
            return False

    def clean_all_collections(self, backup=False):
        """Clean all collections"""
        collections = self.list_collections()
        if not collections:
            logger.info("No collections found to clean")
            return True

        success_count = 0
        for collection_name in collections:
            if self.clean_collection(collection_name, backup):
                success_count += 1

        logger.info(
            f"Successfully cleaned {success_count}/{len(collections)} collections"
        )
        return success_count == len(collections)

    def recreate_all_collections(self):
        """Recreate all collections with fresh configuration"""
        collections = self.list_collections()
        if not collections:
            logger.info("No collections found to recreate")
            return True

        success_count = 0
        for collection_name in collections:
            if self.recreate_collection(collection_name):
                success_count += 1

        logger.info(
            f"Successfully recreated {success_count}/{len(collections)} collections"
        )
        return success_count == len(collections)


def display_menu():
    """Display the interactive menu"""
    print("\n" + "=" * 60)
    print("           Interactive Qdrant Database Cleanup")
    print("=" * 60)
    print("1. List all collections and their info")
    print("2. Clean specific collection")
    print("3. Clean all collections")
    print("4. Recreate specific collection")
    print("5. Recreate all collections")
    print("6. Create backup of specific collection")
    print("7. Exit")
    print("=" * 60)


def get_user_choice():
    """Get and validate user menu choice"""
    while True:
        try:
            choice = input("\nSelect an option (1-7): ").strip()
            if choice in ["1", "2", "3", "4", "5", "6", "7"]:
                return int(choice)
            else:
                print("Invalid choice. Please enter a number between 1 and 7.")
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Please enter a number between 1 and 7.")


def select_collection(cleaner, prompt="Select a collection"):
    """Interactive collection selection"""
    collections = cleaner.list_collections()
    if not collections:
        print("No collections found.")
        return None

    print("\nAvailable collections:")
    for i, collection_name in enumerate(collections, 1):
        info = cleaner.get_collection_info(collection_name)
        count = info["vectors_count"] if info else "unknown"
        print(f"{i}. {collection_name} ({count} vectors)")

    while True:
        try:
            choice = input(f"\n{prompt} (1-{len(collections)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(collections):
                return collections[index]
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(collections)}."
                )
        except (ValueError, KeyboardInterrupt):
            print(
                f"Invalid input. Please enter a number between 1 and {len(collections)}."
            )


def confirm_action(message):
    """Get user confirmation for destructive actions"""
    while True:
        response = input(f"\n{message} (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def ask_for_backup():
    """Ask user if they want to create a backup"""
    while True:
        response = input("\nCreate backup before operation? (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def main():
    """Main interactive function"""
    print("Starting Interactive Qdrant Database Cleanup...")

    # Support both interactive and CLI modes
    if len(sys.argv) > 1:
        # CLI mode (backward compatibility)
        run_cli_mode()
        return

    # Initialize cleaner
    try:
        cleaner = QdrantCleaner()
    except Exception as e:
        print(f"Failed to initialize Qdrant cleaner: {e}")
        sys.exit(1)

    while True:
        try:
            display_menu()
            choice = get_user_choice()

            if choice == 1:
                # List collections
                collections = cleaner.list_collections()
                if collections:
                    print("\nQdrant Collections:")
                    print("=" * 50)
                    for collection_name in collections:
                        info = cleaner.get_collection_info(collection_name)
                        if info:
                            print(
                                f"• {collection_name}: {info['vectors_count']} vectors"
                            )
                    print()
                else:
                    print("\nNo collections found.")

            elif choice == 2:
                # Clean specific collection
                collection_name = select_collection(
                    cleaner, "Select collection to clean"
                )
                if collection_name:
                    if confirm_action(
                        f"WARNING: This will CLEAN collection '{collection_name}' (delete all data)."
                    ):
                        backup = ask_for_backup()
                        print(f"\nCleaning collection '{collection_name}'...")
                        success = cleaner.clean_collection(
                            collection_name, backup=backup
                        )
                        if success:
                            print(
                                f"SUCCESS: Collection '{collection_name}' cleaned successfully!"
                            )
                        else:
                            print(
                                f"ERROR: Failed to clean collection '{collection_name}'. Check logs."
                            )
                    else:
                        print("Operation cancelled.")

            elif choice == 3:
                # Clean all collections
                if confirm_action(
                    "WARNING: This will CLEAN all collections (delete all data)."
                ):
                    backup = ask_for_backup()
                    print("\nCleaning all collections...")
                    success = cleaner.clean_all_collections(backup=backup)
                    if success:
                        print("SUCCESS: All collections cleaned successfully!")
                    else:
                        print("ERROR: Failed to clean some collections. Check logs.")
                else:
                    print("Operation cancelled.")

            elif choice == 4:
                # Recreate specific collection
                collection_name = select_collection(
                    cleaner, "Select collection to recreate"
                )
                if collection_name:
                    if confirm_action(
                        f"WARNING: This will DELETE and recreate collection '{collection_name}'."
                    ):
                        print(f"\nRecreating collection '{collection_name}'...")
                        success = cleaner.recreate_collection(collection_name)
                        if success:
                            print(
                                f"SUCCESS: Collection '{collection_name}' recreated successfully!"
                            )
                        else:
                            print(
                                f"ERROR: Failed to recreate collection '{collection_name}'. Check logs."
                            )
                    else:
                        print("Operation cancelled.")

            elif choice == 5:
                # Recreate all collections
                if confirm_action(
                    "WARNING: This will DELETE and recreate ALL collections."
                ):
                    print("\nRecreating all collections...")
                    success = cleaner.recreate_all_collections()
                    if success:
                        print("SUCCESS: All collections recreated successfully!")
                    else:
                        print("ERROR: Failed to recreate some collections. Check logs.")
                else:
                    print("Operation cancelled.")

            elif choice == 6:
                # Create backup
                collection_name = select_collection(
                    cleaner, "Select collection to backup"
                )
                if collection_name:
                    print(f"\nCreating backup of collection '{collection_name}'...")
                    success = cleaner.backup_collection(collection_name)
                    if success:
                        print(
                            f"SUCCESS: Backup of collection '{collection_name}' created successfully!"
                        )
                    else:
                        print(
                            f"ERROR: Failed to backup collection '{collection_name}'. Check logs."
                        )

            elif choice == 7:
                # Exit
                print("\nGoodbye!")
                break

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"ERROR: An unexpected error occurred: {e}")


def run_cli_mode():
    """Legacy CLI mode for backward compatibility"""
    parser = argparse.ArgumentParser(
        description="Clean Qdrant database collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --all                    # Clean all collections
    %(prog)s --collection crawled_pages  # Clean specific collection
    %(prog)s --recreate               # Delete and recreate all collections
    %(prog)s --backup --all           # Backup before cleaning all
    %(prog)s --list                   # List all collections
        """,
    )

    # Action arguments
    parser.add_argument("--all", action="store_true", help="Clean all collections")
    parser.add_argument("--collection", type=str, help="Clean specific collection")
    parser.add_argument(
        "--recreate", action="store_true", help="Delete and recreate all collections"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all collections and their info"
    )

    # Options
    parser.add_argument(
        "--backup", action="store_true", help="Create backup before cleaning"
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not any([args.all, args.collection, args.recreate, args.list]):
        parser.error("Must specify one of: --all, --collection, --recreate, or --list")

    # Initialize cleaner
    cleaner = QdrantCleaner()

    try:
        # List collections
        if args.list:
            collections = cleaner.list_collections()
            if collections:
                print("\nQdrant Collections:")
                print("=" * 50)
                for collection_name in collections:
                    info = cleaner.get_collection_info(collection_name)
                    if info:
                        print(f"• {collection_name}: {info['vectors_count']} vectors")
                print()
            else:
                print("No collections found.")
            return

        # Confirmation for destructive operations
        if not args.force:
            if args.recreate:
                response = input(
                    "WARNING: This will DELETE and recreate ALL collections. Continue? (yes/no): "
                )
            elif args.all:
                response = input(
                    "WARNING: This will CLEAN all collections (delete all data). Continue? (yes/no): "
                )
            elif args.collection:
                response = input(
                    f"WARNING: This will CLEAN collection '{args.collection}'. Continue? (yes/no): "
                )
            else:
                response = "yes"

            if response.lower() not in ["yes", "y"]:
                print("Operation cancelled.")
                return

        # Execute operations
        success = False

        if args.recreate:
            logger.info("Starting recreation of all collections...")
            success = cleaner.recreate_all_collections()

        elif args.all:
            logger.info("Starting cleanup of all collections...")
            success = cleaner.clean_all_collections(backup=args.backup)

        elif args.collection:
            logger.info(f"Starting cleanup of collection: {args.collection}")
            success = cleaner.clean_collection(args.collection, backup=args.backup)

        # Report results
        if success:
            print("SUCCESS: Operation completed successfully!")
        else:
            print("ERROR: Operation failed. Check logs for details.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
