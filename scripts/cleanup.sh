#!/bin/bash
# Linux/Mac shell script for database cleanup

echo "============================================"
echo "    MCP Crawl4AI RAG Database Cleanup"
echo "============================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python."
    exit 1
fi

# Change to script directory
cd "$(dirname "$0")"

# Show options
echo "Available options:"
echo "  1. Clean both databases (with confirmation)"
echo "  2. Clean both databases (skip confirmation)"
echo "  3. Clean only Qdrant"
echo "  4. Clean only Neo4j"  
echo "  5. Dry run (show what would be deleted)"
echo "  6. Cancel"
echo

read -p "Select option (1-6): " choice

case $choice in
    1)
        echo
        echo "Running: Clean both databases with confirmation"
        python3 cleanup_databases.py
        ;;
    2)
        echo
        echo "Running: Clean both databases without confirmation"
        python3 cleanup_databases.py --confirm
        ;;
    3)
        echo
        echo "Running: Clean only Qdrant"
        python3 cleanup_databases.py --qdrant-only
        ;;
    4)
        echo
        echo "Running: Clean only Neo4j"
        python3 cleanup_databases.py --neo4j-only
        ;;
    5)
        echo
        echo "Running: Dry run (no actual deletion)"
        python3 cleanup_databases.py --dry-run
        ;;
    6)
        echo "Operation cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-6."
        exit 1
        ;;
esac

echo
echo "Cleanup completed!"