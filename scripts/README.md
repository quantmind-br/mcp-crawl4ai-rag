# Qdrant Database Scripts

This directory contains utility scripts for managing the Qdrant vector database.

## clean_qdrant.py

A comprehensive script for cleaning and managing Qdrant collections.

### Prerequisites

Make sure the Qdrant server is running:
```bash
docker-compose up -d qdrant
```

### Usage Examples

#### List all collections
```bash
python scripts/clean_qdrant.py --list
```

#### Clean all collections (with backup)
```bash
python scripts/clean_qdrant.py --backup --all
```

#### Clean a specific collection
```bash
python scripts/clean_qdrant.py --collection crawled_pages
```

#### Recreate all collections (complete reset)
```bash
python scripts/clean_qdrant.py --recreate
```

#### Force operation without confirmation
```bash
python scripts/clean_qdrant.py --force --all
```

### Features

- **Safe Operations**: Includes confirmation prompts and backup options
- **Backup Support**: Can create JSON backups of collection metadata
- **Batch Processing**: Efficiently handles large collections
- **Detailed Logging**: Comprehensive logging of all operations
- **Collection Recreation**: Can delete and recreate collections with fresh configuration
- **Error Handling**: Robust error handling and recovery

### Safety Notes

- Always use `--backup` flag when cleaning important data
- Use `--list` first to see what collections exist
- The script preserves collection structure when cleaning (only removes data)
- Use `--recreate` only when you want to completely reset collections