@echo off
REM Windows batch script for database cleanup

echo ============================================
echo    MCP Crawl4AI RAG Database Cleanup
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python and add to PATH.
    pause
    exit /b 1
)

REM Change to script directory
cd /d "%~dp0"

REM Show options
echo Available options:
echo   1. Clean both databases (with confirmation)
echo   2. Clean both databases (skip confirmation) 
echo   3. Clean only Qdrant
echo   4. Clean only Neo4j
echo   5. Dry run (show what would be deleted)
echo   6. Cancel
echo.

set /p choice="Select option (1-6): "

if "%choice%"=="1" (
    echo.
    echo Running: Clean both databases with confirmation
    python cleanup_databases.py
) else if "%choice%"=="2" (
    echo.
    echo Running: Clean both databases without confirmation
    python cleanup_databases.py --confirm
) else if "%choice%"=="3" (
    echo.
    echo Running: Clean only Qdrant
    python cleanup_databases.py --qdrant-only
) else if "%choice%"=="4" (
    echo.
    echo Running: Clean only Neo4j
    python cleanup_databases.py --neo4j-only
) else if "%choice%"=="5" (
    echo.
    echo Running: Dry run (no actual deletion)
    python cleanup_databases.py --dry-run
) else if "%choice%"=="6" (
    echo Operation cancelled.
    goto :end
) else (
    echo Invalid choice. Please run again and select 1-6.
    goto :end
)

echo.
echo Cleanup completed!

:end
pause