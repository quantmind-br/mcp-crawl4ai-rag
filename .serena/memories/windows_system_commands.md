# Windows System Commands Reference

## File System Operations
```cmd
# Directory listing
dir                    # List current directory
dir /s                # List recursively
dir *.py              # List Python files
tree                  # Show directory tree

# File operations
type filename.txt     # Display file contents (equivalent to cat)
copy source dest      # Copy files
move source dest      # Move files
del filename          # Delete files
mkdir dirname         # Create directory
rmdir dirname         # Remove directory

# Search operations
findstr "pattern" *.py    # Search in files (equivalent to grep)
findstr /s "pattern" *    # Search recursively
```

## Process and Network Management
```cmd
# Process management
tasklist              # Show running processes
tasklist | findstr python    # Find Python processes
taskkill /F /PID 1234       # Kill process by PID
taskkill /F /IM python.exe  # Kill process by name

# Network operations
netstat -an           # Show all network connections
netstat -an | findstr ":8051"  # Check specific port
ping hostname         # Test connectivity
curl -s http://localhost:6333/health  # HTTP requests (if curl installed)
```

## Environment and Path
```cmd
# Environment variables
set                   # Show all environment variables
set PATH              # Show PATH variable
set VARIABLE=value    # Set environment variable
echo %PATH%           # Display environment variable

# Current directory
cd                    # Show current directory
cd dirname            # Change directory
cd ..                 # Go up one directory
cd \                  # Go to root
```

## System Information
```cmd
# System info
ver                   # Windows version
systeminfo            # Detailed system information
whoami                # Current user
hostname              # Computer name

# Disk usage
dir /s                # Show sizes
wmic logicaldisk get size,freespace,caption  # Disk space
```

## Text Processing
```cmd
# Text manipulation
more filename         # Page through file (equivalent to less)
sort filename         # Sort file contents
find "string" filename    # Find string in file
fc file1 file2        # Compare files (equivalent to diff)
```

## Package and Development Tools
```cmd
# Python and UV
python --version      # Check Python version
pip --version         # Check pip version
uv --version          # Check UV version
uv sync               # Install dependencies
uv run -m src         # Run module

# Git operations (Windows Git Bash or Git for Windows)
git status            # Git status
git log --oneline     # Git history
git diff              # Git differences
```

## Docker on Windows
```cmd
# Docker management
docker --version      # Check Docker version
docker ps             # List running containers
docker ps -a          # List all containers
docker images         # List images
docker-compose up -d  # Start services
docker-compose down   # Stop services
docker-compose logs   # View logs
```

## Batch File Specific
```cmd
# Variables in batch files
set "VARIABLE=value"  # Set variable with quotes
if "%VARIABLE%"=="value" echo Match  # Conditional
for %%i in (*.txt) do echo %%i       # Loop through files
%errorlevel%          # Exit code of last command
```

## Service Management
```cmd
# Windows services
sc query              # List services
sc start servicename  # Start service
sc stop servicename   # Stop service
net start servicename # Alternative start
net stop servicename  # Alternative stop
```