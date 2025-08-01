@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

REM setup.bat - Initialize Docker stack for Crawl4AI MCP RAG application

echo ========================================
echo   Crawl4AI MCP RAG - Docker Setup
echo ========================================
echo.

echo [1/4] Verificando Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Docker nao instalado ou inativo.
    pause
    exit /b 1
)
echo OK Docker disponivel

echo [2/4] Verificando docker-compose.yaml...
if not exist "docker-compose.yaml" (
    echo ERRO: docker-compose.yaml nao encontrado.
    pause
    exit /b 1
)
echo OK docker-compose.yaml OK

echo [3/4] Parando containers existentes...
docker-compose down --remove-orphans --volumes 2>nul
echo OK Limpeza concluida

echo [4/4] Iniciando servicos Docker...
echo Iniciando Qdrant e Neo4j...
docker-compose up -d

timeout /t 5 /nobreak >nul
echo.
echo Verificando status...
docker-compose ps

echo.
echo Aguardando servicos inicializarem...

:wait_qdrant
echo Verificando Qdrant...
curl -s http://localhost:6333/health >nul 2>&1
if %errorlevel% neq 0 (
    echo Qdrant ainda iniciando...
    timeout /t 3 /nobreak >nul
    goto wait_qdrant
)
echo OK Qdrant pronto em http://localhost:6333

echo Verificando Neo4j...
timeout /t 10 /nobreak >nul
netstat -an | find "7474" >nul
if %errorlevel% neq 0 (
    echo Neo4j pode ainda estar iniciando...
    timeout /t 5 /nobreak >nul
)
echo OK Neo4j disponivel em http://localhost:7474

echo.
echo ========================================
echo       Setup concluido!
echo ========================================

echo Comandos uteis:
echo   docker-compose logs
echo   docker-compose logs qdrant
echo   docker-compose logs neo4j
echo   docker-compose down

endlocal
pause
