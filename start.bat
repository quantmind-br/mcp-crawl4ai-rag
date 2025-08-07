@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

REM start.bat - Start Crawl4AI MCP RAG server

echo ========================================
echo   Crawl4AI MCP RAG - Server Startup
echo ========================================
echo.

echo [1/6] Checking uv installation...
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: uv nao esta instalado.
    echo Instale com: pip install uv
    pause
    exit /b 1
)
echo OK uv disponivel

echo [2/6] Verificando arquivo .env...
if not exist ".env" (
    echo AVISO: .env nao encontrado. Copiando .env.example...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo OK .env criado a partir de .env.example
    ) else (
        echo ERRO: .env.example ausente.
        pause
        exit /b 1
    )
) else (
    echo OK .env encontrado
)

echo [3/6] Verificando servicos Docker...
echo Verificando Qdrant...
curl -s http://localhost:6333/health >nul 2>&1
if %errorlevel% neq 0 (
    echo AVISO: Qdrant indisponivel em localhost:6333
    echo Execute 'setup.bat' antes para iniciar os servicos.
    set /p "continue=Continuar assim mesmo? (s/N): "
    if /i not "%continue%"=="s" (
        echo Inicio cancelado.
        pause
        exit /b 1
    )
) else (
    echo OK Qdrant acessivel
)

echo Verificando Neo4j...
netstat -an | find "7474" >nul 2>&1
if %errorlevel% neq 0 (
    echo AVISO: Neo4j nao acessivel em localhost:7474
) else (
    echo OK Neo4j acessivel
)

echo Verificando Redis...
netstat -an | find "6379" >nul 2>&1
if %errorlevel% neq 0 (
    echo AVISO: Redis nao acessivel em localhost:6379
    echo Cache de embeddings sera desabilitado.
) else (
    echo OK Redis acessivel
)

echo [4/6] Instalando dependencias...
uv sync
if %errorlevel% neq 0 (
    echo ERRO: Falha ao instalar dependencias.
    pause
    exit /b 1
)
echo OK Dependencias instaladas

echo [5/6] Verificando arquivos do servidor...
if not exist "src\__main__.py" (
    echo ERRO: src\__main__.py nao encontrado.
    pause
    exit /b 1
)
echo OK Arquivos OK

echo [6/7] Verificando se a porta 8051 esta em uso...
set "PID="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8051"') do (
    if not defined PID set "PID=%%a"
)

if defined PID (
    if "%PID%" NEQ "0" (
        echo A porta 8051 esta sendo usada pelo processo com PID %PID%.
        echo Encerrando o processo...
        taskkill /F /PID %PID% >nul
        if !errorlevel! equ 0 (
            echo OK Processo encerrado com sucesso.
        ) else (
            echo AVISO: Nao foi possivel encerrar o processo com PID %PID%.
        )
    )
) else (
    echo OK Porta 8051 esta livre.
)


echo [7/7] Iniciando servidor MCP...
uv run -m src
if %errorlevel% neq 0 (
    echo Tentando alternativa...
    uv run run_server.py
)
set exit_code=%errorlevel%

echo.
if %exit_code% neq 0 (
    echo ========================================
    echo     ERRO Servidor finalizado com erro
    echo ========================================
    echo Codigo: %exit_code%
) else (
    echo ========================================
    echo      OK Servidor finalizado
    echo ========================================
)

endlocal
pause