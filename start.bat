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
REM Carrega configuracoes do .env para verificacao
for /f "tokens=1,* delims==" %%a in ('findstr /r /c:"^QDRANT_HOST=" .env') do set QDRANT_HOST=%%b
for /f "tokens=1,* delims==" %%a in ('findstr /r /c:"^QDRANT_PORT=" .env') do set QDRANT_PORT=%%b
for /f "tokens=1,* delims==" %%a in ('findstr /r /c:"^NEO4J_URI=" .env') do set NEO4J_URI=%%b
for /f "tokens=1,* delims==" %%a in ('findstr /r /c:"^REDIS_HOST=" .env') do set REDIS_HOST=%%b
for /f "tokens=1,* delims==" %%a in ('findstr /r /c:"^REDIS_PORT=" .env') do set REDIS_PORT=%%b
for /f "tokens=1,* delims==" %%a in ('findstr /r /c:"^PORT=" .env') do set SERVER_PORT=%%b

set "QDRANT_HOST=!QDRANT_HOST:localhost=127.0.0.1!"
set "REDIS_HOST=!REDIS_HOST:localhost=127.0.0.1!"

echo Verificando Qdrant em !QDRANT_HOST!:!QDRANT_PORT!...
powershell -Command "(New-Object Net.Sockets.TcpClient).Connect('!QDRANT_HOST!', !QDRANT_PORT!)" 2>nul
if !errorlevel! neq 0 (
    echo AVISO: Qdrant indisponivel. Execute 'setup.bat' para iniciar.
    set /p "continue=Continuar? (s/N): "
    if /i not "!continue!"=="s" ( echo Cancelado. & pause & exit /b 1 )
) else ( echo OK Qdrant acessivel )

echo Verificando Neo4j...
for /f "tokens=2,3 delims=:/ " %%a in ("!NEO4J_URI!") do (
    set NEO4J_HOST=%%a
    set NEO4J_PORT=%%b
)
set "NEO4J_HOST=!NEO4J_HOST:localhost=127.0.0.1!"
powershell -Command "(New-Object Net.Sockets.TcpClient).Connect('!NEO4J_HOST!', !NEO4J_PORT!)" 2>nul
if !errorlevel! neq 0 (
    echo AVISO: Neo4j indisponivel.
) else ( echo OK Neo4j acessivel )

echo Verificando Redis em !REDIS_HOST!:!REDIS_PORT!...
powershell -Command "(New-Object Net.Sockets.TcpClient).Connect('!REDIS_HOST!', !REDIS_PORT!)" 2>nul
if !errorlevel! neq 0 (
    echo AVISO: Redis indisponivel. Cache desabilitado.
) else ( echo OK Redis acessivel )

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

echo [6/7] Verificando se a porta !SERVER_PORT! esta em uso...
set "PID="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":!SERVER_PORT!"') do (
    if not defined PID set "PID=%%a"
)

if defined PID (
    if "%PID%" NEQ "0" (
        echo A porta !SERVER_PORT! esta sendo usada pelo processo com PID %PID%.
        echo Encerrando o processo...
        taskkill /F /PID %PID% >nul
        if !errorlevel! equ 0 (
            echo OK Processo encerrado com sucesso.
        ) else (
            echo AVISO: Nao foi possivel encerrar o processo com PID %PID%.
        )
    )
) else (
    echo OK Porta !SERVER_PORT! esta livre.
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