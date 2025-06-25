@echo off
REM Script para iniciar o MCP Crawl4AI RAG usando Docker Compose (Windows)
REM Usage: start.bat [command]

setlocal enabledelayedexpansion

REM Função para verificar se o arquivo .env existe
:check_env_file
if not exist ".env" (
    echo ❌ Arquivo .env não encontrado!
    echo 📝 Por favor, crie um arquivo .env baseado na documentação.
    echo 💡 Exemplo: copy .env.example .env ^(se existir^)
    pause
    exit /b 1
)
echo ✅ Arquivo .env encontrado
goto :eof

REM Função para verificar se o Docker está rodando
:check_docker
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker não está rodando!
    echo 🐳 Por favor, inicie o Docker e tente novamente.
    pause
    exit /b 1
)
echo ✅ Docker está rodando
goto :eof

REM Função para verificar se o Docker Compose está disponível
:check_docker_compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker Compose não está disponível!
        echo 📦 Por favor, instale o Docker Compose.
        pause
        exit /b 1
    )
)
echo ✅ Docker Compose disponível
goto :eof

REM Função para mostrar ajuda
:show_help
echo 🚀 MCP Crawl4AI RAG - Script de Inicialização
echo.
echo 📋 Comandos disponíveis:
echo   start.bat              - Inicia os serviços
echo   start.bat up           - Inicia os serviços
echo   start.bat down         - Para os serviços
echo   start.bat restart      - Reinicia os serviços
echo   start.bat logs         - Mostra os logs
echo   start.bat status       - Mostra o status dos serviços
echo   start.bat build        - Reconstrói as imagens
echo   start.bat clean        - Para e remove containers/volumes
echo   start.bat help         - Mostra esta ajuda
echo.
echo 🔗 Acesso:
echo   - MCP Server: http://localhost:%%PORT%%
echo   - SSE Endpoint: http://localhost:%%PORT%%/sse
echo.
goto :eof

REM Função para obter a porta do arquivo .env
:get_port
set PORT_VALUE=8051
for /f "tokens=2 delims==" %%a in ('findstr "^PORT=" .env 2^>nul') do set PORT_VALUE=%%a
goto :eof

REM Função principal para executar comandos
:run_command
set cmd=%1
if "%cmd%"=="" set cmd=up

if "%cmd%"=="up" goto :start_services
if "%cmd%"=="start" goto :start_services
if "%cmd%"=="down" goto :stop_services
if "%cmd%"=="stop" goto :stop_services
if "%cmd%"=="restart" goto :restart_services
if "%cmd%"=="logs" goto :show_logs
if "%cmd%"=="status" goto :show_status
if "%cmd%"=="build" goto :build_images
if "%cmd%"=="clean" goto :clean_all
if "%cmd%"=="help" goto :show_help
if "%cmd%"=="-h" goto :show_help
if "%cmd%"=="--help" goto :show_help

echo ❌ Comando inválido: %cmd%
call :show_help
exit /b 1

:start_services
echo 🚀 Iniciando MCP Crawl4AI RAG...
docker-compose up -d
if errorlevel 1 (
    echo ❌ Erro ao iniciar serviços!
    pause
    exit /b 1
)
call :get_port
echo ✅ Serviços iniciados!
echo 🔗 Acesse: http://localhost:!PORT_VALUE!
goto :end

:stop_services
echo 🛑 Parando serviços...
docker-compose down
if errorlevel 1 (
    echo ❌ Erro ao parar serviços!
    pause
    exit /b 1
)
echo ✅ Serviços parados!
goto :end

:restart_services
echo 🔄 Reiniciando serviços...
docker-compose restart
if errorlevel 1 (
    echo ❌ Erro ao reiniciar serviços!
    pause
    exit /b 1
)
echo ✅ Serviços reiniciados!
goto :end

:show_logs
echo 📄 Mostrando logs...
docker-compose logs -f
goto :end

:show_status
echo 📊 Status dos serviços:
docker-compose ps
goto :end

:build_images
echo 🔨 Reconstruindo imagens...
docker-compose build --no-cache
if errorlevel 1 (
    echo ❌ Erro durante o build!
    pause
    exit /b 1
)
echo ✅ Build concluído!
goto :end

:clean_all
echo 🧹 Limpando containers e volumes...
docker-compose down -v --remove-orphans
docker system prune -f
echo ✅ Limpeza concluída!
goto :end

REM Função principal
:main
echo 🐳 MCP Crawl4AI RAG - Docker Compose Manager
echo.

REM Verificações iniciais (exceto para help)
if "%1"=="help" goto :run_command
if "%1"=="-h" goto :run_command
if "%1"=="--help" goto :run_command

call :check_env_file
call :check_docker
call :check_docker_compose
echo.

call :run_command %1
goto :end

:end
if "%1"=="logs" (
    REM Não pausa para logs pois fica em modo follow
    exit /b 0
) else (
    pause
    exit /b 0
)

REM Executa o script
call :main %1