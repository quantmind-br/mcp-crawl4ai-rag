@echo off
REM Ativa o ambiente virtual Python e executa o comando desejado

call .venv\Scripts\activate.bat
uv run python -m src.crawl4ai_mcp
