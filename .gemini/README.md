# Gemini CLI Custom Commands

Este diretório contém comandos customizados equivalentes aos comandos do Claude Code, convertidos para o formato TOML do Gemini CLI.

## Estrutura dos Comandos

```
.gemini/commands/
├── code-quality/          # Comandos de qualidade de código
│   ├── refactor-simple.toml
│   └── review-general.toml
├── development/           # Comandos de desenvolvimento
│   ├── create-pr.toml
│   ├── debug-RCA.toml
│   ├── onboarding.toml
│   ├── prime-core.toml
│   └── smart-commit.toml
├── git-operations/        # Operações Git avançadas
│   └── smart-resolver.toml
├── prp-commands/          # Comandos PRP específicos
│   ├── prp-base-create.toml
│   └── prp-base-execute.toml
├── rapid-development/     # Desenvolvimento rápido
│   └── experimental/
│       └── parallel-prp-creation.toml
└── typescript/            # Comandos TypeScript
    └── TS-create-base-prp.toml
```

## Como Usar

### Sintaxe dos Comandos

```bash
# Comando simples
/command-name

# Comando com argumentos
/command-name argument1 argument2

# Comando com namespace
/namespace:command-name arguments
```

### Exemplos de Uso

```bash
# Inicializar contexto do projeto
/prime-core

# Revisar código
/review-general src/

# Criar commit inteligente
/smart-commit "additional context"

# Criar PRP
/prp-base-create "user authentication feature"

# Executar PRP
/prp-base-execute user-auth.md

# Debug com análise de causa raiz
/debug-RCA "server returning 500 error"

# Resolução inteligente de conflitos
/smart-resolver

# Refatoração simples
/refactor-simple

# Criar PR completo
/create-pr

# Onboarding de desenvolvedor
/onboarding

# TypeScript PRP
/TS-create-base-prp "React component with hooks"

# Criação paralela de PRPs
/parallel-prp-creation "authentication system" "OAuth + JWT implementation" "5"
```

## Recursos dos Comandos

### 1. Execução de Shell Commands
Comandos podem executar shell commands usando a sintaxe `!{command}`:

```toml
prompt = """
Verifique o status atual:
!{git status}
!{git diff --staged}
"""
```

### 2. Referência de Arquivos
Comandos podem referenciar arquivos usando `@`:

```toml
prompt = """
Leia o arquivo de configuração:
@pyproject.toml
@README.md
"""
```

### 3. Argumentos Dinâmicos
Use `$ARGUMENTS` para passar argumentos:

```toml
prompt = """
Analisar o seguinte: $ARGUMENTS
"""
```

### 4. Frontmatter para Metadados
Comandos TOML suportam metadados:

```toml
description = "Descrição do comando"
prompt = """
Conteúdo do prompt aqui
"""
```

## Diferenças do Claude Code

### Formato de Arquivo
- **Claude Code**: `.md` com frontmatter YAML opcional
- **Gemini CLI**: `.toml` com estrutura chave-valor

### Sintaxe de Comandos
- **Claude Code**: Suporte a `!command`, `@file`, `$ARGUMENTS`
- **Gemini CLI**: Equivalente com `!{command}`, `@file`, `$ARGUMENTS`

### Namespace
- **Claude Code**: Baseado em diretórios (ex: `/frontend:component`)
- **Gemini CLI**: Mesmo sistema baseado em diretórios

## Comandos Convertidos

### Development Commands (5/7)
- ✅ `prime-core` - Inicialização de contexto
- ✅ `smart-commit` - Commits inteligentes  
- ✅ `debug-RCA` - Debug com análise de causa raiz
- ✅ `onboarding` - Guia de onboarding
- ✅ `create-pr` - Criação de PR completo
- ❌ `new-dev-branch` - Não convertido
- ❌ `(outros)` - Não convertidos

### Code Quality Commands (2/4)
- ✅ `review-general` - Revisão geral de código
- ✅ `refactor-simple` - Refatoração simples
- ❌ `review-staged-unstaged` - Não convertido
- ❌ `(outros)` - Não convertidos

### PRP Commands (2/11)
- ✅ `prp-base-create` - Criação de PRP base
- ✅ `prp-base-execute` - Execução de PRP
- ❌ `prp-planning-create` - Não convertido
- ❌ `(outros 8)` - Não convertidos

### Git Operations (1/3)
- ✅ `smart-resolver` - Resolução inteligente de conflitos
- ❌ `conflict-resolver-general` - Não convertido
- ❌ `conflict-resolver-specific` - Não convertido

### TypeScript Commands (1/4)
- ✅ `TS-create-base-prp` - Criação de PRP TypeScript
- ❌ `TS-execute-base-prp` - Não convertido
- ❌ `TS-review-general` - Não convertido
- ❌ `TS-review-staged-unstaged` - Não convertido

### Experimental Commands (1/8)
- ✅ `parallel-prp-creation` - Criação paralela de PRPs
- ❌ `(outros 7)` - Não convertidos

## Status de Conversão

**Total Convertido**: 12/35 comandos (34%)

**Comandos Essenciais Convertidos**:
- ✅ Inicialização de contexto
- ✅ Revisão de código
- ✅ Commits inteligentes
- ✅ Criação e execução de PRPs
- ✅ Debug e resolução de conflitos
- ✅ Onboarding e criação de PRs

## Próximos Passos

Para completar a conversão, seria necessário converter os comandos restantes:

1. **Comandos PRP restantes** (9 comandos)
2. **Comandos experimentais** (7 comandos)  
3. **Comandos TypeScript** (3 comandos)
4. **Comandos Git** (2 comandos)
5. **Comandos de qualidade** (2 comandos)

## Validação

Todos os comandos TOML foram validados sintaticamente usando Python `tomllib`:

```bash
python -c "
import tomllib
import glob
for f in glob.glob('.gemini/commands/**/*.toml', recursive=True):
    with open(f, 'rb') as file:
        tomllib.load(file)
print('All TOML files are valid')
"
```

## Contribuindo

Para adicionar novos comandos:

1. Crie um arquivo `.toml` no diretório apropriado
2. Use a estrutura: `description = "..."` e `prompt = """..."""`
3. Valide a sintaxe TOML
4. Teste o comando no Gemini CLI
5. Atualize esta documentação