#!/bin/bash
# Script de publicação automática kazuba-converter no PyPI
# Uso: ./publish_pypi_auto.sh [version_bump]
# Exemplo: ./publish_pypi_auto.sh patch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/home/gabrielgadea/projects/analise/kazuba-products/p1-converter"

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PUBLICAÇÃO AUTOMÁTICA PyPI${NC}"
echo -e "${BLUE}  kazuba-converter${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Verificar argumentos
VERSION_BUMP="${1:-patch}"

# Carregar variáveis de ambiente
if [ -f "$HOME/.openclaw/env/pypi_env.sh" ]; then
    source "$HOME/.openclaw/env/pypi_env.sh"
    echo -e "${GREEN}✅ Variáveis de ambiente carregadas${NC}"
else
    echo -e "${RED}❌ Arquivo de ambiente não encontrado: ~/.openclaw/env/pypi_env.sh${NC}"
    exit 1
fi

# Navegar para o diretório do projeto
cd "$PROJECT_DIR"

# Verificar se há mudanças não commitadas
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}⚠️  Há mudanças não commitadas:${NC}"
    git status --short
    read -p "Deseja continuar mesmo assim? (s/N): " continue_anyway
    if [ "$continue_anyway" != "s" ] && [ "$continue_anyway" != "S" ]; then
        echo -e "${RED}❌ Publicação cancelada${NC}"
        exit 1
    fi
fi

# Obter versão atual
CURRENT_VERSION=$(grep -E '^version = "' pyproject.toml | sed 's/version = "//;s/"$//')
echo -e "${BLUE}📦 Versão atual: ${CURRENT_VERSION}${NC}"

# Calcular nova versão
if [ "$VERSION_BUMP" = "patch" ]; then
    NEW_VERSION=$(echo $CURRENT_VERSION | awk -F. '{printf "%d.%d.%d", $1, $2, $3+1}')
elif [ "$VERSION_BUMP" = "minor" ]; then
    NEW_VERSION=$(echo $CURRENT_VERSION | awk -F. '{printf "%d.%d.%d", $1, $2+1, 0}')
elif [ "$VERSION_BUMP" = "major" ]; then
    NEW_VERSION=$(echo $CURRENT_VERSION | awk -F. '{printf "%d.%d.%d", $1+1, 0, 0}')
else
    NEW_VERSION="$VERSION_BUMP"
fi

echo -e "${BLUE}📦 Nova versão: ${NEW_VERSION}${NC}"
read -p "Confirmar publicação da v${NEW_VERSION}? (s/N): " confirm

if [ "$confirm" != "s" ] && [ "$confirm" != "S" ]; then
    echo -e "${RED}❌ Publicação cancelada${NC}"
    exit 1
fi

# Atualizar versão no pyproject.toml
echo -e "${BLUE}📝 Atualizando versão...${NC}"
sed -i "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
echo -e "${GREEN}✅ Versão atualizada para ${NEW_VERSION}${NC}"

# Limpar builds anteriores
echo -e "${BLUE}🧹 Limpando builds anteriores...${NC}"
rm -rf dist/ build/ *.egg-info

# Ativar ambiente virtual
source .venv/bin/activate

# Build
echo -e "${BLUE}🔨 Buildando pacote...${NC}"
python -m build

# Verificar build
echo -e "${BLUE}🔍 Verificando pacote...${NC}"
twine check dist/*

# Publicar
echo -e "${BLUE}📤 Publicando no PyPI...${NC}"
twine upload dist/*

# Commit da nova versão
echo -e "${BLUE}💾 Fazendo commit da versão...${NC}"
git add pyproject.toml
if [ -f "CHANGELOG.md" ]; then
    git add CHANGELOG.md
fi
git commit -m "Bump version: ${CURRENT_VERSION} → ${NEW_VERSION}"
git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"
git push origin main
git push origin "v${NEW_VERSION}"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ PUBLICAÇÃO COMPLETA${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Versão: ${GREEN}v${NEW_VERSION}${NC}"
echo -e "PyPI:   ${GREEN}https://pypi.org/project/kazuba-converter/${NEW_VERSION}/${NC}"
echo ""
echo -e "Instalação: ${BLUE}pip install kazuba-converter==${NEW_VERSION}${NC}"
echo ""

# Atualizar memory
MEMORY_FILE="$HOME/.openclaw/workspace/memory/$(date +%Y-%m-%d).md"
if [ -f "$MEMORY_FILE" ]; then
    echo "" >> "$MEMORY_FILE"
    echo "### $(date +%H:%M) — Publicação kazuba-converter v${NEW_VERSION}" >> "$MEMORY_FILE"
    echo "- **Ação:** Publicação automática PyPI" >> "$MEMORY_FILE"
    echo "- **Versão:** v${NEW_VERSION}" >> "$MEMORY_FILE"
    echo "- **URL:** https://pypi.org/project/kazuba-converter/${NEW_VERSION}/" >> "$MEMORY_FILE"
    echo "- **Método:** Script automático com variáveis de ambiente" >> "$MEMORY_FILE"
fi