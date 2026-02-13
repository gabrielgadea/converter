#!/bin/bash
# Script para publicar CONVERTER no GitHub e PyPI
# Execute após configurar credenciais

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  PUBLICAÇÃO CONVERTER v0.1.0"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cd /home/gabrielgadea/projects/analise/kazuba-products/p1-converter

echo -e "${BLUE}📁 Diretório:${NC} $(pwd)"
echo ""

# Verificar se gh está instalado
if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}⚠️  GitHub CLI (gh) não encontrado${NC}"
    echo "Instalando..."
    
    # Instalar gh
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y gh
    elif command -v brew &> /dev/null; then
        brew install gh
    else
        echo "❌ Não foi possível instalar gh automaticamente"
        echo "Instale manualmente: https://cli.github.com/"
        exit 1
    fi
fi

# Verificar autenticação
echo -e "${BLUE}🔐 Verificando autenticação GitHub...${NC}"
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}⚠️  Não autenticado no GitHub${NC}"
    echo "Execute: gh auth login"
    gh auth login
fi

echo -e "${GREEN}✅ Autenticado no GitHub${NC}"
echo ""

# Criar repositório no GitHub
echo -e "${BLUE}📦 Criando repositório kazuba/converter...${NC}"

if gh repo view kazuba/converter &> /dev/null; then
    echo -e "${YELLOW}⚠️  Repositório já existe${NC}"
else
    gh repo create kazuba/converter --public --source=. --push
    echo -e "${GREEN}✅ Repositório criado${NC}"
fi

echo ""

# Verificar remote
echo -e "${BLUE}🔗 Configurando remote...${NC}"
if ! git remote | grep -q origin; then
    git remote add origin https://github.com/kazuba/converter.git
    echo "Remote adicionado"
fi

# Push
echo -e "${BLUE}📤 Fazendo push...${NC}"
git push -u origin main || git push -u origin master
echo -e "${GREEN}✅ Push completo${NC}"
echo ""

# Criar release no GitHub
echo -e "${BLUE}🏷️  Criando release v0.1.0...${NC}"
gh release create v0.1.0 \
    --title "CONVERTER v0.1.0" \
    --notes "Primeira release do CONVERTER

Baseado em Batch2MD v7.2 (3.190 linhas, validado em produção ANTT)

Features:
- Conversão PDF, DOCX, XLSX, HTML → Markdown
- OCR Cascade: PaddleOCR → EasyOCR → Tesseract
- Extração de tabelas com Docling
- Suporte a 5 níveis de ZIP aninhado
- GPU-accelerated (CUDA/MPS/CPU)

Instalação: pip install converter" \
    || echo -e "${YELLOW}⚠️  Release pode já existir${NC}"

echo -e "${GREEN}✅ GitHub completo${NC}"
echo ""

# PyPI
echo "═══════════════════════════════════════════════════════════════"
echo "  PUBLISH PyPI"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo -e "${BLUE}📦 Verificando build...${NC}"
if [ ! -f "dist/converter-0.1.0-py3-none-any.whl" ]; then
    echo "Buildando..."
    uv build --wheel
fi
echo -e "${GREEN}✅ Build verificado${NC}"
echo ""

echo -e "${BLUE}🔑 Verificando credenciais PyPI...${NC}"
if ! twine check dist/converter-0.1.0-py3-none-any.whl &> /dev/null; then
    echo "⚠️  Problema no pacote"
    exit 1
fi

echo -e "${BLUE}📤 Publicando no PyPI...${NC}"
echo "Você precisará do token PyPI"
echo ""
echo "Comando: twine upload dist/converter-0.1.0-py3-none-any.whl"
echo ""
read -p "Deseja publicar agora? (s/n): " confirm

if [ "$confirm" = "s" ] || [ "$confirm" = "S" ]; then
    twine upload dist/converter-0.1.0-py3-none-any.whl
    echo -e "${GREEN}✅ Publicado no PyPI${NC}"
else
    echo -e "${YELLOW}⏸️  Publicação PyPI adiada${NC}"
    echo "Execute manualmente: twine upload dist/converter-0.1.0-py3-none-any.whl"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}PHASE 1 COMPLETA${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "GitHub: https://github.com/kazuba/converter"
echo "PyPI:   https://pypi.org/project/converter/ (após publicação)"
echo ""
echo "Instalação: pip install converter"
echo ""
