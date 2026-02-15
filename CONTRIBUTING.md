# Contributing to kazuba-converter

Obrigado por seu interesse em contribuir! 🎉

## Como Contribuir

### 1. Reportando Bugs

Use [GitHub Issues](https://github.com/gabrielgadea/converter/issues) com template:

```markdown
**Descrição:**
Descrição clara do bug

**Reprodução:**
1. Comando executado
2. Erro obtido
3. Comportamento esperado

**Ambiente:**
- OS: Ubuntu 22.04
- Python: 3.12
- Versão: 0.2.4
```

### 2. Sugestões de Features

Abra uma issue com label `enhancement` descrevendo:
- Problema que resolve
- API/CLI proposta
- Casos de uso

### 3. Pull Requests

1. Fork o repositório
2. Crie branch: `git checkout -b feature/nome-da-feature`
3. Commit: `git commit -m "feat: descrição"`
4. Push: `git push origin feature/nome-da-feature`
5. Abra PR para `main`

## Convenções de Código

### Commits (Conventional Commits)

```
feat: nova funcionalidade
fix: correção de bug
docs: documentação
style: formatação
test: testes
chore: manutenção
```

### Python

- PEP 8
- Type hints obrigatórios
- Docstrings (Google style)

```python
def convert_file(file_path: Path, ocr: bool = False) -> ConversionResult:
    """Converte arquivo para Markdown.
    
    Args:
        file_path: Caminho do arquivo
        ocr: Ativar OCR para PDFs escaneados
        
    Returns:
        Resultado da conversão
        
    Raises:
        FileNotFoundError: Arquivo não existe
    """
```

### Testes

```bash
# Rodar testes
pytest tests/ -v

# Com coverage
pytest tests/ --cov=src/converter --cov-report=html
```

## Áreas de Prioridade

1. **Testes** — Aumentar cobertura para 90%+
2. **Documentação** — Tutoriais e exemplos
3. **Performance** — Otimizações de OCR
4. **Plataformas** — Windows/macOS

## Licença

Contribuições são sob MIT License.
