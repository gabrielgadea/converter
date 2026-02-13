# Colab Híbrido Rust + Python

**Link público:** https://colab.research.google.com/drive/1AjNkcLnar1JzEx8-JVpdDsHXodNmvv7b?usp=sharing

---

## 📋 Visão Geral

O **Colab Híbrido Rust + Python** é uma arquitetura de processamento de documentos que combina o melhor dos dois mundos:

- **Rust** → Performance crítica, paralelização com Rayon, segurança de memória
- **Python** → Ecossistema rico (pymupdf4llm, pandoc, OCR), prototipagem rápida

Este notebook é ideal para processamento em escala no Google Colab, especialmente quando você tem acesso a GPUs A100/V100.

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    GOOGLE COLAB                              │
│  (A100/V100/T4 GPU + 12-80GB VRAM + 85-150GB RAM)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌─────────────────────────────┐   │
│  │   RUST       │◄───────►│      PYTHON WORKERS         │   │
│  │ ORCHESTRATOR │  PyO3   │                             │   │
│  │              │         │  • pymupdf4llm (PDF→MD)     │   │
│  │  • Rayon     │         │  • pandoc (DOCX→MD)         │   │
│  │  • Paralelo  │         │  • tesseract (OCR)          │   │
│  │  • Seguro    │         │  • beautifulsoup (HTML)     │   │
│  └──────────────┘         └─────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              GOOGLE DRIVE (Input/Output)             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Funcionalidades

### 1. **Orquestrador Rust (Rayon)**
- Paralelização automática de tarefas I/O-bound
- Load balancing dinâmico entre workers
- Zero-cost abstractions (performance nativa)

### 2. **Workers Python Especializados**
| Worker | Função | Biblioteca |
|--------|--------|------------|
| PDF Extractor | Texto + layout | pymupdf4llm |
| Office Converter | DOCX/XLSX/PPTX | pandoc + python-docx |
| OCR Engine | PDFs escaneados | tesseract + pytesseract |
| HTML Parser | Web→Markdown | beautifulsoup4 |
| MSG Reader | E-mails Outlook | extract-msg |

### 3. **Pipeline de 5 Etapas**
```
Input → Detecção → Extração → Normalização → Output
              ↓           ↓              ↓
         (Rust)     (Python)        (Rust)
       FileType   Conversion    Post-process
```

### 4. **Fallback Inteligente**
- Se pymupdf4llm falhar → tenta pdfplumber
- Se OCR falhar → retry com parâmetros diferentes
- Se Rust panics → fallback para Python puro

---

## 🚀 Vantagens vs. CONVERTER Local (Python)

| Aspecto | Colab Híbrido | CONVERTER Local |
|---------|--------------|-----------------|
| **Hardware** | A100/V100 (80GB VRAM) | Sua máquina (RTX 4060 8GB) |
| **CPU** | 8-12 cores Xeon | i9-14900HX (24 cores) |
| **RAM** | 85-150GB | 64GB |
| **Paralelismo** | Rayon (Rust) | multiprocessing (Python) |
| **Cold Start** | ~2 min (instalação) | Instantâneo |
| **Persistência** | Drive/Google Cloud | Local filesystem |
| **Custo** | Gratuito/Pro | $0 (sua máquina) |
| **Offline** | ❌ Não | ✅ Sim |
| **Batch Size** | 64-128 arquivos | 8-16 arquivos |
| **OCR GPU** | Sim (CUDA) | Sim (CUDA) |

---

## 📊 Quando Usar Cada Um

### Use o **Colab Híbrido** quando:
- ✅ Processar **>1000 arquivos** de uma vez
- ✅ Arquivos grandes (**>100MB cada**)
- ✅ Precisa de **GPU A100** para OCR em massa
- ✅ Não quer ocupar sua máquina por horas
- ✅ Quer processar enquanto trabalha em outra coisa
- ✅ Precisa de **RAM >64GB** para PDFs complexos

### Use o **CONVERTER Local** quando:
- ✅ Precisa de **resposta instantânea**
- ✅ Trabalha com **dados sensíveis** (offline)
- ✅ Desenvolvimento/iteração rápida
- ✅ Arquivos pequenos (<100 arquivos)
- ✅ Não tem internet estável
- ✅ Quer integrar em scripts locais

---

## 🔧 Como Usar o Colab Híbrido

### Passo 1: Abrir o Notebook
```
1. Acesse: https://colab.research.google.com/drive/1AjNkcLnar1JzEx8-JVpdDsHXodNmvv7b
2. Faça uma cópia para seu Drive: File → Save a copy in Drive
```

### Passo 2: Configurar Ambiente (Cell 1)
```python
# A célula detecta automaticamente:
# - Tipo de GPU (A100/V100/T4)
# - VRAM disponível
# - Otimiza batch_size e workers

# Para A100-80GB:
BATCH_SIZE = 64
WORKERS = 12

# Para T4 (free tier):
BATCH_SIZE = 16
WORKERS = 4
```

### Passo 3: Upload de Arquivos (Cell 2)
**Opção A: Google Drive**
```python
# Monte seu Drive
from google.colab import drive
drive.mount('/content/drive')

# Configure pasta de origem
DRIVE_SOURCE = '/content/drive/MyDrive/MeusDocumentos'
```

**Opção B: Upload Direto**
```python
from google.colab import files
uploaded = files.upload()  # Selecione arquivos
```

### Passo 4: Executar Conversão (Cell 3)
```python
# O pipeline Rust+Python processa automaticamente:
# 1. Detecta tipo de arquivo (Rust)
# 2. Seleciona worker apropriado (Rust)
# 3. Executa conversão (Python)
# 4. Normaliza saída (Rust)
# 5. Salva em Drive/output

result = pipeline.process(
    input_dir='/content/input',
    output_dir='/content/output',
    enable_ocr=True,
    extract_tables=True
)
```

### Passo 5: Download (Cell 4)
```python
# Compacta e faz download
!zip -r output.zip /content/output
files.download('output.zip')
```

---

## 📈 Performance Comparada

### Cenário: 100 PDFs de 50MB cada

| Métrica | Colab Híbrido (A100) | CONVERTER Local (RTX 4060) |
|---------|---------------------|---------------------------|
| **Tempo total** | ~8 minutos | ~25 minutos |
| **Throughput** | 12.5 PDFs/min | 4 PDFs/min |
| **VRAM usada** | ~40GB | ~6GB (limite) |
| **RAM usada** | ~60GB | ~32GB |
| **Energia** | Google paga | Seu PC |

### Cenário: 10 PDFs de 5MB cada

| Métrica | Colab Híbrido | CONVERTER Local |
|---------|--------------|-----------------|
| **Setup** | 2 min | 0 min |
| **Processamento** | 30 seg | 45 seg |
| **Total** | 2.5 min | 45 seg |
| **Vencedor** | ❌ | ✅ |

**Conclusão:** Colab vale a pena para batches grandes (>50 arquivos).

---

## 🔬 Detalhes Técnicos

### PyO3 Integration
```rust
// Rust expõe funções para Python
#[pyfunction]
fn process_batch(files: Vec<String>) -> PyResult<Vec<ConversionResult>> {
    // Rayon parallelizes across all CPU cores
    let results: Vec<_> = files
        .par_iter()
        .map(|f| convert_file(f))
        .collect();
    Ok(results)
}
```

### Python Worker Pattern
```python
# Worker especializado em PDFs
class PDFWorker:
    def __init__(self):
        self.doc = fitz.open()
    
    def convert(self, path: str) -> str:
        # Usa pymupdf4llm com GPU se disponível
        return pymupdf4llm.to_markdown(path)
```

---

## 🛠️ Troubleshooting

| Problema | Solução |
|----------|---------|
| "CUDA out of memory" | Reduzir BATCH_SIZE para 8 |
| "Rust compilation failed" | Restart runtime (Runtime → Restart) |
| "Drive not mounting" | Re-autorizar em outra aba |
| "Tesseract not found" | Re-executar Cell 1 |
| Timeout no download | Usar Drive ao invés de download direto |

---

## 🔄 Sincronia com CONVERTER Local

Você pode usar os dois em conjunto:

```
1. Desenvolva/teste localmente com CONVERTER
2. Quando pronto, escale no Colab Híbrido
3. Resultados voltam para seu Drive
4. Continue trabalhando localmente
```

**Fluxo ideal:**
1. **Protótipo** → CONVERTER local (rápido)
2. **Validação** → Colab Híbrido (batch médio)
3. **Produção** → Colab Híbrido (batch grande)
4. **Integração** → CONVERTER local (pipeline contínuo)

---

## 📚 Recursos

- **Notebook:** https://colab.research.google.com/drive/1AjNkcLnar1JzEx8-JVpdDsHXodNmvv7b
- **Documentação Rust:** `packages/kazuba-rust-core/`
- **Documentação Python:** `docs/reference/kazuba-converters.md`
- **Comparação de formatos:** `docs/why-formats-matter.md`

---

## 📝 Resumo

| | Colab Híbrido | CONVERTER Local |
|--|--------------|-----------------|
| **Melhor para** | Escala, GPU pesada | Velocidade, privacidade |
| **Hardware** | Cloud (A100) | Local (RTX 4060) |
| **Setup** | 2 min | Instantâneo |
| **Custo** | Gratuito/Pro | $0 |
| **Offline** | Não | Sim |

**Use os dois!** Colab para processamento pesado, CONVERTER para trabalho diário.

---

*Atualizado: 2026-02-12*
