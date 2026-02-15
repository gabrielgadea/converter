"""CONVERTER — Liberte seu conhecimento de documentos.

Baseado em Batch2MD v7.2 (3.190 linhas, validado em produção ANTT desde 2025-12).
"""

__version__ = "0.2.3"
__author__ = "Gabriel Gadêa"
__license__ = "MIT"

# Importar classe principal do core.py (adaptado de Batch2MD-v7.2)
# Nota: A classe será adaptada para ser importável como módulo

__all__ = ["BatchConverter"]

# Lazy import para evitar carregar dependências pesadas na importação
def __getattr__(name):
    if name == "BatchConverter":
        from .core import BatchConverter
        return BatchConverter
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
