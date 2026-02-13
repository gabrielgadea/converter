#!/usr/bin/env python3
"""CLI do CONVERTER — Interface para Batch2MD v7.2

Uso:
    converter documento.pdf ./output/
    converter pasta/ ./output/ --ocr --workers 8
    converter arquivo.zip ./output/ --extract-nested
"""

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from pathlib import Path
import sys
import os

# Adicionar src ao path (para desenvolvimento local)
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Importar do core (Batch2MD v7.2 adaptado)
try:
    from core import BatchConverter
except ImportError:
    # Fallback para instalação via pip
    from converter.core import BatchConverter

console = Console()

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option(
    '--ocr', 
    is_flag=True, 
    help='Forçar OCR para PDFs escaneados'
)
@click.option(
    '--gpu/--no-gpu', 
    default=True, 
    help='Usar GPU se disponível (padrão: True)'
)
@click.option(
    '--tables/--no-tables', 
    default=True, 
    help='Extrair tabelas (padrão: True)'
)
@click.option(
    '--workers', 
    default=4, 
    type=int,
    help='Número de workers paralelos (padrão: 4)'
)
@click.option(
    '--extract-nested', 
    is_flag=True,
    help='Extrair ZIPs aninhados recursivamente'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Modo verbose com logs detalhados'
)
@click.version_option(
    version='0.1.0',
    prog_name='converter'
)
def main(input_path, output_path, ocr, gpu, tables, workers, extract_nested, verbose):
    """
    Converta documentos (PDF, DOCX, XLSX, HTML, ZIP) para Markdown estruturado.
    
    O CONVERTER preserva a estrutura semântica dos documentos, tornando-os
    legíveis por qualquer modelo de linguagem (GPT, Claude, Gemini, etc).
    
    \b
    Exemplos:
    
        \b
        # Converter um PDF
        converter documento.pdf ./output/
        
        \b
        # Converter com OCR (para PDFs escaneados)
        converter documento.pdf ./output/ --ocr
        
        \b
        # Converter pasta inteira (batch)
        converter ./pasta_documentos/ ./output/ --workers 8
        
        \b
        # Extrair e converter ZIP
        converter arquivo.zip ./output/ --extract-nested
    
    Para mais informações: https://github.com/kazuba/converter
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Criar diretório de saída se não existir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Banner
    console.print(Panel.fit(
        "[bold blue]CONVERTER[/bold blue] — Liberte seu conhecimento\n"
        "[dim]Baseado em Batch2MD v7.2 — 3.190 linhas validadas[/dim]",
        border_style="blue"
    ))
    
    console.print(f"📄 Input: {input_path}")
    console.print(f"📁 Output: {output_path}")
    console.print(f"⚙️  Config: OCR={ocr}, GPU={gpu}, Tables={tables}, Workers={workers}")
    console.print()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=not verbose
        ) as progress:
            
            task = progress.add_task("Convertendo...", total=None)
            
            # Inicializar converter (Batch2MD v7.2)
            converter = BatchConverter(
                use_ocr=ocr,
                use_gpu=gpu,
                extract_tables=tables,
                max_workers=workers,
                extract_nested=extract_nested,
                verbose=verbose
            )
            
            # Processar
            result = converter.process(input_path, output_path)
            
            progress.update(task, completed=True)
        
        # Resultado
        console.print()
        console.print(Panel.fit(
            f"✅ [bold green]Conversão completa![/bold green]\n\n"
            f"📊 Arquivos processados: {result.get('files_processed', 'N/A')}\n"
            f"📄 Páginas convertidas: {result.get('pages_processed', 'N/A')}\n"
            f"⏱️  Tempo: {result.get('elapsed_time', 'N/A')}\n"
            f"📁 Saída: {output_path}",
            border_style="green"
        ))
        
        console.print()
        console.print("[dim]💡 Dica: Use 'converter --help' para ver todas as opções[/dim]")
        
        return 0
        
    except Exception as e:
        console.print()
        console.print(Panel.fit(
            f"❌ [bold red]Erro na conversão[/bold red]\n\n"
            f"{str(e)}\n\n"
            f"[dim]Use --verbose para ver o traceback completo[/dim]",
            border_style="red"
        ))
        
        if verbose:
            import traceback
            console.print_exception()
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
