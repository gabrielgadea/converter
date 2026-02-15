#!/usr/bin/env python3
"""CLI do CONVERTER — Interface para Batch2MD v7.2

Uso:
    converter convert documento.pdf ./output/
    converter batch ./pasta/ ./output/ --workers 8
    converter --help
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

console = Console()

@click.group()
@click.version_option(version='0.2.3', prog_name='kazuba-converter')
def cli():
    """CONVERTER — Liberte seu conhecimento de documentos."""
    pass

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--ocr', is_flag=True, help='Forçar OCR para PDFs escaneados')
@click.option('--gpu/--no-gpu', default=True, help='Usar GPU se disponível (padrão: True)')
@click.option('--tables/--no-tables', default=True, help='Extrair tabelas (padrão: True)')
@click.option('--workers', '-w', default=4, help='Número de workers paralelos (padrão: 4)')
@click.option('--verbose', '-v', is_flag=True, help='Modo verbose com logs detalhados')
@click.option('--extract-nested', is_flag=True, help='Extrair ZIPs aninhados')
def convert(input_path, output_path, ocr, gpu, tables, workers, verbose, extract_nested):
    """Converte documentos para Markdown (modo simples)."""
    # Importar aqui para evitar overhead no startup
    try:
        from converter.core import BatchConverter
    except ImportError:
        from core import BatchConverter
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        "[bold blue]CONVERTER[/bold blue] — Liberte seu conhecimento\n"
        "[dim]Batch2MD v7.2 — Conversão individual[/dim]",
        border_style="blue"
    ))
    
    console.print(f"📄 Input: {input_path}")
    console.print(f"📁 Output: {output_path}")
    console.print(f"⚙️  OCR={ocr}, GPU={gpu}, Tables={tables}, Workers={workers}")
    console.print()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=not verbose
        ) as progress:
            
            task = progress.add_task("Convertendo...", total=None)
            
            converter = BatchConverter(
                use_ocr=ocr,
                use_gpu=gpu,
                extract_tables=tables,
                max_workers=workers,
                extract_nested=extract_nested,
                verbose=verbose
            )
            
            result = converter.process(input_path, output_path)
            progress.update(task, completed=True)
        
        console.print()
        console.print(Panel.fit(
            f"✅ [bold green]Conversão completa![/bold green]\n\n"
            f"📊 Arquivos: {result.get('files_processed', 'N/A')}\n"
            f"📄 Páginas: {result.get('pages_processed', 'N/A')}\n"
            f"⏱️  Tempo: {result.get('elapsed_time', 'N/A')}",
            border_style="green"
        ))
        
        return 0
        
    except Exception as e:
        console.print()
        console.print(Panel.fit(
            f"❌ [bold red]Erro:[/bold red] {str(e)}\n"
            f"[dim]Use --verbose para traceback[/dim]",
            border_style="red"
        ))
        if verbose:
            import traceback
            console.print_exception()
        return 1

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('--workers', '-w', type=int, default=None, help='Número de workers (padrão: auto)')
@click.option('--pattern', '-p', default='*.pdf', help='Padrão de arquivos')
@click.option('--checkpoint', '-c', default='checkpoint.json', help='Arquivo de checkpoint')
@click.option('--reset', is_flag=True, help='Resetar checkpoint e reprocessar tudo')
@click.option('--verbose', '-v', is_flag=True, help='Modo verbose')
def batch(input_dir, output_dir, workers, pattern, checkpoint, reset, verbose):
    """
    Processa batch com worker pool persistente e checkpoint/resume.
    
    Ideal para jobs grandes (1000+ arquivos) e execuções longas.
    Elimina overhead de subprocess ao manter workers rodando.
    """
    from converter.batch_processor import BatchProcessor
    
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel.fit(
        "[bold blue]CONVERTER BATCH[/bold blue] — Worker Pool Persistente\n"
        "[dim]Com checkpoint/resume para jobs longos[/dim]",
        border_style="blue"
    ))
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    processor = BatchProcessor(
        input_dir=input_path,
        output_dir=output_path,
        num_workers=workers,
        checkpoint_file=Path(checkpoint)
    )
    
    if reset:
        if click.confirm("⚠️  Isso apagará o progresso salvo. Continuar?"):
            processor.checkpoint.reset()
        else:
            console.print("Cancelado.")
            return 0
    
    stats = processor.process_batch(pattern=pattern)
    
    console.print()
    console.print(Panel.fit(
        f"✅ [bold green]Batch completo![/bold green]\n\n"
        f"📊 Processados: {stats['completed']}\n"
        f"❌ Falhas: {stats['failed']}\n"
        f"⏭️  Pulados: {stats['skipped']}\n"
        f"⏱️  Tempo: {stats['total_time']:.1f}s\n"
        f"⚡ Throughput: {stats['completed'] / (stats['total_time'] / 60):.1f} arquivos/min",
        border_style="green"
    ))
    
    return 0 if stats['failed'] == 0 else 1

# Compatibilidade com chamadas antigas
main = cli

if __name__ == '__main__':
    cli()
