#!/usr/bin/env python3
"""
Benchmark para kazuba-converter
Mede throughput real em diferentes cenários
"""

import time
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime

def create_test_files(output_dir: Path, count: int = 10):
    """Cria arquivos de teste sintéticos (simulação)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Para benchmark real, você deve ter PDFs de teste
    # Aqui criamos um relatório simples
    print(f"⚠️  Nota: Benchmark requer arquivos PDF de teste em {output_dir}")
    print("   Coloque alguns PDFs no diretório e reexecute.")
    return []

def run_benchmark(input_dir: Path, output_dir: Path, workers: int = 4) -> dict:
    """Executa benchmark de conversão."""
    
    # Verificar arquivos de entrada
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        return {
            "error": "Nenhum PDF encontrado no diretório de entrada",
            "input_dir": str(input_dir),
            "timestamp": datetime.now().isoformat()
        }
    
    print(f"📁 Encontrados {len(pdf_files)} PDFs")
    print(f"⚙️  Workers configurados: {workers}")
    
    # Criar diretório de saída
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Medir tempo
    start_time = time.time()
    
    try:
        # Executar conversão via CLI
        result = subprocess.run(
            [
                sys.executable, "-m", "converter.cli",
                str(input_dir),
                str(output_dir),
                "--workers", str(workers)
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos timeout
        )
        
        elapsed = time.time() - start_time
        
        # Contar arquivos de saída
        output_files = list(output_dir.glob("*.md"))
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "input_files": len(pdf_files),
            "output_files": len(output_files),
            "elapsed_seconds": round(elapsed, 2),
            "workers": workers,
            "files_per_minute": round((len(pdf_files) / elapsed) * 60, 1),
            "success": result.returncode == 0,
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        if result.returncode != 0:
            metrics["error"] = result.stderr
        
        return metrics
        
    except subprocess.TimeoutExpired:
        return {
            "error": "Timeout (5 minutos)",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Executa benchmark completo."""
    print("=" * 60)
    print("📊 KAZUBA-CONVERTER BENCHMARK")
    print("=" * 60)
    
    # Detectar workers ótimo
    import os
    cpu_count = os.cpu_count() or 2
    workers_list = [2, 4, min(cpu_count, 8)]
    
    input_dir = Path("./benchmark_input")
    
    # Criar diretório de input se não existir
    if not input_dir.exists():
        input_dir.mkdir()
        print(f"\n⚠️  Diretório de input criado: {input_dir}")
        print("   Adicione PDFs de teste neste diretório e reexecute.")
        return
    
    results = []
    
    for workers in workers_list:
        output_dir = Path(f"./benchmark_output_{workers}w")
        print(f"\n🔧 Testando com {workers} workers...")
        
        # Limpar saída anterior
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        
        metrics = run_benchmark(input_dir, output_dir, workers)
        results.append(metrics)
        
        if "error" in metrics:
            print(f"   ❌ Erro: {metrics['error']}")
        else:
            print(f"   ✅ {metrics['files_per_minute']} arquivos/min")
    
    # Salvar resultados
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_runs": results,
        "recommendation": f"Use {min(cpu_count, 4)} workers para seu hardware"
    }
    
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Benchmark completo!")
    print(f"📄 Relatório salvo em: benchmark_report.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
