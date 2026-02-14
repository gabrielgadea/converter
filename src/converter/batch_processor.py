#!/usr/bin/env python3
"""
Batch Processor com Worker Pool Persistente e Checkpoint/Resume

Elimina overhead de subprocess ao manter workers Python rodando como daemons.
Implementa checkpoint/resume para jobs longos (resiliência a desconexões).
"""

import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from queue import Empty
import signal
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConversionJob:
    """Representa um job de conversão."""
    file_path: str
    output_dir: str
    job_id: str
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConversionJob':
        return cls(**data)


@dataclass
class ConversionResult:
    """Resultado de uma conversão."""
    job_id: str
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)


class CheckpointManager:
    """Gerencia checkpoint/resume para jobs longos."""
    
    def __init__(self, checkpoint_file: Path = Path("conversion_checkpoint.json")):
        self.checkpoint_file = checkpoint_file
        self.processed_files: Set[str] = set()
        self.failed_files: Dict[str, str] = {}  # path -> error
        self.load()
    
    def load(self):
        """Carrega checkpoint existente."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed', []))
                    self.failed_files = data.get('failed', {})
                logger.info(f"📂 Checkpoint carregado: {len(self.processed_files)} processados, "
                           f"{len(self.failed_files)} falhas")
            except Exception as e:
                logger.warning(f"⚠️  Erro ao carregar checkpoint: {e}")
                self.processed_files = set()
                self.failed_files = {}
    
    def save(self):
        """Salva checkpoint atual."""
        data = {
            'processed': list(self.processed_files),
            'failed': self.failed_files,
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(self.processed_files),
            'total_failed': len(self.failed_files)
        }
        try:
            # Salvar em arquivo temporário primeiro (atomic write)
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.checkpoint_file)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar checkpoint: {e}")
    
    def mark_processed(self, file_path: str):
        """Marca arquivo como processado."""
        self.processed_files.add(file_path)
        self.failed_files.pop(file_path, None)  # Remove de failed se existir
        self.save()
    
    def mark_failed(self, file_path: str, error: str):
        """Marca arquivo como falho."""
        self.failed_files[file_path] = error
        self.save()
    
    def is_processed(self, file_path: str) -> bool:
        """Verifica se arquivo já foi processado."""
        return file_path in self.processed_files
    
    def reset(self):
        """Reseta checkpoint (cuidado!)."""
        self.processed_files = set()
        self.failed_files = {}
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        logger.info("🔄 Checkpoint resetado")


def worker_process(
    worker_id: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    convert_func: Callable
):
    """
    Processo worker que roda continuamente, processando jobs da fila.
    
    Args:
        worker_id: ID único do worker
        job_queue: Fila de jobs a processar
        result_queue: Fila de resultados
        stop_event: Evento para sinalizar parada
        convert_func: Função de conversão (do core.py)
    """
    logger.info(f"👷 Worker {worker_id} iniciado")
    
    # Ignorar SIGINT no worker (deixar o pai lidar)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    while not stop_event.is_set():
        try:
            # Pegar job da fila (timeout para verificar stop_event periodicamente)
            job = job_queue.get(timeout=1.0)
            
            if job is None:  # Sinal de término
                break
            
            if not isinstance(job, ConversionJob):
                job = ConversionJob.from_dict(job)
            
            logger.debug(f"Worker {worker_id} processando: {job.file_path}")
            
            start_time = time.time()
            
            try:
                # Executar conversão
                result = convert_func(
                    Path(job.file_path),
                    Path(job.output_dir)
                )
                
                processing_time = time.time() - start_time
                
                conversion_result = ConversionResult(
                    job_id=job.job_id,
                    success=True,
                    output_path=str(result) if result else None,
                    processing_time=processing_time
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                conversion_result = ConversionResult(
                    job_id=job.job_id,
                    success=False,
                    error_message=str(e),
                    processing_time=processing_time
                )
            
            # Enviar resultado
            result_queue.put(conversion_result.to_dict())
            
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id} erro: {e}")
    
    logger.info(f"👷 Worker {worker_id} encerrado")


class PersistentWorkerPool:
    """Pool de workers persistentes (eliminam overhead de subprocess)."""
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        convert_func: Optional[Callable] = None
    ):
        self.num_workers = num_workers or max(2, mp.cpu_count())
        self.convert_func = convert_func or self._default_convert
        self.workers: List[mp.Process] = []
        self.job_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.stop_event = mp.Event()
        self.checkpoint = CheckpointManager()
        
    def _default_convert(self, input_path: Path, output_dir: Path) -> Path:
        """Função de conversão padrão (importa do core)."""
        try:
            from converter.core import process_single_file
            return process_single_file(str(input_path), str(output_dir))
        except ImportError:
            # Fallback simples se core não estiver disponível
            logger.warning("Core não disponível, usando fallback")
            output_path = output_dir / f"{input_path.stem}.md"
            output_path.write_text(f"# Converted: {input_path.name}\n\n(Fallback)")
            return output_path
    
    def start(self):
        """Inicia o pool de workers."""
        logger.info(f"🚀 Iniciando pool com {self.num_workers} workers")
        
        self.job_queue = mp.Queue(maxsize=self.num_workers * 2)
        self.result_queue = mp.Queue()
        self.stop_event.clear()
        
        # Criar workers
        for i in range(self.num_workers):
            worker = mp.Process(
                target=worker_process,
                args=(
                    i,
                    self.job_queue,
                    self.result_queue,
                    self.stop_event,
                    self.convert_func
                )
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"✅ {self.num_workers} workers iniciados")
    
    def stop(self):
        """Para o pool de workers."""
        logger.info("🛑 Parando pool de workers")
        self.stop_event.set()
        
        # Enviar sinal de término para cada worker
        for _ in self.workers:
            try:
                self.job_queue.put(None, timeout=1.0)
            except:
                pass
        
        # Aguardar término
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning(f"Worker {worker.pid} não respondeu, terminando")
                worker.terminate()
        
        self.workers = []
        logger.info("✅ Pool encerrado")
    
    def submit_job(self, job: ConversionJob) -> bool:
        """Submete um job para processamento."""
        try:
            self.job_queue.put(job.to_dict(), timeout=5.0)
            return True
        except:
            return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[ConversionResult]:
        """Obtém um resultado da fila."""
        try:
            result_dict = self.result_queue.get(timeout=timeout)
            return ConversionResult(**result_dict)
        except Empty:
            return None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class BatchProcessor:
    """Processador em lote com checkpoint e worker pool persistente."""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        num_workers: Optional[int] = None,
        checkpoint_file: Optional[Path] = None
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers or max(2, mp.cpu_count())
        self.checkpoint = CheckpointManager(checkpoint_file or Path("checkpoint.json"))
        self.stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0.0
        }
    
    def collect_files(self, pattern: str = "*") -> List[Path]:
        """Coleta arquivos a processar."""
        files = list(self.input_dir.rglob(pattern))
        # Filtrar por extensões suportadas
        supported = {'.pdf', '.docx', '.xlsx', '.pptx', '.html', '.msg'}
        files = [f for f in files if f.suffix.lower() in supported]
        return sorted(files)
    
    def process_batch(
        self,
        files: Optional[List[Path]] = None,
        pattern: str = "*.pdf",
        progress_callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict:
        """
        Processa batch de arquivos com checkpoint/resume.
        
        Args:
            files: Lista específica de arquivos (opcional)
            pattern: Padrão de busca se files não fornecido
            progress_callback: Função chamada a cada progresso
            
        Returns:
            Estatísticas do processamento
        """
        if files is None:
            files = self.collect_files(pattern)
        
        # Filtrar já processados
        pending_files = [
            f for f in files 
            if not self.checkpoint.is_processed(str(f))
        ]
        
        self.stats['skipped'] = len(files) - len(pending_files)
        
        logger.info(f"📁 Total: {len(files)} | "
                   f"⏳ Pendentes: {len(pending_files)} | "
                   f"⏭️  Pulados: {self.stats['skipped']}")
        
        if not pending_files:
            logger.info("✅ Nenhum arquivo pendente!")
            return self.stats
        
        start_time = time.time()
        
        # Usar worker pool persistente
        with PersistentWorkerPool(self.num_workers) as pool:
            # Submeter todos os jobs - mapear job_id -> file_path
            jobs_pending = 0
            job_to_file: Dict[str, str] = {}  # Mapeamento job_id -> file_path
            for i, file_path in enumerate(pending_files):
                job_id = f"job_{i:06d}"
                job = ConversionJob(
                    file_path=str(file_path),
                    output_dir=str(self.output_dir),
                    job_id=job_id
                )
                job_to_file[job_id] = str(file_path)
                
                if pool.submit_job(job):
                    jobs_pending += 1
                    self.stats['submitted'] += 1
                else:
                    logger.warning(f"⚠️  Falha ao submeter: {file_path}")
            
            logger.info(f"📤 {jobs_pending} jobs submetidos")
            
            # Coletar resultados
            results_received = 0
            while results_received < jobs_pending:
                result = pool.get_result(timeout=0.5)
                
                if result:
                    results_received += 1
                    
                    # Obter file_path original do mapeamento
                    original_file = job_to_file.get(result.job_id, result.job_id)
                    
                    if result.success:
                        self.stats['completed'] += 1
                        self.checkpoint.mark_processed(original_file)
                        logger.debug(f"✅ {result.job_id}: {result.output_path}")
                    else:
                        self.stats['failed'] += 1
                        self.checkpoint.mark_failed(original_file, result.error_message or "Unknown")
                        logger.error(f"❌ {result.job_id}: {result.error_message}")
                    
                    # Callback de progresso
                    if progress_callback:
                        progress_callback({
                            'completed': self.stats['completed'],
                            'failed': self.stats['failed'],
                            'total': jobs_pending,
                            'percent': (results_received / jobs_pending) * 100
                        })
                
                # Log de progresso a cada 10%
                if results_received % max(1, jobs_pending // 10) == 0:
                    percent = (results_received / jobs_pending) * 100
                    logger.info(f"⏳ Progresso: {percent:.1f}% ({results_received}/{jobs_pending})")
        
        self.stats['total_time'] = time.time() - start_time
        
        # Log final
        logger.info("=" * 60)
        logger.info("📊 RESUMO DO PROCESSAMENTO")
        logger.info("=" * 60)
        logger.info(f"✅ Completados: {self.stats['completed']}")
        logger.info(f"❌ Falhas: {self.stats['failed']}")
        logger.info(f"⏭️  Pulados (já processados): {self.stats['skipped']}")
        logger.info(f"⏱️  Tempo total: {self.stats['total_time']:.1f}s")
        if self.stats['completed'] > 0:
            logger.info(f"⚡ Throughput: {self.stats['completed'] / (self.stats['total_time'] / 60):.1f} arquivos/min")
        logger.info("=" * 60)
        
        return self.stats


def main():
    """CLI para o batch processor."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch Processor com Worker Pool Persistente'
    )
    parser.add_argument('input_dir', help='Diretório de entrada')
    parser.add_argument('output_dir', help='Diretório de saída')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Número de workers (padrão: CPU count)')
    parser.add_argument('--pattern', '-p', default='*.pdf',
                       help='Padrão de arquivos (padrão: *.pdf)')
    parser.add_argument('--reset', action='store_true',
                       help='Resetar checkpoint e reprocessar tudo')
    parser.add_argument('--checkpoint', '-c', default='checkpoint.json',
                       help='Arquivo de checkpoint')
    
    args = parser.parse_args()
    
    processor = BatchProcessor(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        num_workers=args.workers,
        checkpoint_file=Path(args.checkpoint)
    )
    
    if args.reset:
        processor.checkpoint.reset()
    
    stats = processor.process_batch(pattern=args.pattern)
    
    # Exit code baseado em sucesso
    sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
