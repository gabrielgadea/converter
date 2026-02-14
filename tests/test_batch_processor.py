"""Testes para batch_processor."""

import pytest
import tempfile
from pathlib import Path
import time
from converter.batch_processor import (
    ConversionJob, ConversionResult,
    CheckpointManager, PersistentWorkerPool, BatchProcessor
)


class TestConversionJob:
    """Testes para ConversionJob."""
    
    def test_job_creation(self):
        job = ConversionJob(
            file_path="/input/test.pdf",
            output_dir="/output",
            job_id="job_001"
        )
        assert job.file_path == "/input/test.pdf"
        assert job.job_id == "job_001"
        assert job.retries == 0
    
    def test_job_serialization(self):
        job = ConversionJob(
            file_path="/input/test.pdf",
            output_dir="/output",
            job_id="job_001"
        )
        data = job.to_dict()
        restored = ConversionJob.from_dict(data)
        assert restored.file_path == job.file_path
        assert restored.job_id == job.job_id


class TestCheckpointManager:
    """Testes para CheckpointManager."""
    
    def test_checkpoint_save_load(self, tmp_path):
        checkpoint_file = tmp_path / "test_checkpoint.json"
        manager = CheckpointManager(checkpoint_file)
        
        # Marcar alguns arquivos
        manager.mark_processed("/file1.pdf")
        manager.mark_processed("/file2.pdf")
        manager.mark_failed("/file3.pdf", "error message")
        
        # Recriar manager (simula reload)
        manager2 = CheckpointManager(checkpoint_file)
        assert manager2.is_processed("/file1.pdf")
        assert manager2.is_processed("/file2.pdf")
        assert not manager2.is_processed("/file3.pdf")  # Failed != processed
        assert "/file3.pdf" in manager2.failed_files
    
    def test_checkpoint_reset(self, tmp_path):
        checkpoint_file = tmp_path / "test_checkpoint.json"
        manager = CheckpointManager(checkpoint_file)
        
        manager.mark_processed("/file1.pdf")
        assert manager.is_processed("/file1.pdf")
        
        manager.reset()
        assert not manager.is_processed("/file1.pdf")
        assert not checkpoint_file.exists()


class TestBatchProcessor:
    """Testes para BatchProcessor."""
    
    def test_collect_files(self, tmp_path):
        # Criar arquivos de teste
        (tmp_path / "file1.pdf").touch()
        (tmp_path / "file2.pdf").touch()
        (tmp_path / "file3.docx").touch()
        (tmp_path / "ignore.txt").touch()
        
        processor = BatchProcessor(tmp_path, tmp_path / "output")
        files = processor.collect_files("*")
        
        # Deve encontrar apenas os suportados
        assert len(files) == 3  # 2 PDFs + 1 DOCX
    
    def test_skip_processed_files(self, tmp_path):
        # Criar arquivos
        pdf_file = tmp_path / "file1.pdf"
        pdf_file.touch()
        
        # Marcar como processado
        processor = BatchProcessor(tmp_path, tmp_path / "output")
        processor.checkpoint.mark_processed(str(pdf_file))
        
        # Coletar deve ignorar o processado
        files = [pdf_file]
        pending = [f for f in files if not processor.checkpoint.is_processed(str(f))]
        assert len(pending) == 0


class TestIntegration:
    """Testes de integração simples."""
    
    def test_worker_pool_lifecycle(self):
        """Testa iniciar e parar o worker pool."""
        def dummy_convert(input_path, output_dir):
            return output_dir / "output.md"
        
        pool = PersistentWorkerPool(num_workers=2, convert_func=dummy_convert)
        
        # Iniciar
        pool.start()
        assert len(pool.workers) == 2
        assert all(w.is_alive() for w in pool.workers)
        
        # Parar
        pool.stop()
        assert len(pool.workers) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
