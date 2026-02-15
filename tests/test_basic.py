"""Testes básicos de integração para CONVERTER.

Nota: O código Batch2MD v7.2 já foi validado em produção (ANTT, desde 2025-12).
Estes testes verificam empacotamento e CLI, não a lógica de conversão.
"""

import pytest
from pathlib import Path
import sys

# Adicionar src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestImports:
    """Testes de importação."""
    
    def test_module_imports(self):
        """O módulo converter importa corretamente."""
        import converter
        assert converter.__version__ == "0.2.3"
    
    def test_core_imports(self):
        """O core (Batch2MD v7.2) importa corretamente."""
        # Nota: Este teste pode falhar se dependências pesadas não estiverem instaladas
        # É aceitável — o core é opcional no import
        try:
            from converter.core import BatchConverter
            assert BatchConverter is not None
        except ImportError as e:
            pytest.skip(f"Dependências do core não instaladas: {e}")


class TestCLI:
    """Testes de interface de linha de comando."""
    
    def test_cli_imports(self):
        """O CLI importa corretamente."""
        from converter.cli import main
        assert main is not None
    
    def test_cli_help(self):
        """CLI mostra ajuda sem erros."""
        from click.testing import CliRunner
        from converter.cli import main
        
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "CONVERTER" in result.output
        assert "--version" in result.output
    
    def test_cli_version(self):
        """CLI mostra versão."""
        from click.testing import CliRunner
        from converter.cli import main
        
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert "0.2.3" in result.output


class TestPackageStructure:
    """Testes de estrutura do pacote."""
    
    def test_src_structure(self):
        """Estrutura de diretórios está correta."""
        src_path = Path(__file__).parent.parent / "src" / "converter"
        
        assert src_path.exists()
        assert (src_path / "__init__.py").exists()
        assert (src_path / "cli.py").exists()
        assert (src_path / "core.py").exists()


class TestPyProject:
    """Testes de configuração do pacote."""
    
    def test_pyproject_exists(self):
        """pyproject.toml existe."""
        project_root = Path(__file__).parent.parent
        assert (project_root / "pyproject.toml").exists()
    
    def test_readme_exists(self):
        """README.md existe."""
        project_root = Path(__file__).parent.parent
        assert (project_root / "README.md").exists()
