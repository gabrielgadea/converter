# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-12

### Added
- Initial release of CONVERTER
- Complete packaging of Batch2MD v7.2 (3,190 lines of production-validated code)
- CLI interface with Click and Rich for beautiful output
- Support for multiple document formats: PDF, DOCX, XLSX, HTML, ZIP
- OCR Cascade: PaddleOCR → EasyOCR → Tesseract with automatic fallback
- Table extraction using Docling TableFormer (97.9% accuracy)
- Nested ZIP support (up to 5 levels deep)
- GPU acceleration with CUDA/MPS/CPU fallback
- Batch processing with parallel workers
- Beautiful progress indicators and status reporting

### Technical
- Based on 2+ years of production use at ANTT
- Optimized for Google Colab (A100/V100/T4)
- Zero-LLM architecture (no API keys needed)
- MIT License

[0.1.0]: https://github.com/kazuba/converter/releases/tag/v0.1.0
