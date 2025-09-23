# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MinerU is a tool for converting PDFs to machine-readable formats (markdown, JSON). It supports multiple parsing backends:
- **pipeline**: Traditional OCR-based parsing with layout analysis
- **vlm-transformers**: Vision-language model parsing using HuggingFace transformers
- **vlm-sglang**: Accelerated VLM parsing using SGLang

The project processes documents through different stages: preprocessing, analysis, content extraction, and output generation.

## Common Commands

### Installation & Environment Setup
```bash
# Install core features (recommended for development)
pip install --upgrade pip
pip install uv
uv pip install -e ".[core]"

# Install specific feature sets
uv pip install -e ".[pipeline]"  # OCR-based backend only
uv pip install -e ".[vlm]"       # VLM backend only
uv pip install -e ".[test]"      # Testing dependencies
uv pip install -e ".[all]"       # All features including SGLang

# Download required models (essential first step)
mineru-models-download
```

### Development & Testing
```bash
# Run all tests with coverage
pytest --cov=mineru --cov-report html

# Run specific test file
pytest tests/unittest/test_e2e.py

# Run coverage analysis and validate minimum threshold (20%)
python tests/get_coverage.py

# Clean coverage artifacts
python tests/clean_coverage.py

# Build package
python -m build

# Lint and format (no built-in linting - use external tools)
# Note: Project doesn't have built-in linting configuration
```

### Basic Usage & CLI
```bash
# Basic PDF parsing
mineru -p input.pdf -o output_directory

# Parse with specific backend
mineru -p input.pdf -o output_directory -b pipeline
mineru -p input.pdf -o output_directory -b vlm-transformers

# Parse with specific method (pipeline backend only)
mineru -p input.pdf -o output_directory -m auto|txt|ocr

# Parse page ranges
mineru -p input.pdf -o output_directory --start-page-id 0 --end-page-id 5
```

### Services & APIs
```bash
# Start FastAPI server (default port 8000)
mineru-api --host 0.0.0.0 --port 8000

# Start Gradio web interface (default port 7860)
mineru-gradio --server-name 0.0.0.0 --server-port 7860

# Start Gradio with SGLang engine enabled
mineru-gradio --server-name 0.0.0.0 --server-port 7860 --enable-sglang-engine true

# Start SGLang server for VLM acceleration (default port 30000)
mineru-sglang-server --port 30000
```

### Configuration
```bash
# Set model source (if HuggingFace is inaccessible)
export MINERU_MODEL_SOURCE=modelscope

# Use local models (after setting models-dir in config)
export MINERU_MODEL_SOURCE=local
```

## Architecture

### High-Level Document Processing Flow
MinerU follows a modular pipeline architecture with three main processing paths:

1. **Backend Selection**: Choose between `pipeline` (OCR-based), `vlm-transformers`, or `vlm-sglang`
2. **Document Analysis**: Extract content using chosen backend's models/algorithms
3. **Post-processing**: Convert raw outputs to structured formats (markdown/JSON)
4. **Output Generation**: Create final deliverables with visualization overlays

### Backend Architecture Comparison

**Pipeline Backend** (`mineru/backend/pipeline/`):
- **Models**: PaddleOCR + YOLO layout detection + UniMERNet (formulas)
- **Strengths**: Multilingual support, works on CPU, handles scanned documents
- **Processing**: Sequential OCR → Layout analysis → Formula recognition → Content assembly
- **Best for**: Scanned PDFs, complex layouts, CPU-only environments

**VLM Backend** (`mineru/backend/vlm/`):
- **Models**: Vision-language models (transformers/SGLang)
- **Strengths**: End-to-end understanding, better semantic extraction
- **Processing**: Single-pass multimodal inference → Structured output
- **Best for**: High-quality native PDFs, when GPU memory (8GB+) available

### Key Entry Points & CLI Architecture
- `mineru/cli/client.py`: Main CLI entry point (`mineru` command)
- `mineru/cli/gradio_app.py`: Web interface with batch processing capabilities
- `mineru/cli/fast_api.py`: REST API server
- `mineru/cli/vlm_sglang_server.py`: SGLang acceleration server
- `mineru/cli/models_download.py`: Model management system

### Processing Pipeline Deep Dive
The core processing happens in three phases:

1. **PDF Preprocessing** (`mineru/cli/common.py`):
   - PDF page extraction using pypdfium2
   - Image conversion and preprocessing
   - Page-level parallelization setup

2. **Content Analysis** (Backend-specific):
   - **Pipeline**: `pipeline_analyze.py` → `pipeline_middle_json_mkcontent.py`
   - **VLM**: `vlm_analyze.py` → `vlm_middle_json_mkcontent.py`

3. **Output Generation**:
   - Middle JSON → Content List JSON conversion
   - Markdown generation with embedded images
   - Visualization overlays (layout bounding boxes)

### Configuration System
- **Template**: `mineru.template.json` defines all configurable options
- **User Config**: Auto-generated `~/mineru.json` on first model download
- **Environment Variables**: `MINERU_MODEL_SOURCE` for model source switching
- **Key Configs**: Model paths, S3 storage, LaTeX delimiters, LLM integration

### Model Management Architecture
- **Download System**: `mineru-models-download` handles model acquisition
- **Model Sources**: HuggingFace Hub, ModelScope, local filesystem
- **Storage**: Configurable model directory with backend separation
- **Dependencies**: Different model sets for pipeline vs VLM backends

### Web Interface Features (Gradio)
- **Single File Processing**: Individual PDF upload and conversion
- **Batch Processing**: Directory-based bulk processing with error handling
- **Real-time Progress**: Live status updates and error reporting
- **Error Management**: Failed files moved to `ERRORED/` subdirectory
- **Output Packaging**: Automatic ZIP archive creation

### Platform & Performance Considerations
- **Python**: 3.10-3.13 supported
- **Pipeline Backend**: Cross-platform (Windows/Linux/macOS), CPU/GPU optional
- **VLM Backends**: Linux/Windows only, GPU required (8GB+ VRAM)
- **SGLang**: Linux/WSL2 only, provides 20-30x VLM speedup
- **Memory Usage**: Pipeline ~2-4GB RAM, VLM ~8-16GB GPU memory

### Testing & Quality Assurance
- **Coverage Target**: Minimum 20% (enforced by `tests/get_coverage.py`)
- **E2E Testing**: `tests/unittest/test_e2e.py` validates full processing pipeline
- **Coverage Config**: Excludes CLI tools and SGLang-specific code from coverage
- **Test Scope**: Core processing logic, model integration, output validation

### Development Workflow Patterns
1. **Model-First Development**: Always run `mineru-models-download` before testing
2. **Backend-Specific Testing**: Use `demo/demo.py` to test different backends
3. **Coverage-Driven QA**: Use coverage tools to ensure code quality
4. **Configuration Testing**: Test with different `mineru.json` configurations
5. **Platform Testing**: Validate on target deployment platforms