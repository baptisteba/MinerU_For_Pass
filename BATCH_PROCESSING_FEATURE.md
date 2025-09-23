# MinerU Batch Processing Feature

## Overview

The MinerU Gradio UI now includes a new **Batch Processing** feature that allows users to process entire directories of PDF files while maintaining the original folder structure. This feature automatically scans directories recursively, processes PDFs based on system capacity, and outputs results in an organized manner.

## Features

### 🔍 **Directory Scanning**
- Recursively scans input directories for PDF files
- Preserves relative path information for folder structure recreation
- Validates directory accessibility and permissions

### 🚀 **Smart Capacity Management**
- Automatically detects system resources (CPU, memory)
- Adjusts concurrent processing based on backend type:
  - **Pipeline backend**: More concurrent files (less memory intensive)
  - **VLM backends**: Fewer concurrent files (more memory intensive)
- Fallback to conservative settings if system detection fails

### 📁 **Folder Structure Preservation**
- Creates `MinerU_Outputs` subdirectory in the specified output path
- Recreates the exact folder structure from the input directory
- Each processed file maintains its relative path position

### 📊 **Progress Tracking**
- Real-time progress updates during processing
- Detailed logging of successful and failed conversions
- Processing summary with statistics

### 📦 **Results Management**
- Automatic creation of ZIP archive with all processed files
- Maintains folder structure within the archive
- Easy download and distribution of results

## Usage

### 1. Access the Interface
Start MinerU Gradio interface:
```bash
mineru-gradio
```

### 2. Navigate to Batch Processing Tab
The interface now has two tabs:
- **Single File**: Original single PDF processing
- **Batch Processing**: New batch directory processing

### 3. Configure Batch Processing

#### Input Configuration:
- **Input Directory Path**: Full path to directory containing PDF files
  - Example: `C:\Documents\PDFs` or `/home/user/documents/pdfs`
- **Output Directory Path**: Where processed files will be saved
  - Example: `C:\Documents\Output` or `/home/user/output`

#### Processing Options:
- **Max convert pages**: Maximum pages per PDF to process
- **Backend**: Choose processing backend (pipeline, vlm-transformers, vlm-sglang-client, vlm-sglang-engine)
- **Recognition Options**:
  - Formula recognition
  - Table recognition
  - Images in markdown output
- **Language**: OCR language selection (for pipeline backend)
- **Force OCR**: Enable OCR processing

#### Server Configuration:
- **Server URL**: Required for vlm-sglang-client backend

### 4. Validate and Process
1. Click **Validate Directories** to check input/output paths
2. Review the directory status and PDF count
3. Click **Start Batch Processing** to begin

### 5. Monitor Progress
- Watch real-time progress updates in the progress panel
- View processing summary with success/failure statistics
- Download the results archive when complete

## Output Structure

```
Output Directory/
└── MinerU_Outputs/
    ├── subdirectory1/
    │   ├── file1_YYMMDD_HHMMSS/
    │   │   ├── auto/
    │   │   │   ├── images/
    │   │   │   ├── file1_YYMMDD_HHMMSS.md
    │   │   │   ├── file1_YYMMDD_HHMMSS_content_list.json
    │   │   │   └── file1_YYMMDD_HHMMSS_middle.json
    │   └── file2_YYMMDD_HHMMSS/
    │       └── auto/
    │           └── ...
    └── subdirectory2/
        └── ...
```

## System Requirements

### Optional Dependencies
- **psutil**: For advanced system resource detection
  - If not available, falls back to conservative processing settings
  - Install with: `pip install psutil`

### Recommended System Specs
- **Pipeline Backend**: 4GB+ RAM, any CPU
- **VLM Backends**: 8GB+ RAM, 8GB+ VRAM for GPU acceleration

## Error Handling

The batch processing feature includes robust error handling:

- **Directory Access Errors**: Clear validation messages for invalid paths
- **Processing Errors**: Individual file failures don't stop batch processing
- **Resource Constraints**: Automatic adjustment of concurrent processing
- **Partial Failures**: Complete summary with successful and failed file counts

## Technical Implementation

### Key Functions
- `scan_directory_for_pdfs()`: Recursive PDF discovery
- `get_system_capacity()`: Resource-aware capacity detection
- `batch_process_directory()`: Main batch processing orchestration
- `process_single_pdf_batch()`: Individual PDF processing with structure preservation

### Concurrency Management
- Uses `asyncio.gather()` for concurrent processing
- Batch size determined by system resources and backend type
- Conservative fallbacks for resource-constrained systems

### Progress Callbacks
- Real-time updates via callback functions
- Non-blocking progress reporting
- Detailed success/failure tracking

## Integration with Existing Features

The batch processing feature seamlessly integrates with:
- All existing MinerU backends (pipeline, vlm-transformers, vlm-sglang-engine, vlm-sglang-client)
- All recognition options (formulas, tables, images)
- All language settings and OCR options
- Existing output formats (markdown, JSON, images)

## Future Enhancements

Potential improvements for future versions:
- Resume interrupted batch processing
- Custom file filtering patterns
- Processing queue management
- Distributed processing across multiple machines
- Integration with cloud storage providers

---

This feature significantly enhances MinerU's capability for large-scale document processing while maintaining the quality and flexibility of single-file processing.