# Copyright (c) Opendatalab. All rights reserved.

import base64
import json
import os
import re
import time
import zipfile
import tempfile
from pathlib import Path
import asyncio
import threading
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict
import multiprocessing

import click
import gradio as gr
from gradio_pdf import PDF
from loguru import logger

# Fix for CUDA + multiprocessing fork issue
# NOTE: We DON'T set spawn globally anymore because it causes memory issues
# with large documents. Instead, we've removed multiprocessing.Pool usage
# from crash_recovery.py which was the actual problem.
# If multiprocessing is needed elsewhere, use get_context('spawn') locally

# Optional import for HTML table parsing
try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    logger.warning("BeautifulSoup not available. HTML to CSV conversion will use basic regex parsing.")

from mineru.cli.common import prepare_env, read_fn, aio_do_parse, pdf_suffixes, image_suffixes
from mineru.cli.crash_recovery import ProcessingCheckpoint, SafeProcessor
from mineru.utils.cli_parser import arg_parse
from mineru.utils.hash_utils import str_sha256

# Optional psutil import for system capacity detection
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available - using conservative batch processing settings")


class BatchProcessingTracker:
    """Tracks progress and errors for batch processing operations."""

    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.successful_files = 0
        self.failed_files = 0
        self.error_details: List[Dict] = []
        self.current_file = ""

    def update_current_file(self, file_path: str):
        """Update the currently processing file."""
        self.current_file = file_path

    def mark_success(self, file_path: str):
        """Mark a file as successfully processed."""
        self.processed_files += 1
        self.successful_files += 1

    def mark_failure(self, file_path: str, error_message: str):
        """Mark a file as failed and record error details."""
        self.processed_files += 1
        self.failed_files += 1
        self.error_details.append({
            'file_path': file_path,
            'error': error_message,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })

    def get_status_message(self) -> str:
        """Get current status message."""
        remaining = self.total_files - self.processed_files
        progress_pct = (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0

        status = f"📊 Progress: {self.processed_files}/{self.total_files} ({progress_pct:.1f}%)\n"
        status += f"✅ Successful: {self.successful_files}\n"
        status += f"❌ Failed: {self.failed_files}\n"
        status += f"⏳ Remaining: {remaining}\n"

        if self.current_file:
            status += f"🔄 Current: {os.path.basename(self.current_file)}"

        return status

    def get_error_summary(self) -> str:
        """Get a summary of all errors encountered."""
        if not self.error_details:
            return "✅ No errors encountered during processing."

        summary = f"❌ Error Summary ({len(self.error_details)} files failed):\n\n"
        for i, error in enumerate(self.error_details, 1):
            filename = os.path.basename(error['file_path'])
            summary += f"{i}. {filename}\n"
            summary += f"   Error: {error['error']}\n"
            summary += f"   Time: {error['timestamp']}\n\n"

        return summary


def move_file_to_error_folder(file_path: str, output_dir: str, relative_path: str) -> str:
    """
    Move a file that failed processing to the ERRORED folder while maintaining directory structure.

    Args:
        file_path: Absolute path to the original file
        output_dir: Base output directory
        relative_path: Relative path from input directory

    Returns:
        Path to the moved file in ERRORED folder
    """
    try:
        # Create ERRORED directory structure
        error_base_dir = os.path.join(output_dir, "ERRORED")
        relative_dir = os.path.dirname(relative_path)
        error_subdir = os.path.join(error_base_dir, relative_dir) if relative_dir else error_base_dir

        os.makedirs(error_subdir, exist_ok=True)

        # Create destination path
        filename = os.path.basename(file_path)
        error_file_path = os.path.join(error_subdir, filename)

        # Copy file to error folder (don't move original in case user wants to retry)
        shutil.copy2(file_path, error_file_path)

        return error_file_path
    except Exception as e:
        logger.warning(f"Failed to move file to ERRORED folder: {e}")
        return None


async def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, formula_enable, table_enable, language, backend, url, images_enable=False, md_only=False, fast_mode=False):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{safe_stem(Path(doc_path).stem)}_{time.strftime("%y%m%d_%H%M%S")}'
        pdf_data = read_fn(doc_path)
        if is_ocr:
            parse_method = 'ocr'
        else:
            parse_method = 'auto'

        if backend.startswith("vlm"):
            parse_method = "vlm"

        # Apply MD-only and fast mode optimizations
        if md_only:
            # In MD-only mode, disable images to prevent images folder creation
            images_enable = False

        if fast_mode and md_only:
            # In fast mode with MD-only, disable formula recognition but keep table recognition
            formula_enable = False
            # table_enable remains unchanged - keep tables for better content extraction
            # Also consider reducing OCR precision for speed (can add more optimizations here)
            # For now, we rely on the backend optimizations

        if md_only:
            # For MD-only mode, override parse_method to be empty so prepare_env doesn't create subdirectory
            actual_parse_method = ""
            local_md_dir = os.path.join(output_dir, file_name)
            # Don't create images directory in MD-only mode
            local_image_dir = None
        else:
            # Use standard environment preparation for full output
            local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method, create_images_dir=images_enable)
            actual_parse_method = parse_method

        await aio_do_parse(
            output_dir=output_dir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_data],
            p_lang_list=[language],
            parse_method=actual_parse_method,
            end_page_id=end_page_id,
            formula_enable=formula_enable,
            table_enable=table_enable,
            backend=backend,
            server_url=url,
            images_enable=images_enable,
            # MD-only flags
            f_draw_layout_bbox=not md_only,
            f_draw_span_bbox=not md_only,
            f_dump_md=True,  # Always generate MD
            f_dump_middle_json=not md_only,
            f_dump_model_output=not md_only,
            f_dump_orig_pdf=not md_only,
            f_dump_content_list=not md_only,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)
        return None


def compress_directory_to_zip(directory_path, output_zip_path):
    """压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def html_table_to_csv(html_table):
    """
    Convert HTML table to CSV format for markdown display.

    Args:
        html_table (str): HTML table string

    Returns:
        str: CSV formatted table for markdown
    """
    try:
        if HAS_BEAUTIFULSOUP:
            # Use BeautifulSoup for robust HTML parsing
            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')

            if not table:
                return html_table  # Return original if no table found

            rows = []

            # Extract all rows (including header and body)
            for tr in table.find_all('tr'):
                cells = []
                for cell in tr.find_all(['td', 'th']):
                    # Get cell text and clean it
                    cell_text = cell.get_text(strip=True)
                    # Escape any commas or quotes in cell content
                    if ',' in cell_text or '"' in cell_text or '\n' in cell_text:
                        cell_text = '"' + cell_text.replace('"', '""') + '"'
                    cells.append(cell_text)

                if cells:  # Only add non-empty rows
                    rows.append(','.join(cells))

            if not rows:
                return html_table  # Return original if no rows found

            # Format as markdown CSV table
            csv_content = '\n'.join(rows)

            # Wrap in markdown code block with csv language for proper formatting
            return f"```csv\n{csv_content}\n```"

        else:
            # Fallback: Basic regex-based parsing when BeautifulSoup is not available
            return _html_table_to_csv_regex(html_table)

    except Exception as e:
        logger.warning(f"Failed to convert HTML table to CSV: {e}")
        return html_table  # Return original HTML on error


def _html_table_to_csv_regex(html_table):
    """
    Fallback function to convert HTML table to CSV using regex when BeautifulSoup is not available.

    Args:
        html_table (str): HTML table string

    Returns:
        str: CSV formatted table for markdown
    """
    try:
        import re

        # Extract table content
        table_match = re.search(r'<table[^>]*>(.*?)</table>', html_table, re.DOTALL | re.IGNORECASE)
        if not table_match:
            return html_table

        table_content = table_match.group(1)

        # Extract rows
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        row_matches = re.findall(row_pattern, table_content, re.DOTALL | re.IGNORECASE)

        csv_rows = []
        for row_html in row_matches:
            # Extract cells (both td and th)
            cell_pattern = r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>'
            cell_matches = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)

            if cell_matches:
                cells = []
                for cell_html in cell_matches:
                    # Clean up HTML tags and entities
                    cell_text = re.sub(r'<[^>]+>', '', cell_html)  # Remove HTML tags
                    cell_text = cell_text.strip()

                    # Handle basic HTML entities
                    cell_text = cell_text.replace('&nbsp;', ' ')
                    cell_text = cell_text.replace('&amp;', '&')
                    cell_text = cell_text.replace('&lt;', '<')
                    cell_text = cell_text.replace('&gt;', '>')
                    cell_text = cell_text.replace('&quot;', '"')

                    # Escape CSV special characters
                    if ',' in cell_text or '"' in cell_text or '\n' in cell_text:
                        cell_text = '"' + cell_text.replace('"', '""') + '"'

                    cells.append(cell_text)

                if cells:
                    csv_rows.append(','.join(cells))

        if not csv_rows:
            return html_table

        # Format as markdown CSV table
        csv_content = '\n'.join(csv_rows)
        return f"```csv\n{csv_content}\n```"

    except Exception as e:
        logger.warning(f"Failed to convert HTML table to CSV using regex fallback: {e}")
        return html_table


def convert_tables_to_csv_in_markdown(markdown_text, csv_tables_enabled=True):
    """
    Convert all HTML tables in markdown text to CSV format.

    Args:
        markdown_text (str): Markdown text containing HTML tables
        csv_tables_enabled (bool): Whether to convert tables to CSV

    Returns:
        str: Markdown text with CSV tables (if enabled) or original text
    """
    if not csv_tables_enabled:
        return markdown_text

    # Pattern to match HTML tables in markdown
    table_pattern = r'<table[^>]*>.*?</table>'

    def replace_table(match):
        html_table = match.group(0)
        return html_table_to_csv(html_table)

    # Replace all HTML tables with CSV format
    converted_text = re.sub(table_pattern, replace_table, markdown_text, flags=re.DOTALL | re.IGNORECASE)

    return converted_text


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f'![{relative_path}](data:image/jpeg;base64,{base64_image})'

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


async def to_markdown(file_path, end_pages=10, is_ocr=False, formula_enable=True, table_enable=True, language="ch", backend="pipeline", url=None, images_enable=False, md_only=False, fast_mode=False, csv_tables=True):
    file_path = to_pdf(file_path)
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = await parse_pdf(file_path, './output', end_pages - 1, is_ocr, formula_enable, table_enable, language, backend, url, images_enable, md_only, fast_mode)

    # Only create archive if not MD-only mode
    archive_zip_path = None
    if not md_only:
        archive_zip_path = os.path.join('./output', str_sha256(local_md_dir) + '.zip')
        zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
        if zip_archive_success == 0:
            logger.info('Compression successful')
        else:
            logger.error('Compression failed')
            archive_zip_path = None

    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()

    # Apply CSV table conversion if enabled
    if csv_tables:
        txt_content = convert_tables_to_csv_in_markdown(txt_content, csv_tables_enabled=True)

    md_content = replace_image_with_base64(txt_content, local_md_dir)

    # Only return PDF path if not MD-only mode
    new_pdf_path = None
    if not md_only:
        layout_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')
        if os.path.exists(layout_pdf_path):
            new_pdf_path = layout_pdf_path

    return md_content, txt_content, archive_zip_path, new_pdf_path


latex_delimiters_type_a = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
]
latex_delimiters_type_b = [
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]
latex_delimiters_type_all = latex_delimiters_type_a + latex_delimiters_type_b

header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'header.html')
with open(header_path, 'r') as header_file:
    header = header_file.read()


latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',  # noqa: E126
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
        'rs_cyrillic', 'bg', 'mn', 'abq', 'ady', 'kbd', 'ava',  # noqa: E126
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
east_slavic_lang = ["ru", "be", "uk"]
devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',  # noqa: E126
        'sa', 'bgc'
]
other_lang = ['ch', 'ch_lite', 'ch_server', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', "el", "th"]
add_lang = ['latin', 'arabic', 'east_slavic', 'cyrillic', 'devanagari']

# all_lang = ['', 'auto']
all_lang = []
# all_lang.extend([*other_lang, *latin_lang, *arabic_lang, *cyrillic_lang, *devanagari_lang])
all_lang.extend([*other_lang, *add_lang])


def safe_stem(file_path):
    stem = Path(file_path).stem
    # 只保留字母、数字、下划线和点，其他字符替换为下划线
    return re.sub(r'[^\w.]', '_', stem)


def to_pdf(file_path):

    if file_path is None:
        return None

    pdf_bytes = read_fn(file_path)

    # unique_filename = f'{uuid.uuid4()}.pdf'
    unique_filename = f'{safe_stem(file_path)}.pdf'

    # 构建完整的文件路径
    tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

    # 将字节数据写入文件
    with open(tmp_file_path, 'wb') as tmp_pdf_file:
        tmp_pdf_file.write(pdf_bytes)

    return tmp_file_path


def scan_directory_for_pdfs(directory_path: str) -> List[Tuple[str, str]]:
    """
    Recursively scan directory for PDF files.

    Args:
        directory_path: Path to the directory to scan

    Returns:
        List of tuples (absolute_path, relative_path_from_root) sorted by path
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")

    pdf_files = []
    root_path = Path(directory_path).resolve()

    for root, dirs, files in os.walk(directory_path):
        # Sort directories and files to ensure consistent order
        dirs.sort()
        files.sort()
        for file in files:
            if file.lower().endswith(('.pdf',)):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(root_path)
                pdf_files.append((str(file_path), str(relative_path)))

    # Sort the final list by relative path to ensure consistent ordering
    pdf_files.sort(key=lambda x: x[1])

    return pdf_files


def get_system_capacity() -> int:
    """
    Determine system capacity for batch processing based on available resources.
    
    Returns:
        Number of files to process simultaneously
    """
    if HAS_PSUTIL:
        try:
            # Get system memory in GB
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Get CPU count
            cpu_count = psutil.cpu_count(logical=False) or 1
            
            # Conservative capacity calculation
            # For pipeline backend: can handle more files (less memory intensive)
            # For VLM backends: requires more memory per file
            if memory_gb >= 16:
                return min(cpu_count, 4)
            elif memory_gb >= 8:
                return min(cpu_count, 2)
            else:
                return 1
        except Exception:
            logger.warning("Failed to detect system resources, using conservative settings")
            return 1
    else:
        # Fallback to conservative settings when psutil is not available
        try:
            cpu_count = os.cpu_count() or 1
            return min(cpu_count, 2)
        except:
            return 1


async def process_single_pdf_batch(pdf_info: Tuple[str, str], output_base_dir: str, processing_params: dict, tracker: BatchProcessingTracker = None, progress_callback=None):
    """
    Process a single PDF file while maintaining directory structure.

    Args:
        pdf_info: Tuple of (absolute_path, relative_path)
        output_base_dir: Base output directory
        processing_params: Dictionary containing processing parameters
        tracker: BatchProcessingTracker instance for progress tracking
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (success: bool, file_path: str, error_message: str)
    """
    pdf_path, relative_path = pdf_info

    # Update tracker with current file
    if tracker:
        tracker.update_current_file(pdf_path)
        if progress_callback:
            progress_callback(tracker.get_status_message())

    try:
        # Create output directory maintaining folder structure
        relative_dir = os.path.dirname(relative_path)
        output_subdir = os.path.join(output_base_dir, "MinerU_Outputs", relative_dir) if relative_dir else os.path.join(output_base_dir, "MinerU_Outputs")
        
        # Generate unique filename for this processing run
        pdf_name = Path(pdf_path).stem
        file_name = f'{safe_stem(pdf_name)}_{time.strftime("%y%m%d_%H%M%S")}'
        
        # Setup processing parameters
        backend = processing_params.get('backend', 'pipeline')
        parse_method = 'ocr' if processing_params.get('is_ocr', False) else 'auto'
        if backend.startswith("vlm"):
            parse_method = "vlm"

        # Get MD-only and fast mode settings
        md_only = processing_params.get('md_only', False)
        fast_mode = processing_params.get('fast_mode', False)

        # Apply MD-only and fast mode optimizations
        formula_enable = processing_params.get('formula_enable', True)
        table_enable = processing_params.get('table_enable', True)
        images_enable = processing_params.get('images_enable', False)
        csv_tables = processing_params.get('csv_tables', True)

        if md_only:
            # In MD-only mode, disable images to prevent images folder creation
            images_enable = False

        if fast_mode and md_only:
            formula_enable = False
            # table_enable remains unchanged - keep tables for better content extraction
            # Additional fast mode optimizations can be added here

        # For MD-only mode, override parse_method to be empty so prepare_env doesn't create subdirectory
        if md_only:
            actual_parse_method = ""
        else:
            actual_parse_method = parse_method

        # Prepare environment - for batch processing, place files directly in output_subdir without parse_method folder
        final_output_dir = os.path.join(output_subdir, file_name)
        local_md_dir = final_output_dir

        # Only create and use images directory if images are enabled (not in MD-only or fast mode)
        if md_only or not images_enable:
            local_image_dir = None
        else:
            local_image_dir = os.path.join(final_output_dir, "images")
            os.makedirs(local_image_dir, exist_ok=True)

        os.makedirs(local_md_dir, exist_ok=True)
        
        # Read PDF data
        pdf_data = read_fn(pdf_path)

        # Process the PDF
        await aio_do_parse(
            output_dir=output_subdir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_data],
            p_lang_list=[processing_params.get('language', 'en')],
            parse_method=actual_parse_method,
            end_page_id=processing_params.get('end_page_id', 999),
            formula_enable=formula_enable,
            table_enable=table_enable,
            backend=backend,
            server_url=processing_params.get('server_url'),
            images_enable=images_enable,
            # MD-only flags
            f_draw_layout_bbox=not md_only,
            f_draw_span_bbox=not md_only,
            f_dump_md=True,  # Always generate MD
            f_dump_middle_json=not md_only,
            f_dump_model_output=not md_only,
            f_dump_orig_pdf=not md_only,
            f_dump_content_list=not md_only,
        )

        # Apply CSV table conversion if enabled
        if csv_tables:
            try:
                md_file_path = os.path.join(local_md_dir, file_name + '.md')
                if os.path.exists(md_file_path):
                    with open(md_file_path, 'r', encoding='utf-8') as f:
                        md_content = f.read()

                    # Convert tables to CSV format
                    converted_content = convert_tables_to_csv_in_markdown(md_content, csv_tables_enabled=True)

                    # Write back the converted content
                    with open(md_file_path, 'w', encoding='utf-8') as f:
                        f.write(converted_content)
            except Exception as e:
                logger.warning(f"Failed to apply CSV table conversion: {e}")

        # Mark as successful
        if tracker:
            tracker.mark_success(pdf_path)
            if progress_callback:
                progress_callback(tracker.get_status_message())

        if progress_callback:
            progress_callback(f"✓ Processed: {relative_path}")

        return True, pdf_path, ""

    except Exception as e:
        error_msg = f"Error processing {relative_path}: {str(e)}"
        logger.exception(error_msg)

        # Move failed file to ERRORED folder
        try:
            error_file_path = move_file_to_error_folder(pdf_path, output_base_dir, relative_path)
            if error_file_path:
                error_msg += f" (copied to ERRORED folder)"
        except Exception as move_error:
            logger.warning(f"Failed to move error file: {move_error}")

        # Mark as failed
        if tracker:
            tracker.mark_failure(pdf_path, str(e))
            if progress_callback:
                progress_callback(tracker.get_status_message())

        if progress_callback:
            progress_callback(f"✗ Failed: {relative_path} - {str(e)}")

        return False, pdf_path, error_msg

    finally:
        # Aggressive resource cleanup after each file to prevent C++ memory corruption
        # Using upstream's improved clean_memory (includes torch.cuda.ipc_collect)
        try:
            from mineru.utils.model_utils import clean_memory
            clean_memory('cuda')  # Upstream version - better than manual empty_cache!
        except ImportError:
            # Fallback if model_utils not available
            import gc
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            gc.collect()

        # Small delay to allow C++ destructors to complete
        await asyncio.sleep(0.05)


async def batch_process_directory_with_auto_restart(
    input_dir: str,
    output_dir: str,
    max_pages: int = 10,
    is_ocr: bool = False,
    formula_enable: bool = True,
    table_enable: bool = True,
    images_enable: bool = False,
    language: str = "en",
    backend: str = "pipeline",
    server_url: str = None,
    md_only: bool = False,
    fast_mode: bool = False,
    csv_tables: bool = True,
    progress_callback=None,
    resume_checkpoint: Optional[str] = None,
    max_restart_attempts: int = 10
):
    """
    Batch process with automatic crash recovery and restart.

    This wrapper automatically restarts processing from the last checkpoint
    when a crash occurs, eliminating the need for manual restarts every 5-6 files.

    Args:
        max_restart_attempts: Maximum number of auto-restart attempts (default: 10)
        All other args: Same as batch_process_directory

    Returns:
        Same as batch_process_directory
    """
    restart_count = 0
    last_checkpoint_file = resume_checkpoint

    while restart_count < max_restart_attempts:
        try:
            # Attempt processing
            result = await batch_process_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                max_pages=max_pages,
                is_ocr=is_ocr,
                formula_enable=formula_enable,
                table_enable=table_enable,
                images_enable=images_enable,
                language=language,
                backend=backend,
                server_url=server_url,
                md_only=md_only,
                fast_mode=fast_mode,
                csv_tables=csv_tables,
                progress_callback=progress_callback,
                resume_checkpoint=last_checkpoint_file
            )

            # Success! Processing completed without crash
            if progress_callback:
                if restart_count > 0:
                    progress_callback(f"✅ Processing completed successfully after {restart_count} auto-restart(s)")

            return result

        except Exception as e:
            restart_count += 1
            error_msg = str(e)

            logger.error(f"💥 Crash detected (restart attempt {restart_count}/{max_restart_attempts}): {error_msg}")

            if progress_callback:
                progress_callback(f"⚠️ Crash detected: {error_msg}")
                progress_callback(f"🔄 Auto-restarting from checkpoint (attempt {restart_count}/{max_restart_attempts})...")

            # Check if we've exceeded max attempts
            if restart_count >= max_restart_attempts:
                logger.error("❌ Maximum restart attempts reached - cannot continue")
                if progress_callback:
                    progress_callback(f"❌ Max restart attempts ({max_restart_attempts}) reached. Processing incomplete.")
                return 0, 0, 1, None

            # Find the most recent checkpoint to resume from
            try:
                checkpoint_dir = get_checkpoint_dir()
                checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))

                if checkpoints:
                    # Find most recent checkpoint
                    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    last_checkpoint_file = str(latest_checkpoint)

                    # Load checkpoint to see progress
                    with open(latest_checkpoint, 'r') as f:
                        checkpoint_data = json.load(f)
                        processed_count = len(checkpoint_data.get('processed_files', []))

                    logger.info(f"📂 Found checkpoint: {latest_checkpoint.name}")
                    logger.info(f"   Already processed: {processed_count} files")

                    if progress_callback:
                        progress_callback(f"📂 Resuming from checkpoint with {processed_count} files already processed")

                else:
                    logger.warning("⚠️ No checkpoint found for auto-resume")
                    if progress_callback:
                        progress_callback("⚠️ No checkpoint found - cannot auto-restart")
                    return 0, 0, 1, None

            except Exception as checkpoint_error:
                logger.error(f"Failed to load checkpoint for restart: {checkpoint_error}")
                if progress_callback:
                    progress_callback(f"❌ Failed to load checkpoint: {checkpoint_error}")
                return 0, 0, 1, None

            # Small delay before restart to allow resources to settle
            await asyncio.sleep(2)

            # Continue loop to retry with checkpoint


async def batch_process_directory(
    input_dir: str,
    output_dir: str,
    max_pages: int = 10,
    is_ocr: bool = False,
    formula_enable: bool = True,
    table_enable: bool = True,
    images_enable: bool = False,
    language: str = "en",
    backend: str = "pipeline",
    server_url: str = None,
    md_only: bool = False,
    fast_mode: bool = False,
    csv_tables: bool = True,
    progress_callback=None,
    resume_checkpoint: Optional[str] = None
):
    """
    Batch process all PDFs in a directory while maintaining folder structure.

    Args:
        input_dir: Input directory to scan for PDFs
        output_dir: Output directory for processed files
        Other parameters: Processing options
        progress_callback: Optional callback for progress updates
        resume_checkpoint: Optional checkpoint file to resume from

    Returns:
        Tuple of (total_files, successful_files, failed_files, results_zip_path)
    """
    # Initialize checkpoint
    checkpoint = ProcessingCheckpoint()

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        # Resume from existing checkpoint
        checkpoint.load_session(resume_checkpoint)
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        if progress_callback:
            progress_callback(f"📂 Resuming from checkpoint with {len(checkpoint.state['processed_files'])} already processed")
    else:
        # Create new checkpoint
        session_id = f"{Path(input_dir).name}_{time.strftime('%Y%m%d_%H%M%S')}"
        checkpoint_file = checkpoint.create_session(session_id)

        # Save processing parameters to checkpoint for future resume
        checkpoint.update(
            processing_params={
                'input_dir': input_dir,
                'output_dir': output_dir,
                'max_pages': max_pages,
                'is_ocr': is_ocr,
                'formula_enable': formula_enable,
                'table_enable': table_enable,
                'images_enable': images_enable,
                'language': language,
                'backend': backend,
                'server_url': server_url,
                'md_only': md_only,
                'fast_mode': fast_mode,
                'csv_tables': csv_tables
            }
        )

        logger.info(f"Created checkpoint: {checkpoint_file}")

    try:
        if progress_callback:
            progress_callback("🔍 Scanning directory for PDF files...")
        
        # Scan for PDF files
        pdf_files = scan_directory_for_pdfs(input_dir)

        # Filter out already processed files if resuming
        if resume_checkpoint:
            all_file_paths = [pdf_info[0] for pdf_info in pdf_files]
            remaining_files = checkpoint.get_remaining_files(all_file_paths)
            pdf_files = [pdf_info for pdf_info in pdf_files if pdf_info[0] in remaining_files]

            if progress_callback:
                progress_callback(f"📊 {len(remaining_files)} files remaining to process")

        if not pdf_files:
            if progress_callback:
                progress_callback("✅ All files have been processed" if resume_checkpoint else "❌ No PDF files found in the specified directory")
            return 0, 0, 0, None
        
        if progress_callback:
            progress_callback(f"📁 Found {len(pdf_files)} PDF files")

        # Initialize progress tracker
        tracker = BatchProcessingTracker(len(pdf_files))

        # Determine processing capacity
        system_capacity = get_system_capacity()
        if backend.startswith("vlm"):
            # VLM backends are more memory intensive
            batch_size = max(1, system_capacity // 2)
        else:
            batch_size = system_capacity
        
        if progress_callback:
            progress_callback(f"🚀 Processing {batch_size} files at a time (system capacity: {system_capacity})")
        
        # Prepare processing parameters
        processing_params = {
            'backend': backend,
            'is_ocr': is_ocr,
            'language': language,
            'end_page_id': max_pages - 1,
            'formula_enable': formula_enable,
            'table_enable': table_enable,
            'images_enable': images_enable,
            'server_url': server_url,
            'md_only': md_only,
            'fast_mode': fast_mode,
            'csv_tables': csv_tables
        }
        
        # Process files in batches
        results = []

        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]

            if progress_callback:
                progress_callback(f"📄 Processing batch {i//batch_size + 1}/{(len(pdf_files) + batch_size - 1)//batch_size}")

            # Process batch concurrently
            tasks = [
                process_single_pdf_batch(pdf_info, output_dir, processing_params, tracker, progress_callback)
                for pdf_info in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count results and handle exceptions, update checkpoint
            for j, result in enumerate(batch_results):
                pdf_path, relative_path = batch[j]

                if isinstance(result, Exception):
                    # Handle unexpected exceptions not caught by process_single_pdf_batch
                    error_msg = str(result)
                    tracker.mark_failure(pdf_path, error_msg)
                    results.append((False, pdf_path, error_msg))
                    checkpoint.mark_file_processed(pdf_path, success=False)

                    # Try to move file to ERRORED folder
                    try:
                        move_file_to_error_folder(pdf_path, output_dir, relative_path)
                    except Exception as move_error:
                        logger.warning(f"Failed to move error file: {move_error}")
                else:
                    results.append(result)
                    # Mark successful processing in checkpoint
                    if result[0]:  # First element is success flag
                        checkpoint.mark_file_processed(pdf_path, success=True)
                    else:
                        checkpoint.mark_file_processed(pdf_path, success=False)
        
        # Create results archive only if not MD-only mode
        output_root = os.path.join(output_dir, "MinerU_Outputs")
        archive_zip_path = None
        if not md_only and os.path.exists(output_root):
            archive_zip_path = os.path.join(output_dir, f'MinerU_batch_results_{time.strftime("%y%m%d_%H%M%S")}.zip')
            compress_result = compress_directory_to_zip(output_root, archive_zip_path)

            if compress_result == 0:
                if progress_callback:
                    progress_callback(f"📦 Created results archive: {archive_zip_path}")
            else:
                archive_zip_path = None
        elif md_only and progress_callback:
            progress_callback("📝 MD-only mode: No archive created")
        
        # Final status update
        if progress_callback:
            progress_callback(f"✅ Batch processing completed: {tracker.successful_files} successful, {tracker.failed_files} failed")

            # Add error summary if there were failures
            if tracker.failed_files > 0:
                progress_callback("\n" + "="*50)
                progress_callback(tracker.get_error_summary())

        # Mark checkpoint as completed
        checkpoint.update(status="completed")
        logger.info("Batch processing completed successfully - checkpoint marked as completed")

        return len(pdf_files), tracker.successful_files, tracker.failed_files, archive_zip_path
        
    except Exception as e:
        error_msg = f"Batch processing error: {str(e)}"
        logger.exception(error_msg)

        # Mark checkpoint as crashed so it can be resumed
        checkpoint.update(status="crashed", error=str(e))
        logger.error(f"Batch processing crashed - checkpoint saved for recovery at: {checkpoint.checkpoint_file}")

        if progress_callback:
            progress_callback(f"❌ {error_msg}")
            progress_callback(f"💾 Checkpoint saved for recovery. You can resume from checkpoint after restarting.")

        return 0, 0, 1, None


def get_checkpoint_dir() -> Path:
    """
    Get the checkpoint directory path.

    Uses permanent location in user's home directory (~/.mineru/checkpoints)
    to ensure checkpoints survive web server restarts and system reboots.
    """
    home_dir = Path.home()
    checkpoint_dir = home_dir / ".mineru" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def list_available_checkpoints() -> List[Tuple[str, str]]:
    """
    List all available checkpoint files.

    Returns:
        List of tuples (display_name, checkpoint_path)
    """
    checkpoint_dir = get_checkpoint_dir()
    checkpoints = []

    if checkpoint_dir.exists():
        for checkpoint_file in sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    session_id = data.get('session_id', 'Unknown')
                    created_at = data.get('created_at', 'Unknown')
                    processed_count = len(data.get('processed_files', []))
                    failed_count = len(data.get('failed_files', []))
                    status = data.get('status', 'unknown')

                    display_name = f"{session_id} | {created_at} | Processed: {processed_count}, Failed: {failed_count} | Status: {status}"
                    checkpoints.append((display_name, str(checkpoint_file)))
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")

    return checkpoints


def load_checkpoint_info(checkpoint_path: str) -> Dict:
    """
    Load checkpoint information for display.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint information
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}

    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        # Format information for display
        info = {
            'session_id': data.get('session_id', 'Unknown'),
            'created_at': data.get('created_at', 'Unknown'),
            'last_update': data.get('last_update', 'Unknown'),
            'status': data.get('status', 'unknown'),
            'processed_count': len(data.get('processed_files', [])),
            'failed_count': len(data.get('failed_files', [])),
            'processing_params': data.get('processing_params', {})
        }

        return info
    except Exception as e:
        logger.error(f"Failed to load checkpoint info: {e}")
        return {}


def format_checkpoint_info(info: Dict) -> str:
    """Format checkpoint info for display in UI."""
    if not info:
        return "No checkpoint selected"

    status = info.get('status', 'unknown')

    # Add status emoji
    status_emoji = {
        'started': '🔄',
        'completed': '✅',
        'crashed': '❌',
        'unknown': '❓'
    }.get(status, '❓')

    text = f"📋 **Checkpoint Information**\n\n"
    text += f"**Session:** {info.get('session_id', 'Unknown')}\n"
    text += f"**Created:** {info.get('created_at', 'Unknown')}\n"
    text += f"**Last Update:** {info.get('last_update', 'Unknown')}\n"
    text += f"**Status:** {status_emoji} {status}\n"
    text += f"**Processed Files:** {info.get('processed_count', 0)}\n"
    text += f"**Failed Files:** {info.get('failed_count', 0)}\n\n"

    # Highlight if resumable
    if status in ['started', 'crashed']:
        remaining = info.get('processing_params', {}).get('total_files', 0) - info.get('processed_count', 0) - info.get('failed_count', 0)
        if remaining > 0 or status == 'crashed':
            text += f"⚠️ **This session can be resumed!**\n"
            text += f"Files still need processing or session was interrupted.\n\n"

    params = info.get('processing_params', {})
    if params:
        text += "**Original Settings:**\n"
        text += f"- Input Directory: {params.get('input_dir', 'N/A')}\n"
        text += f"- Output Directory: {params.get('output_dir', 'N/A')}\n"
        text += f"- Backend: {params.get('backend', 'N/A')}\n"
        text += f"- Max Pages: {params.get('max_pages', 'N/A')}\n"
        text += f"- Language: {params.get('language', 'N/A')}\n"
        text += f"- Formula Recognition: {params.get('formula_enable', 'N/A')}\n"
        text += f"- Table Recognition: {params.get('table_enable', 'N/A')}\n"
        text += f"- MD Only: {params.get('md_only', 'N/A')}\n"
        text += f"- Fast Mode: {params.get('fast_mode', 'N/A')}\n"

    return text


def validate_directory_path(dir_path: str) -> str:
    """Validate and return directory path status."""
    if not dir_path:
        return "Please enter a directory path"

    if not os.path.exists(dir_path):
        return "Directory does not exist"

    if not os.path.isdir(dir_path):
        return "Path is not a directory"

    try:
        # Check if we can scan the directory
        pdf_files = scan_directory_for_pdfs(dir_path)
        return f"✓ Directory is valid - Found {len(pdf_files)} PDF files"
    except Exception as e:
        return f"Error scanning directory: {str(e)}"


def run_batch_processing_with_subprocess_protection(
    input_dir, output_dir, max_pages, is_ocr, formula_enable,
    table_enable, images_enable, csv_tables, language, backend, url, md_only, fast_mode,
    resume_checkpoint=None, max_restart_attempts=10
):
    """
    Run batch processing in a subprocess to protect against core dumps and crashes.

    This function launches the batch processor as a subprocess and monitors it.
    If the subprocess crashes (for any reason including core dumps), it automatically
    restarts from the last checkpoint until all files are processed or max attempts reached.

    Args:
        max_restart_attempts: Maximum number of restart attempts (default: 10)
        All other args: Same as run_batch_processing
    """
    if not input_dir or not input_dir.strip():
        return "❌ Please specify an input directory", None, "Processing failed - no input directory specified"

    if not output_dir or not output_dir.strip():
        return "❌ Please specify an output directory", None, "Processing failed - no output directory specified"

    # Validate directories
    input_status = validate_directory_path(input_dir.strip())
    if not input_status.startswith("✓"):
        return f"❌ Input directory error: {input_status}", None, "Processing failed - invalid input directory"

    # Create output directory if it doesn't exist
    output_dir = output_dir.strip()
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        return f"❌ Cannot create output directory: {str(e)}", None, "Processing failed - cannot create output directory"

    # Track progress messages
    progress_messages = []

    def add_progress(message):
        """Add progress message to tracking."""
        if isinstance(message, str):
            for line in message.split('\n'):
                if line.strip():
                    progress_messages.append(line.strip())
        else:
            progress_messages.append(str(message))

    # Get path to worker script
    worker_script = Path(__file__).parent / "batch_worker.py"
    if not worker_script.exists():
        return f"❌ Worker script not found: {worker_script}", None, "Processing failed - worker script missing"

    # Initial configuration for worker
    config = {
        'input_dir': input_dir.strip(),
        'output_dir': output_dir,
        'max_pages': max_pages,
        'is_ocr': is_ocr,
        'formula_enable': formula_enable,
        'table_enable': table_enable,
        'images_enable': images_enable,
        'language': language,
        'backend': backend,
        'server_url': url if url and url.strip() else None,
        'md_only': md_only,
        'fast_mode': fast_mode,
        'csv_tables': csv_tables,
        'resume_checkpoint': resume_checkpoint
    }

    restart_count = 0
    last_result = None

    while restart_count <= max_restart_attempts:
        try:
            # Write configuration to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
                json.dump(config, config_file, indent=2)
                config_path = config_file.name

            add_progress(f"🚀 Starting batch processing (attempt {restart_count + 1}/{max_restart_attempts + 1})...")
            if config['resume_checkpoint']:
                add_progress(f"📂 Resuming from checkpoint: {Path(config['resume_checkpoint']).name}")

            # Launch worker subprocess - don't capture stdout/stderr so logs go to terminal
            process = subprocess.Popen(
                [sys.executable, str(worker_script), config_path],
                stdout=subprocess.PIPE,
                stderr=None,  # stderr goes directly to terminal
                text=True,
                bufsize=1
            )

            # Monitor subprocess output
            result_data = None
            error_data = None

            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    # Always print the line to terminal for visibility
                    print(line.rstrip(), flush=True)

                    line = line.strip()
                    if line.startswith("PROGRESS:"):
                        # Extract progress message
                        msg = line[9:].strip()
                        add_progress(msg)
                    elif line.startswith("RESULT:"):
                        # Extract result JSON
                        result_json = line[7:].strip()
                        try:
                            result_data = json.loads(result_json)
                        except json.JSONDecodeError:
                            pass
                    elif line.startswith("ERROR:"):
                        # Extract error JSON
                        error_json = line[6:].strip()
                        try:
                            error_data = json.loads(error_json)
                        except json.JSONDecodeError:
                            pass

            # Wait for process to complete
            returncode = process.wait()

            # Clean up config file
            try:
                os.unlink(config_path)
            except:
                pass

            # Check result
            if returncode == 0 and result_data and result_data.get('status') == 'completed':
                # Success!
                last_result = result_data
                add_progress("✅ Batch processing completed successfully!")

                if restart_count > 0:
                    add_progress(f"📊 Processing survived {restart_count} crash(es) and recovered automatically")

                # Generate summary
                total_files = result_data.get('total_files', 0)
                successful_files = result_data.get('successful_files', 0)
                failed_files = result_data.get('failed_files', 0)
                zip_path = result_data.get('zip_path')

                success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
                summary = f"✅ Batch processing completed:\n"
                summary += f"• Total files: {total_files}\n"
                summary += f"• Successful: {successful_files} ({success_rate:.1f}%)\n"
                summary += f"• Failed: {failed_files}\n"
                if zip_path:
                    summary += f"• Results archive: {os.path.basename(zip_path)}\n"
                if failed_files > 0:
                    summary += f"• Failed files copied to: {os.path.join(output_dir, 'ERRORED')}\n"
                summary += f"• Output directory: {os.path.join(output_dir, 'MinerU_Outputs')}"

                return summary, zip_path, "\n".join(progress_messages)

            # Process crashed or failed
            restart_count += 1

            if returncode != 0:
                if returncode < 0:
                    # Killed by signal (e.g., SIGSEGV, SIGTRAP)
                    signal_name = {-11: "SIGSEGV (Segmentation Fault)", -5: "SIGTRAP (Core Dump)"}.get(returncode, f"Signal {-returncode}")
                    add_progress(f"💥 Worker crashed: {signal_name}")
                    logger.error(f"Worker process crashed with signal: {signal_name}")
                else:
                    add_progress(f"💥 Worker failed with exit code: {returncode}")
                    logger.error(f"Worker process failed with exit code: {returncode}")

                if error_data:
                    add_progress(f"   Error: {error_data.get('error', 'Unknown')}")

            # Check if we should restart
            if restart_count > max_restart_attempts:
                add_progress(f"❌ Maximum restart attempts ({max_restart_attempts}) reached")
                summary = "❌ Processing failed after maximum restart attempts"
                return summary, None, "\n".join(progress_messages)

            # Find latest checkpoint to resume from
            add_progress(f"🔄 Attempting automatic restart (attempt {restart_count + 1}/{max_restart_attempts + 1})...")

            checkpoint_dir = get_checkpoint_dir()
            checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))

            if checkpoints:
                # Find most recent checkpoint
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

                # Load checkpoint to verify it's valid
                try:
                    with open(latest_checkpoint, 'r') as f:
                        checkpoint_data = json.load(f)
                        processed_count = len(checkpoint_data.get('processed_files', []))
                        status = checkpoint_data.get('status', 'unknown')

                    if status != 'completed':
                        config['resume_checkpoint'] = str(latest_checkpoint)
                        add_progress(f"📂 Found checkpoint: {latest_checkpoint.name}")
                        add_progress(f"   Already processed: {processed_count} files")
                        add_progress(f"   Status: {status}")

                        # Small delay before restart
                        time.sleep(2)

                        # Continue loop to restart with checkpoint
                        continue
                    else:
                        add_progress("✅ All files already processed according to checkpoint")
                        break

                except Exception as checkpoint_error:
                    logger.error(f"Failed to load checkpoint: {checkpoint_error}")
                    add_progress(f"⚠️ Failed to load checkpoint: {checkpoint_error}")
            else:
                add_progress("⚠️ No checkpoint found - cannot auto-restart")
                break

        except Exception as e:
            error_msg = f"❌ Subprocess manager error: {str(e)}"
            logger.exception(error_msg)
            add_progress(error_msg)

            restart_count += 1
            if restart_count > max_restart_attempts:
                break

            add_progress(f"🔄 Retrying after error (attempt {restart_count + 1}/{max_restart_attempts + 1})...")
            time.sleep(2)

    # If we get here, we've exhausted retries or hit an error
    summary = "❌ Batch processing incomplete - see progress log for details"
    return summary, None, "\n".join(progress_messages)


def run_batch_processing(input_dir, output_dir, max_pages, is_ocr, formula_enable,
                        table_enable, images_enable, csv_tables, language, backend, url, md_only, fast_mode,
                        resume_checkpoint=None):
    """
    Wrapper function to run batch processing from Gradio interface.

    Args:
        resume_checkpoint: Optional path to checkpoint file to resume from
    """
    if not input_dir or not input_dir.strip():
        return "❌ Please specify an input directory", None, "Processing failed - no input directory specified"

    if not output_dir or not output_dir.strip():
        return "❌ Please specify an output directory", None, "Processing failed - no output directory specified"

    # Validate directories
    input_status = validate_directory_path(input_dir.strip())
    if not input_status.startswith("✓"):
        return f"❌ Input directory error: {input_status}", None, "Processing failed - invalid input directory"

    # Create output directory if it doesn't exist
    output_dir = output_dir.strip()
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        return f"❌ Cannot create output directory: {str(e)}", None, "Processing failed - cannot create output directory"
    
    # Create progress tracking
    progress_messages = []

    def progress_callback(message):
        # Handle multi-line messages (like status updates and error summaries)
        if isinstance(message, str):
            for line in message.split('\n'):
                if line.strip():  # Only add non-empty lines
                    progress_messages.append(line.strip())
        else:
            progress_messages.append(str(message))

        # Keep more messages to show full processing history
        return "\n".join(progress_messages[-50:])  # Keep last 50 messages
    
    # Run the batch processing
    try:
        # Use asyncio to run the batch processing
        import asyncio
        
        # Create event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the batch processing with auto-restart enabled
        result = loop.run_until_complete(
            batch_process_directory_with_auto_restart(
                input_dir=input_dir.strip(),
                output_dir=output_dir,
                max_pages=max_pages,
                is_ocr=is_ocr,
                formula_enable=formula_enable,
                table_enable=table_enable,
                images_enable=images_enable,
                language=language,
                backend=backend,
                server_url=url if url and url.strip() else None,
                md_only=md_only,
                fast_mode=fast_mode,
                csv_tables=csv_tables,
                progress_callback=progress_callback,
                resume_checkpoint=resume_checkpoint,
                max_restart_attempts=10  # Allow up to 10 auto-restarts
            )
        )
        
        total_files, successful_files, failed_files, zip_path = result
        
        # Generate summary
        if total_files == 0:
            summary = "❌ No PDF files found in the specified directory"
        else:
            success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
            summary = f"✅ Batch processing completed:\n"
            summary += f"• Total files: {total_files}\n"
            summary += f"• Successful: {successful_files} ({success_rate:.1f}%)\n"
            summary += f"• Failed: {failed_files}\n"
            if zip_path:
                summary += f"• Results archive: {os.path.basename(zip_path)}\n"
            if failed_files > 0:
                summary += f"• Failed files copied to: {os.path.join(output_dir, 'ERRORED')}\n"
            summary += f"• Output directory: {os.path.join(output_dir, 'MinerU_Outputs')}"
        
        progress_text = "\n".join(progress_messages)
        
        return summary, zip_path, progress_text
        
    except Exception as e:
        error_msg = f"❌ Batch processing failed: {str(e)}"
        logger.exception(error_msg)
        return error_msg, None, "\n".join(progress_messages) + f"\n{error_msg}"


# 更新界面函数
def update_interface(backend_choice):
    if backend_choice in ["vlm-transformers", "vlm-sglang-engine"]:
        return gr.update(visible=False), gr.update(visible=False)
    elif backend_choice in ["vlm-sglang-client"]:
        return gr.update(visible=True), gr.update(visible=False)
    elif backend_choice in ["pipeline"]:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        pass


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option(
    '--enable-example',
    'example_enable',
    type=bool,
    help="Enable example files for input."
         "The example files to be input need to be placed in the `example` folder within the directory where the command is currently executed.",
    default=True,
)
@click.option(
    '--enable-sglang-engine',
    'sglang_engine_enable',
    type=bool,
    help="Enable SgLang engine backend for faster processing.",
    default=False,
)
@click.option(
    '--enable-api',
    'api_enable',
    type=bool,
    help="Enable gradio API for serving the application.",
    default=True,
)
@click.option(
    '--max-convert-pages',
    'max_convert_pages',
    type=int,
    help="Set the maximum number of pages to convert from PDF to Markdown.",
    default=1000,
)
@click.option(
    '--server-name',
    'server_name',
    type=str,
    help="Set the server name for the Gradio app.",
    default=None,
)
@click.option(
    '--server-port',
    'server_port',
    type=int,
    help="Set the server port for the Gradio app.",
    default=None,
)
@click.option(
    '--latex-delimiters-type',
    'latex_delimiters_type',
    type=click.Choice(['a', 'b', 'all']),
    help="Set the type of LaTeX delimiters to use in Markdown rendering:"
         "'a' for type '$', 'b' for type '()[]', 'all' for both types.",
    default='all',
)
def main(ctx,
        example_enable, sglang_engine_enable, api_enable, max_convert_pages,
        server_name, server_port, latex_delimiters_type, **kwargs
):

    kwargs.update(arg_parse(ctx))

    if latex_delimiters_type == 'a':
        latex_delimiters = latex_delimiters_type_a
    elif latex_delimiters_type == 'b':
        latex_delimiters = latex_delimiters_type_b
    elif latex_delimiters_type == 'all':
        latex_delimiters = latex_delimiters_type_all
    else:
        raise ValueError(f"Invalid latex delimiters type: {latex_delimiters_type}.")

    if sglang_engine_enable:
        try:
            print("Start init SgLang engine...")
            from mineru.backend.vlm.vlm_analyze import ModelSingleton
            model_singleton = ModelSingleton()
            predictor = model_singleton.get_model(
                "sglang-engine",
                None,
                None,
                **kwargs
            )
            print("SgLang engine init successfully.")
        except Exception as e:
            logger.exception(e)

    suffixes = pdf_suffixes + image_suffixes
    with gr.Blocks() as demo:
        gr.HTML(header)
        
        with gr.Tabs():
            # Single File Processing Tab
            with gr.Tab("Single File"):
                with gr.Row():
                    with gr.Column(variant='panel', scale=5):
                        with gr.Row():
                            input_file = gr.File(label='Please upload a PDF or image', file_types=suffixes)
                        with gr.Row():
                            max_pages = gr.Slider(1, max_convert_pages, int(max_convert_pages/2), step=1, label='Max convert pages')
                        with gr.Row():
                            if sglang_engine_enable:
                                drop_list = ["pipeline", "vlm-sglang-engine"]
                                preferred_option = "vlm-sglang-engine"
                            else:
                                drop_list = ["pipeline", "vlm-transformers", "vlm-sglang-client"]
                                preferred_option = "pipeline"
                            backend = gr.Dropdown(drop_list, label="Backend", value=preferred_option)
                        with gr.Row(visible=False) as client_options:
                            url = gr.Textbox(label='Server URL', value='http://localhost:30000', placeholder='http://localhost:30000')
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                gr.Markdown("**Recognition Options:**")
                                formula_enable = gr.Checkbox(label='Enable formula recognition', value=True)
                                table_enable = gr.Checkbox(label='Enable table recognition', value=True)
                                images_enable = gr.Checkbox(label='Enable images in markdown output', value=False)
                                csv_tables = gr.Checkbox(label='📊 Convert tables to CSV format', value=True)
                            with gr.Column():
                                gr.Markdown("**Output Options:**")
                                md_only = gr.Checkbox(label='Output markdown only (no PDFs, JSONs, images)', value=True)
                                fast_mode = gr.Checkbox(label='🚀 Fast mode (disable formula recognition when MD-only)', value=True)
                                gr.Markdown("*Fast mode automatically disables formula recognition when MD-only is enabled for faster processing. Table recognition remains enabled for better content extraction.*", elem_classes=["text-sm", "text-gray-600"])
                            with gr.Column(visible=False) as ocr_options:
                                language = gr.Dropdown(all_lang, label='Language', value='en')
                                is_ocr = gr.Checkbox(label='Force enable OCR', value=False)
                        with gr.Row():
                            change_bu = gr.Button('Convert')
                            clear_bu = gr.ClearButton(value='Clear')
                        pdf_show = PDF(label='PDF preview', interactive=False, visible=True, height=800)
                        if example_enable:
                            example_root = os.path.join(os.getcwd(), 'examples')
                            if os.path.exists(example_root):
                                with gr.Accordion('Examples:'):
                                    gr.Examples(
                                        examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                                  _.endswith(tuple(suffixes))],
                                        inputs=input_file
                                    )

                    with gr.Column(variant='panel', scale=5):
                        output_file = gr.File(label='convert result', interactive=False)
                        with gr.Tabs():
                            with gr.Tab('Markdown rendering'):
                                md = gr.Markdown(label='Markdown rendering', height=1100, show_copy_button=True,
                                                 latex_delimiters=latex_delimiters,
                                                 line_breaks=True)
                            with gr.Tab('Markdown text'):
                                md_text = gr.TextArea(lines=45, show_copy_button=True)
            
            # Batch Processing Tab
            with gr.Tab("Batch Processing"):
                with gr.Row():
                    with gr.Column(variant='panel', scale=5):
                        gr.Markdown("## Batch Directory Processing")
                        gr.Markdown("Process all PDF files in a directory while maintaining folder structure.")

                        # Crash Recovery Section
                        with gr.Accordion("🔄 Crash Recovery - Resume from Checkpoint", open=False):
                            gr.Markdown("Resume processing from a previous session that was interrupted or crashed.")
                            with gr.Row():
                                resume_mode = gr.Checkbox(
                                    label='Enable Resume Mode',
                                    value=False,
                                    info='Check this to resume from a checkpoint'
                                )
                            with gr.Row():
                                refresh_checkpoints_btn = gr.Button('🔄 Refresh Checkpoint List', size='sm')
                            with gr.Row():
                                checkpoint_dropdown = gr.Dropdown(
                                    label='Select Checkpoint',
                                    choices=[],
                                    interactive=True,
                                    info='Select a checkpoint to resume from'
                                )
                            with gr.Row():
                                checkpoint_info_display = gr.Markdown(
                                    value="No checkpoint selected",
                                    label='Checkpoint Details'
                                )
                            with gr.Row():
                                load_checkpoint_settings_btn = gr.Button(
                                    '📥 Load Settings from Checkpoint',
                                    variant='secondary',
                                    size='sm'
                                )
                            gr.Markdown("*Resume mode will skip already processed files and continue from where it stopped.*")

                        with gr.Row():
                            batch_input_dir = gr.Textbox(
                                label='Input Directory Path',
                                placeholder='Enter the path to directory containing PDF files (e.g., C:\\Documents\\PDFs)',
                                lines=1
                            )
                        with gr.Row():
                            batch_output_dir = gr.Textbox(
                                label='Output Directory Path',
                                placeholder='Enter the path where processed files will be saved (e.g., C:\\Documents\\Output)',
                                lines=1
                            )
                        with gr.Row():
                            dir_status = gr.Textbox(
                                label='Directory Status',
                                interactive=False,
                                lines=2
                            )
                        with gr.Row():
                            validate_btn = gr.Button('Validate Directories', variant='secondary')
                        
                        with gr.Row():
                            batch_max_pages = gr.Slider(1, max_convert_pages, int(max_convert_pages/2), step=1, label='Max convert pages')
                        with gr.Row():
                            if sglang_engine_enable:
                                batch_drop_list = ["pipeline", "vlm-sglang-engine"]
                                batch_preferred_option = "vlm-sglang-engine"
                            else:
                                batch_drop_list = ["pipeline", "vlm-transformers", "vlm-sglang-client"]
                                batch_preferred_option = "pipeline"
                            batch_backend = gr.Dropdown(batch_drop_list, label="Backend", value=batch_preferred_option)
                        with gr.Row(visible=False) as batch_client_options:
                            batch_url = gr.Textbox(label='Server URL', value='http://localhost:30000', placeholder='http://localhost:30000')
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                gr.Markdown("**Recognition Options:**")
                                batch_formula_enable = gr.Checkbox(label='Enable formula recognition', value=True)
                                batch_table_enable = gr.Checkbox(label='Enable table recognition', value=True)
                                batch_images_enable = gr.Checkbox(label='Enable images in markdown output', value=False)
                                batch_csv_tables = gr.Checkbox(label='📊 Convert tables to CSV format', value=True)
                            with gr.Column():
                                gr.Markdown("**Output Options:**")
                                batch_md_only = gr.Checkbox(label='Output markdown only (no PDFs, JSONs, images)', value=True)
                                batch_fast_mode = gr.Checkbox(label='🚀 Fast mode (disable formula recognition when MD-only)', value=True)
                                gr.Markdown("*Fast mode automatically disables formula recognition when MD-only is enabled for faster processing. Table recognition remains enabled for better content extraction.*", elem_classes=["text-sm", "text-gray-600"])
                            with gr.Column(visible=False) as batch_ocr_options:
                                batch_language = gr.Dropdown(all_lang, label='Language', value='en')
                                batch_is_ocr = gr.Checkbox(label='Force enable OCR', value=False)
                        
                        with gr.Row():
                            batch_process_btn = gr.Button('Start Batch Processing', variant='primary', size='lg')
                            batch_clear_btn = gr.ClearButton(value='Clear', variant='secondary')
                        
                        gr.Markdown("**Note:** Processing will create a 'MinerU_Outputs' subdirectory in your output path with the same folder structure as your input directory. Failed files will be copied to an 'ERRORED' folder for troubleshooting.")
                    
                    with gr.Column(variant='panel', scale=5):
                        batch_summary = gr.Textbox(
                            label='Processing Summary',
                            interactive=False,
                            lines=8
                        )
                        batch_output_file = gr.File(label='Results Archive', interactive=False)
                        batch_progress = gr.TextArea(
                            label='Processing Progress',
                            lines=25,
                            autoscroll=True,
                            show_copy_button=True
                        )

        # Function to handle batch interface updates
        def update_batch_interface(backend_choice):
            if backend_choice in ["vlm-transformers", "vlm-sglang-engine"]:
                return gr.update(visible=False), gr.update(visible=False)
            elif backend_choice in ["vlm-sglang-client"]:
                return gr.update(visible=True), gr.update(visible=False)
            elif backend_choice in ["pipeline"]:
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False)
        
        # Function to validate directory paths
        def validate_directories(input_dir, output_dir):
            messages = []
            if input_dir:
                input_status = validate_directory_path(input_dir.strip())
                messages.append(f"Input: {input_status}")
            else:
                messages.append("Input: Please enter input directory path")

            if output_dir:
                if os.path.exists(output_dir.strip()):
                    if os.path.isdir(output_dir.strip()):
                        messages.append("Output: ✓ Directory exists and is valid")
                    else:
                        messages.append("Output: Path exists but is not a directory")
                else:
                    messages.append("Output: Directory will be created")
            else:
                messages.append("Output: Please enter output directory path")

            return "\n".join(messages)

        # Function to refresh checkpoint list
        def refresh_checkpoint_list():
            checkpoints = list_available_checkpoints()
            if checkpoints:
                choices = [display_name for display_name, _ in checkpoints]
                return gr.update(choices=choices, value=None)
            else:
                return gr.update(choices=["No checkpoints available"], value=None)

        # Function to handle checkpoint selection
        def on_checkpoint_selected(checkpoint_display_name):
            if not checkpoint_display_name or checkpoint_display_name == "No checkpoints available":
                return "No checkpoint selected"

            # Find the actual checkpoint path from display name
            checkpoints = list_available_checkpoints()
            checkpoint_path = None
            for display_name, path in checkpoints:
                if display_name == checkpoint_display_name:
                    checkpoint_path = path
                    break

            if not checkpoint_path:
                return "Invalid checkpoint selected"

            # Load and format checkpoint info
            info = load_checkpoint_info(checkpoint_path)
            return format_checkpoint_info(info)

        # Function to load settings from checkpoint
        def load_settings_from_checkpoint(checkpoint_display_name):
            if not checkpoint_display_name or checkpoint_display_name == "No checkpoints available":
                return tuple([gr.update() for _ in range(11)])  # Return no updates

            # Find the actual checkpoint path from display name
            checkpoints = list_available_checkpoints()
            checkpoint_path = None
            for display_name, path in checkpoints:
                if display_name == checkpoint_display_name:
                    checkpoint_path = path
                    break

            if not checkpoint_path:
                return tuple([gr.update() for _ in range(11)])

            # Load checkpoint info
            info = load_checkpoint_info(checkpoint_path)
            params = info.get('processing_params', {})

            if not params:
                return tuple([gr.update() for _ in range(11)])

            # Return updates for all fields
            return (
                gr.update(value=params.get('input_dir', '')),           # batch_input_dir
                gr.update(value=params.get('output_dir', '')),          # batch_output_dir
                gr.update(value=params.get('max_pages', 500)),          # batch_max_pages
                gr.update(value=params.get('backend', 'pipeline')),     # batch_backend
                gr.update(value=params.get('server_url', '')),          # batch_url
                gr.update(value=params.get('is_ocr', False)),           # batch_is_ocr
                gr.update(value=params.get('formula_enable', True)),    # batch_formula_enable
                gr.update(value=params.get('table_enable', True)),      # batch_table_enable
                gr.update(value=params.get('images_enable', False)),    # batch_images_enable
                gr.update(value=params.get('csv_tables', True)),        # batch_csv_tables
                gr.update(value=params.get('language', 'en')),          # batch_language
                gr.update(value=params.get('md_only', True)),           # batch_md_only
                gr.update(value=params.get('fast_mode', True))          # batch_fast_mode
            )

        # Function to handle batch processing with resume support
        def run_batch_with_resume(input_dir, output_dir, max_pages, is_ocr, formula_enable,
                                  table_enable, images_enable, csv_tables, language, backend, url,
                                  md_only, fast_mode, resume_enabled, checkpoint_display_name):
            # Determine checkpoint path if resume mode is enabled
            checkpoint_path = None
            if resume_enabled and checkpoint_display_name and checkpoint_display_name != "No checkpoints available":
                checkpoints = list_available_checkpoints()
                for display_name, path in checkpoints:
                    if display_name == checkpoint_display_name:
                        checkpoint_path = path
                        break

            # Call the subprocess-protected batch processing function
            # This version will automatically restart on crashes (core dumps, SIGSEGV, etc.)
            return run_batch_processing_with_subprocess_protection(
                input_dir, output_dir, max_pages, is_ocr, formula_enable,
                table_enable, images_enable, csv_tables, language, backend, url,
                md_only, fast_mode, resume_checkpoint=checkpoint_path,
                max_restart_attempts=10  # Allow up to 10 automatic restarts
            )

        # Function to handle fast mode toggle for single file processing
        def handle_fast_mode_single(md_only_val, fast_mode_val):
            if fast_mode_val and md_only_val:
                # Fast mode enabled with MD-only: disable formula recognition but keep tables
                return (gr.update(value=False), gr.update(), gr.update(value=False, interactive=False))  # formula_enable disabled, table_enable unchanged, images_enable disabled
            elif md_only_val:
                # MD-only mode: disable images but allow manual control of formula/table
                return (gr.update(), gr.update(), gr.update(value=False, interactive=False))  # images_enable disabled
            else:
                # Normal mode: enable all controls
                return (gr.update(), gr.update(), gr.update(interactive=True))

        # Function to handle MD-only toggle for single file processing
        def handle_md_only_single(md_only_val):
            if md_only_val:
                # MD-only mode: disable images
                return gr.update(value=False, interactive=False)  # images_enable
            else:
                # Normal mode: enable images control
                return gr.update(interactive=True)

        # Function to handle fast mode toggle for batch processing
        def handle_fast_mode_batch(md_only_val, fast_mode_val):
            if fast_mode_val and md_only_val:
                # Fast mode enabled with MD-only: disable formula recognition but keep tables
                return (gr.update(value=False), gr.update(), gr.update(value=False, interactive=False))  # batch_formula_enable disabled, batch_table_enable unchanged, batch_images_enable disabled
            elif md_only_val:
                # MD-only mode: disable images but allow manual control of formula/table
                return (gr.update(), gr.update(), gr.update(value=False, interactive=False))  # batch_images_enable disabled
            else:
                # Normal mode: enable all controls
                return (gr.update(), gr.update(), gr.update(interactive=True))

        # Function to handle MD-only toggle for batch processing
        def handle_md_only_batch(md_only_val):
            if md_only_val:
                # MD-only mode: disable images
                return gr.update(value=False, interactive=False)  # batch_images_enable
            else:
                # Normal mode: enable images control
                return gr.update(interactive=True)

        # 添加事件处理 - Single File Tab
        backend.change(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options],
            api_name=False
        )
        # 添加demo.load事件，在页面加载时触发一次界面更新
        demo.load(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options],
            api_name=False
        )

        # Load available checkpoints on page load
        demo.load(
            fn=refresh_checkpoint_list,
            inputs=[],
            outputs=[checkpoint_dropdown],
            api_name=False
        )

        # Single file fast mode handlers
        fast_mode.change(
            fn=handle_fast_mode_single,
            inputs=[md_only, fast_mode],
            outputs=[formula_enable, table_enable, images_enable],
            api_name=False
        )

        md_only.change(
            fn=handle_fast_mode_single,
            inputs=[md_only, fast_mode],
            outputs=[formula_enable, table_enable, images_enable],
            api_name=False
        )
        clear_bu.add([input_file, md, pdf_show, md_text, output_file, is_ocr, md_only, fast_mode, formula_enable, table_enable, images_enable, csv_tables])

        # Batch Processing Tab Event Handlers
        batch_backend.change(
            fn=update_batch_interface,
            inputs=[batch_backend],
            outputs=[batch_client_options, batch_ocr_options],
            api_name=False
        )
        
        # Directory validation
        validate_btn.click(
            fn=validate_directories,
            inputs=[batch_input_dir, batch_output_dir],
            outputs=[dir_status],
            api_name=False
        )
        
        # Auto-validate on directory input change
        batch_input_dir.change(
            fn=validate_directories,
            inputs=[batch_input_dir, batch_output_dir],
            outputs=[dir_status],
            api_name=False
        )
        
        batch_output_dir.change(
            fn=validate_directories,
            inputs=[batch_input_dir, batch_output_dir],
            outputs=[dir_status],
            api_name=False
        )

        # Batch processing fast mode handlers
        batch_fast_mode.change(
            fn=handle_fast_mode_batch,
            inputs=[batch_md_only, batch_fast_mode],
            outputs=[batch_formula_enable, batch_table_enable, batch_images_enable],
            api_name=False
        )

        batch_md_only.change(
            fn=handle_fast_mode_batch,
            inputs=[batch_md_only, batch_fast_mode],
            outputs=[batch_formula_enable, batch_table_enable, batch_images_enable],
            api_name=False
        )
        
        # Clear button for batch processing
        batch_clear_btn.add([
            batch_input_dir, batch_output_dir, dir_status, batch_summary,
            batch_output_file, batch_progress, batch_is_ocr, batch_md_only, batch_fast_mode,
            batch_formula_enable, batch_table_enable, batch_images_enable, batch_csv_tables
        ])

        if api_enable:
            api_name = None
        else:
            api_name = False

        # Single file processing
        input_file.change(fn=to_pdf, inputs=input_file, outputs=pdf_show, api_name=api_name)
        change_bu.click(
            fn=to_markdown,
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, language, backend, url, images_enable, md_only, fast_mode, csv_tables],
            outputs=[md, md_text, output_file, pdf_show],
            api_name=api_name
        )
        
        # Checkpoint management event handlers
        refresh_checkpoints_btn.click(
            fn=refresh_checkpoint_list,
            inputs=[],
            outputs=[checkpoint_dropdown],
            api_name=False
        )

        checkpoint_dropdown.change(
            fn=on_checkpoint_selected,
            inputs=[checkpoint_dropdown],
            outputs=[checkpoint_info_display],
            api_name=False
        )

        load_checkpoint_settings_btn.click(
            fn=load_settings_from_checkpoint,
            inputs=[checkpoint_dropdown],
            outputs=[
                batch_input_dir, batch_output_dir, batch_max_pages, batch_backend,
                batch_url, batch_is_ocr, batch_formula_enable, batch_table_enable,
                batch_images_enable, batch_csv_tables, batch_language, batch_md_only,
                batch_fast_mode
            ],
            api_name=False
        )

        # Batch processing with resume support
        batch_process_btn.click(
            fn=run_batch_with_resume,
            inputs=[
                batch_input_dir, batch_output_dir, batch_max_pages, batch_is_ocr,
                batch_formula_enable, batch_table_enable, batch_images_enable, batch_csv_tables,
                batch_language, batch_backend, batch_url, batch_md_only, batch_fast_mode,
                resume_mode, checkpoint_dropdown
            ],
            outputs=[batch_summary, batch_output_file, batch_progress],
            api_name=api_name
        )

    demo.launch(server_name=server_name, server_port=server_port, show_api=api_enable)


if __name__ == '__main__':
    main()