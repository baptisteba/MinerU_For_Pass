#!/usr/bin/env python3
"""
Batch processing worker script that runs as a subprocess.
This allows the main Gradio app to survive worker crashes and automatically restart processing.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mineru.cli.gradio_app import batch_process_directory
from loguru import logger


async def run_worker(
    input_dir: str,
    output_dir: str,
    max_pages: int,
    is_ocr: bool,
    formula_enable: bool,
    table_enable: bool,
    images_enable: bool,
    language: str,
    backend: str,
    server_url: Optional[str],
    md_only: bool,
    fast_mode: bool,
    csv_tables: bool,
    resume_checkpoint: Optional[str]
):
    """
    Worker function that performs batch processing in a subprocess.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        # Simple progress callback that prints to stdout
        def progress_callback(message):
            print(f"PROGRESS: {message}", flush=True)

        # Run batch processing
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
            resume_checkpoint=resume_checkpoint
        )

        total_files, successful_files, failed_files, zip_path = result

        # Write result to stdout as JSON for parent to read
        result_data = {
            "status": "completed",
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "zip_path": zip_path
        }
        print(f"RESULT: {json.dumps(result_data)}", flush=True)

        return 0 if failed_files == 0 else 1

    except Exception as e:
        logger.exception(f"Worker process error: {e}")
        error_data = {
            "status": "error",
            "error": str(e)
        }
        print(f"ERROR: {json.dumps(error_data)}", flush=True)
        return 1


def main():
    """Main entry point for worker subprocess."""
    if len(sys.argv) != 2:
        print("ERROR: Worker requires config file path as argument", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Extract configuration
        input_dir = config['input_dir']
        output_dir = config['output_dir']
        max_pages = config.get('max_pages', 500)
        is_ocr = config.get('is_ocr', False)
        formula_enable = config.get('formula_enable', True)
        table_enable = config.get('table_enable', True)
        images_enable = config.get('images_enable', False)
        language = config.get('language', 'en')
        backend = config.get('backend', 'pipeline')
        server_url = config.get('server_url')
        md_only = config.get('md_only', False)
        fast_mode = config.get('fast_mode', False)
        csv_tables = config.get('csv_tables', True)
        resume_checkpoint = config.get('resume_checkpoint')

        # Run worker
        exit_code = asyncio.run(run_worker(
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
            resume_checkpoint=resume_checkpoint
        ))

        sys.exit(exit_code)

    except Exception as e:
        logger.exception(f"Worker initialization error: {e}")
        print(f"ERROR: Worker initialization failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
