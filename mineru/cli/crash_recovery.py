"""
Crash recovery and checkpoint system for MinerU batch processing.
Provides fault-tolerant processing with automatic resume capability.
"""

import os
import json
import pickle
import signal
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import threading
import multiprocessing
from contextlib import contextmanager


class ProcessingCheckpoint:
    """Manages checkpointing for batch PDF processing to enable crash recovery."""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files. If None, uses temp directory.
        """
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path(tempfile.gettempdir()) / "mineru_checkpoints"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = None
        self.state = {}

    def create_session(self, session_id: str) -> str:
        """Create a new checkpoint session.

        Args:
            session_id: Unique identifier for this processing session

        Returns:
            Path to the checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{session_id}_{timestamp}.json"
        self.state = {
            "session_id": session_id,
            "created_at": timestamp,
            "status": "started",
            "processed_files": [],
            "failed_files": [],
            "current_file": None,
            "total_files": 0,
            "last_update": timestamp,
            "processing_params": {}
        }
        self._save()
        return str(self.checkpoint_file)

    def load_session(self, checkpoint_file: str) -> Dict:
        """Load an existing checkpoint session.

        Args:
            checkpoint_file: Path to the checkpoint file

        Returns:
            The checkpoint state dictionary
        """
        self.checkpoint_file = Path(checkpoint_file)
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.state = json.load(f)
        return self.state

    def update(self, **kwargs):
        """Update checkpoint state with given parameters."""
        self.state.update(kwargs)
        self.state["last_update"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save()

    def mark_file_processed(self, file_path: str, success: bool = True):
        """Mark a file as processed.

        Args:
            file_path: Path to the processed file
            success: Whether processing was successful
        """
        if success:
            if file_path not in self.state["processed_files"]:
                self.state["processed_files"].append(file_path)
        else:
            if file_path not in self.state["failed_files"]:
                self.state["failed_files"].append(file_path)

        # Clear current file
        if self.state.get("current_file") == file_path:
            self.state["current_file"] = None

        self._save()

    def get_remaining_files(self, all_files: List[str]) -> List[str]:
        """Get list of files that haven't been processed yet.

        Args:
            all_files: Complete list of files to process

        Returns:
            List of files that still need processing
        """
        processed = set(self.state.get("processed_files", []))
        failed = set(self.state.get("failed_files", []))
        completed = processed | failed

        return [f for f in all_files if f not in completed]

    def _save(self):
        """Save checkpoint state to disk."""
        if self.checkpoint_file:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.state, f, indent=2)

    def cleanup(self):
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_file and self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


class SafeProcessor:
    """Provides safe execution wrapper with timeout and crash protection."""

    @staticmethod
    @contextmanager
    def timeout(seconds: int):
        """Context manager for timeout protection.

        Args:
            seconds: Timeout in seconds
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Processing timeout after {seconds} seconds")

        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    @staticmethod
    def safe_execute(func, *args, timeout_seconds: int = 300, **kwargs):
        """Execute function with crash protection and timeout.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            timeout_seconds: Maximum execution time in seconds
            **kwargs: Keyword arguments for the function

        Returns:
            Tuple of (success: bool, result: Any, error: str)
        """
        try:
            # Use multiprocessing for true isolation
            with multiprocessing.Pool(processes=1) as pool:
                async_result = pool.apply_async(func, args, kwargs)
                try:
                    result = async_result.get(timeout=timeout_seconds)
                    return True, result, None
                except multiprocessing.TimeoutError:
                    pool.terminate()
                    return False, None, f"Processing timeout after {timeout_seconds} seconds"
                except Exception as e:
                    return False, None, str(e)
        except Exception as e:
            logger.error(f"Failed to execute safely: {e}")
            return False, None, str(e)


class TableProcessingSafeguard:
    """Specific safeguards for table processing which often causes crashes."""

    @staticmethod
    def safe_table_predict(table_model, table_res_list, max_retries: int = 3):
        """Safely execute table prediction with retry logic.

        Args:
            table_model: The table prediction model
            table_res_list: List of table resources to process
            max_retries: Maximum number of retry attempts

        Returns:
            Processing result or None if failed
        """
        for attempt in range(max_retries):
            try:
                # Process in smaller batches to reduce memory pressure
                batch_size = max(1, len(table_res_list) // 4)

                for i in range(0, len(table_res_list), batch_size):
                    batch = table_res_list[i:i+batch_size]

                    # Wrap each batch in timeout protection
                    try:
                        with SafeProcessor.timeout(120):  # 2 minute timeout per batch
                            table_model.batch_predict(batch)
                    except TimeoutError:
                        logger.warning(f"Table batch {i//batch_size + 1} timed out, skipping...")
                        # Mark these tables as failed but continue
                        for item in batch:
                            item['table_res']['html'] = '<p>Table processing timeout</p>'
                    except Exception as e:
                        logger.warning(f"Table batch {i//batch_size + 1} failed: {e}")
                        for item in batch:
                            item['table_res']['html'] = '<p>Table processing failed</p>'

                return True

            except Exception as e:
                logger.error(f"Table processing attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for table processing")
                    return False

        return False


def create_safe_batch_processor(original_func):
    """Decorator to wrap batch processing functions with crash protection.

    Args:
        original_func: Original processing function to wrap

    Returns:
        Wrapped function with crash protection
    """
    def safe_wrapper(*args, **kwargs):
        checkpoint = ProcessingCheckpoint()

        # Extract key parameters
        input_dir = kwargs.get('input_dir', '')
        output_dir = kwargs.get('output_dir', '')

        # Create session
        session_id = Path(input_dir).name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint.create_session(session_id)

        logger.info(f"Created checkpoint file: {checkpoint_file}")

        try:
            # Store processing params in checkpoint
            checkpoint.update(processing_params=kwargs)

            # Call original function with checkpoint
            kwargs['checkpoint'] = checkpoint
            result = original_func(*args, **kwargs)

            # Clean up checkpoint on success
            checkpoint.update(status="completed")
            logger.info("Batch processing completed successfully")

            return result

        except Exception as e:
            logger.error(f"Batch processing crashed: {e}")
            checkpoint.update(status="crashed", error=str(e))

            # Attempt to recover
            logger.info("Attempting to resume from checkpoint...")
            remaining_files = checkpoint.get_remaining_files(kwargs.get('pdf_files', []))

            if remaining_files:
                logger.info(f"Resuming processing for {len(remaining_files)} remaining files")
                kwargs['pdf_files'] = remaining_files
                kwargs['resume_from_checkpoint'] = True

                # Recursive call with remaining files
                return safe_wrapper(*args, **kwargs)
            else:
                raise

    return safe_wrapper


# Export key components
__all__ = [
    'ProcessingCheckpoint',
    'SafeProcessor',
    'TableProcessingSafeguard',
    'create_safe_batch_processor'
]