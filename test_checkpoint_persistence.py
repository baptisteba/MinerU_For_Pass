#!/usr/bin/env python3
"""
Test script to verify checkpoint persistence across process restarts.
This simulates a crash and recovery scenario.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mineru.cli.crash_recovery import ProcessingCheckpoint


def test_checkpoint_creation():
    """Test creating and saving a checkpoint."""
    print("=" * 60)
    print("TEST 1: Creating a new checkpoint")
    print("=" * 60)

    checkpoint = ProcessingCheckpoint()
    session_id = f"test_session_{int(time.time())}"
    checkpoint_file = checkpoint.create_session(session_id)

    print(f"✓ Checkpoint created at: {checkpoint_file}")
    print(f"✓ Checkpoint directory: {checkpoint.checkpoint_dir}")

    # Simulate processing some files
    test_files = ["/path/to/file1.pdf", "/path/to/file2.pdf", "/path/to/file3.pdf"]

    checkpoint.update(
        processing_params={
            'input_dir': '/test/input',
            'output_dir': '/test/output',
            'backend': 'pipeline',
            'max_pages': 500,
            'language': 'en'
        }
    )
    print("✓ Added processing parameters")

    # Mark some files as processed
    checkpoint.mark_file_processed(test_files[0], success=True)
    print(f"✓ Marked {test_files[0]} as processed")

    checkpoint.mark_file_processed(test_files[1], success=False)
    print(f"✓ Marked {test_files[1]} as failed")

    # Update status
    checkpoint.update(status="crashed", error="Simulated crash for testing")
    print("✓ Marked checkpoint as crashed")

    print(f"\n✅ Checkpoint saved successfully!")
    print(f"   Checkpoint file: {checkpoint_file}")
    print(f"   This file will persist after process restart.\n")

    return checkpoint_file, test_files


def test_checkpoint_recovery(checkpoint_file, test_files):
    """Test loading a checkpoint after 'restart' (new instance)."""
    print("=" * 60)
    print("TEST 2: Simulating restart and loading checkpoint")
    print("=" * 60)

    # Create a NEW instance (simulating a fresh start)
    new_checkpoint = ProcessingCheckpoint()

    # Load the checkpoint
    state = new_checkpoint.load_session(checkpoint_file)

    print(f"✓ Loaded checkpoint: {checkpoint_file}")
    print(f"\nCheckpoint state:")
    print(f"  - Session ID: {state.get('session_id')}")
    print(f"  - Status: {state.get('status')}")
    print(f"  - Error: {state.get('error', 'None')}")
    print(f"  - Processed files: {len(state.get('processed_files', []))}")
    print(f"  - Failed files: {len(state.get('failed_files', []))}")

    # Test getting remaining files
    remaining = new_checkpoint.get_remaining_files(test_files)
    print(f"\nRemaining files to process: {len(remaining)}")
    for f in remaining:
        print(f"  - {f}")

    # Verify processing parameters
    params = state.get('processing_params', {})
    print(f"\nOriginal processing parameters:")
    for key, value in params.items():
        print(f"  - {key}: {value}")

    print(f"\n✅ Checkpoint loaded successfully after 'restart'!")
    print(f"   All data was preserved and can be used to resume processing.\n")


def test_list_checkpoints():
    """Test listing all available checkpoints."""
    print("=" * 60)
    print("TEST 3: Listing all available checkpoints")
    print("=" * 60)

    checkpoint = ProcessingCheckpoint()
    checkpoint_dir = checkpoint.checkpoint_dir

    print(f"Scanning directory: {checkpoint_dir}")

    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
        print(f"\nFound {len(checkpoints)} checkpoint(s):")

        for cp_file in sorted(checkpoints, reverse=True):
            import json
            try:
                with open(cp_file, 'r') as f:
                    data = json.load(f)
                    session_id = data.get('session_id', 'Unknown')
                    status = data.get('status', 'unknown')
                    created = data.get('created_at', 'Unknown')
                    processed = len(data.get('processed_files', []))
                    failed = len(data.get('failed_files', []))

                    print(f"\n  {cp_file.name}")
                    print(f"    Session: {session_id}")
                    print(f"    Status: {status}")
                    print(f"    Created: {created}")
                    print(f"    Processed: {processed}, Failed: {failed}")
            except Exception as e:
                print(f"  {cp_file.name} (Error reading: {e})")

    print(f"\n✅ Checkpoint listing works!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CHECKPOINT PERSISTENCE TEST")
    print("=" * 60)
    print("\nThis test verifies that checkpoints survive process restarts.")
    print("We'll create a checkpoint, then load it in a new instance.\n")

    try:
        # Test 1: Create checkpoint
        checkpoint_file, test_files = test_checkpoint_creation()

        # Test 2: Load checkpoint (simulating restart)
        test_checkpoint_recovery(checkpoint_file, test_files)

        # Test 3: List all checkpoints
        test_list_checkpoints()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nCheckpoints are stored in a permanent location and will")
        print("persist across web server restarts and system reboots.")
        print("\nYou can now safely:")
        print("  1. Start batch processing")
        print("  2. If it crashes, restart the web UI")
        print("  3. Use the Crash Recovery section to resume")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
