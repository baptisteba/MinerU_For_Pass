# Copyright (c) Opendatalab. All rights reserved.

# RTX 5090 Compatibility Fix
# Import rtx50_compat before any PyTorch imports to enable sm_120 CUDA capability
try:
    import rtx50_compat
except ImportError:
    pass  # rtx50_compat is optional, only needed for RTX 50-series GPUs
