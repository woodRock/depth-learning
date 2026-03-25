#!/usr/bin/env python3
"""
Backward compatibility wrapper for train_fusion.py.
Deprecated: Use 'python train.py fusion' instead.
"""

import sys
import os

# Add ml directory to path
sys.path.insert(0, os.path.dirname(__file__))

from train import main as train_main

# Simulate command-line arguments
sys.argv = [sys.argv[0], "fusion"] + sys.argv[1:]

if __name__ == "__main__":
    train_main()
