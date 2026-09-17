#!/usr/bin/env python3
"""Wrapper to run closed-loop test."""
import subprocess, sys
sys.exit(subprocess.call([
    "/home/wl/miniconda3/envs/autodrive/bin/python",
    "/home/wl/下载/wlwork/auto-drive-1/train/test_closed_loop.py"
]))
