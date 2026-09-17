#!/usr/bin/env python3
"""Quick runner for closed-loop test - avoids terminal encoding issues."""
import os, subprocess, sys

train_dir = os.path.dirname(os.path.abspath(__file__))
python = os.path.expanduser("~/miniconda3/envs/autodrive/bin/python")
script = os.path.join(train_dir, "test_closed_loop.py")

print(f"Running: {python} {script}")
print(f"CWD: {train_dir}")
os.chdir(train_dir)
sys.exit(subprocess.call([python, script]))
