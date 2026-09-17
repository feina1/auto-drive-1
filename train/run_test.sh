#!/bin/bash
cd "/home/wl/下载/wlwork/auto-drive-1/train"
/home/wl/miniconda3/envs/autodrive/bin/python test_closed_loop.py 2>&1 | tee /tmp/closed_loop_v2.log
