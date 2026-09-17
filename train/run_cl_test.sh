#!/usr/bin/env bash
# Kill any existing test processes
pkill -9 -f test_closed_loop 2>/dev/null
sleep 2

# Run the test
cd "/home/wl/下载/wlwork/auto-drive-1/train"
exec /home/wl/miniconda3/envs/autodrive/bin/python test_closed_loop.py 2>&1
