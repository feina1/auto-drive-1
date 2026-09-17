#!/bin/bash
PROJ="$HOME/下载/wlwork/auto-drive-1"
cd "$PROJ/train" || exit 1
PYTHON="$HOME/miniconda3/envs/autodrive/bin/python"
exec "$PYTHON" test_closed_loop.py
