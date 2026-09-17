#!/bin/bash
# 清理旧数据 + 检查磁盘空间
echo "=== 磁盘空间 ==="
df -h /

echo ""
echo "=== 旧数据大小 ==="
du -sh "$HOME/下载/wlwork/auto-drive-1/records_pid/" 2>/dev/null || echo "records_pid: 不存在"
du -sh "$HOME/下载/wlwork/auto-drive-1/records_perturbed/" 2>/dev/null || echo "records_perturbed: 不存在"
du -sh "$HOME/下载/wlwork/auto-drive-1/train/checkpoints/" 2>/dev/null || echo "checkpoints: 不存在"

echo ""
echo "=== 清理旧数据 ==="
rm -rf "$HOME/下载/wlwork/auto-drive-1/records_pid/"
rm -rf "$HOME/下载/wlwork/auto-drive-1/records_perturbed/"
rm -rf "$HOME/下载/wlwork/auto-drive-1/train/checkpoints/"
echo "清理完成"

echo ""
echo "=== 清理后磁盘空间 ==="
df -h /
