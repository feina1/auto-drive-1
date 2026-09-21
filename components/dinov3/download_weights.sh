#!/usr/bin/env bash
# ============================================================
# DINOv3 权重下载脚本
#
# 从 HuggingFace 下载 DINOv3 ViT-S/16 预训练权重到本地。
# 模型: facebook/dinov3-vits16-pretrain-lvd1689m
# 大小: ~83MB (model.safetensors)
#
# 用法:
#   bash components/dinov3/download_weights.sh
#
# 依赖:
#   pip install huggingface_hub
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights/vits16"
MODEL_ID="${DINOV3_MODEL_ID:-facebook/dinov3-vits16-pretrain-lvd1689m}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[DINOv3] 权重下载脚本${NC}"
echo "  模型 ID: ${MODEL_ID}"
echo "  目标目录: ${WEIGHTS_DIR}"
echo ""

# 检查是否已下载
if [ -f "${WEIGHTS_DIR}/model.safetensors" ]; then
    echo -e "${YELLOW}[DINOv3] 权重已存在: ${WEIGHTS_DIR}/model.safetensors${NC}"
    read -p "  是否重新下载? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo -e "${GREEN}[DINOv3] 跳过下载${NC}"
        exit 0
    fi
fi

# 检查 huggingface_hub 是否可用
python3 -c "import huggingface_hub" 2>/dev/null || {
    echo -e "${YELLOW}[DINOv3] 安装 huggingface_hub...${NC}"
    pip install huggingface_hub
}

# 下载
mkdir -p "${WEIGHTS_DIR}"
echo -e "${GREEN}[DINOv3] 开始下载...${NC}"
python3 -c "
from huggingface_hub import snapshot_download
import os

model_id = '${MODEL_ID}'
local_dir = '${WEIGHTS_DIR}'

# 只下载必要文件 (跳过 ONNX/TF 等)
allow_patterns = [
    'config.json',
    'preprocessor_config.json',
    'model.safetensors',
    'LICENSE.md',
    'README.md',
]

path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    allow_patterns=allow_patterns,
)
print(f'下载完成: {path}')
"

echo ""
echo -e "${GREEN}[DINOv3] 下载完成!${NC}"
echo "  路径: ${WEIGHTS_DIR}"
ls -lh "${WEIGHTS_DIR}"
