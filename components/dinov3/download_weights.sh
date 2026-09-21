#!/usr/bin/env bash
# ============================================================
# DINOv3 权重下载脚本
#
# 从 ModelScope 下载 DINOv2 ViT-S/16 预训练权重到本地。
# 模型: AI-ModelScope/dinov2-vit-small-patch16-pretrain
# 大小: ~83MB (model.safetensors)
#
# 用法:
#   bash components/dinov3/download_weights.sh
#
# 依赖:
#   pip install modelscope
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights/vits16"

# ModelScope 模型 ID (国内可访问, 镜像 HuggingFace)
MODELSCOPE_ID="${DINOV3_MODEL_ID:-facebook/dinov3-vits16-pretrain-lvd1689m}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[DINOv3] 权重下载脚本 (ModelScope)${NC}"
echo "  模型 ID: ${MODELSCOPE_ID}"
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

# 检查 modelscope 是否可用
python3 -c "import modelscope" 2>/dev/null || {
    echo -e "${YELLOW}[DINOv3] 安装 modelscope...${NC}"
    pip install modelscope
}

# 下载
mkdir -p "${WEIGHTS_DIR}"
echo -e "${GREEN}[DINOv3] 开始从 ModelScope 下载...${NC}"
python3 << EOF
from modelscope import snapshot_download
import os
import shutil

model_id = '${MODELSCOPE_ID}'
cache_dir = '${WEIGHTS_DIR}'

# 下载到临时缓存
path = snapshot_download(
    model_id=model_id,
    cache_dir=cache_dir,
)
print(f'下载完成: {path}')

# ModelScope 会创建深层目录结构，需要把文件移到顶层
# 查找包含 model.safetensors 的目录
for root, dirs, files in os.walk(cache_dir):
    if 'model.safetensors' in files:
        # 把所有文件移到顶层
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(cache_dir, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        print(f'文件已移到: {cache_dir}')
        break

# 清理临时目录
for d in ['models', '.lock', '.cache']:
    p = os.path.join(cache_dir, d)
    if os.path.exists(p):
        shutil.rmtree(p)
EOF

echo ""
echo -e "${GREEN}[DINOv3] 下载完成!${NC}"
echo "  路径: ${WEIGHTS_DIR}"
ls -lh "${WEIGHTS_DIR}"
