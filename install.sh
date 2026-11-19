#!/usr/bin/env bash
# 用法: ./install.sh
# 创建 conda 环境 autodrive (Python 3.10), 安装 MetaDrive 和所有依赖
set -euo pipefail

ENV_NAME="autodrive"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METADRIVE_SRC="${SCRIPT_DIR}/metadrive-main"

# 创建环境 (已存在则重建)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境已存在, 重建..."
    conda env remove -n "${ENV_NAME}" -y
fi
conda create -n "${ENV_NAME}" python=3.10 -y

# 用新环境的 python
CONDA_PREFIX="$(conda info --base)/envs/${ENV_NAME}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# 安装 MetaDrive (本地源码) + 项目依赖
pip install --upgrade pip
pip install -e "${METADRIVE_SRC}[gym]"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 下载 3D 资源 (代码在本地, 但车辆模型/贴图等需要从 GitHub 下载)
ASSETS_DIR="${METADRIVE_SRC}/metadrive/assets"
if [ ! -f "${ASSETS_DIR}/version.txt" ]; then
    echo "下载 MetaDrive 3D 资源..."
    cd "${METADRIVE_SRC}"
    python -m metadrive.pull_asset
    cd "${SCRIPT_DIR}"
fi

# 验证
python -c "from metadrive.envs.metadrive_env import MetaDriveEnv; print('OK')"

echo ""
echo "安装完成! 使用方式:"
echo "  conda activate autodrive"
echo "  python record_pid_images.py"
