#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 参数1：原始数据目录（支持 .txt/.pcap/.pcapng），默认是项目根目录下的 rawdata
RAWDATA_DIR="${1:-$ROOT_DIR/rawdata}"
# 参数2：输出的模型输入文件（.npz），默认输出到 dataset/bytedance_input.npz
OUTPUT_NPZ="${2:-$ROOT_DIR/dataset/bytedance_input.npz}"
# 参数3：每个样本最多保留多少个包（对应模型的 max_packet_num），默认 20
MAX_PACKET_NUM="${3:-20}"
# 参数4：每个包保留多少个字节（对应模型的 byte_dim），默认 80
BYTE_DIM="${4:-80}"
# 参数5：最少有效包数量阈值（低于该值的样本会被跳过），默认 1
MIN_PACKETS="${5:-1}"

cd "$ROOT_DIR"
python tools/rawdata_to_bytedance_input.py \
  --rawdata-dir "$RAWDATA_DIR" \
  --output "$OUTPUT_NPZ" \
  --max-packet-num "$MAX_PACKET_NUM" \
  --byte-dim "$BYTE_DIM" \
  --min-packets "$MIN_PACKETS"

echo "Finished: ${OUTPUT_NPZ}"
