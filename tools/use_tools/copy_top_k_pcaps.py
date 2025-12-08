import os
import shutil
from pathlib import Path

def copy_top_k_pcaps(src_dir, dst_dir, k=300):
    """
    将原始目录中每个类别文件夹下前k个pcap文件拷贝到目标目录，保留类别目录结构。

    :param src_dir: 原始目录，包含多个类别子目录
    :param dst_dir: 目标目录
    :param k: 每个类别最多复制的pcap数量
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")

    # 遍历每个“类别文件夹”
    for category_dir in src_dir.iterdir():
        if category_dir.is_dir():
            pcap_files = sorted([f for f in category_dir.glob("*.pcap")])[:k]

            # 构建目标类别文件夹路径
            target_category_dir = dst_dir / category_dir.name
            target_category_dir.mkdir(parents=True, exist_ok=True)

            # 复制文件
            for pcap_file in pcap_files:
                target_file = target_category_dir / pcap_file.name
                shutil.copy2(pcap_file, target_file)
                print(f"Copied: {pcap_file} -> {target_file}")



if __name__ == '__main__':
    copy_top_k_pcaps("D:/数据集/self_tor_24", "G:/self_tor")
