import os
import shutil


def copy_top_pcap(source_folder, destination_folder):
    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)

    # 获取源文件夹下的所有子文件夹
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

    for subfolder in subfolders:
        # 获取子文件夹下的所有 pcap 文件
        pcap_files = [file for file in os.listdir(subfolder) if file.endswith('.pcap')]

        # 按文件大小排序
        sorted_pcap_files = sorted(pcap_files, key=lambda f: os.path.getsize(os.path.join(subfolder, f)), reverse=True)

        # 取前100个或者所有文件，确保最小的大于70KB
        top_files = sorted_pcap_files[5:21] if len(sorted_pcap_files) >= 15 else sorted_pcap_files

        for pcap_file in top_files:
            file_path = os.path.join(subfolder, pcap_file)
            if os.path.getsize(file_path) > 70 * 1024:  # 大于70KB
                destination_path = os.path.join(destination_folder, f"{os.path.basename(subfolder)}_{pcap_file}")
                shutil.copy(file_path, destination_path)
                print(f"已拷贝文件: {pcap_file} ({os.path.getsize(file_path) / 1024:.2f} KB)")


if __name__ == '__main__':
    # 用法示例
    source_folder = r"D:\数据集\trojan\YouTube+trojan"
    destination_folder = r"D:\数据集\trojan-clean\YouTube+trojan"
    copy_top_pcap(source_folder, destination_folder)
