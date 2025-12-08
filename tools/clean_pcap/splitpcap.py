
import os
import subprocess

def split_pcap_files(folder_path):
    # 获取文件夹下的所有 pcap 文件
    pcap_files = [file for file in os.listdir(folder_path) if file.endswith('.pcap')]

    # 获取 Splitpcap.exe 的路径
    splitpcap_path = os.path.join(folder_path, 'SplitCap.exe')

    index = 0

    # 遍历每个 pcap 文件并执行 Splitpcap.exe
    for pcap_file in pcap_files:
        print("No {}".format(index))
        index += 1
        pcap_file_path = os.path.join(folder_path, pcap_file)
        s_dir = pcap_file_path.split('.')[0]

        # 构建 cmd 命令
        command = f'"{splitpcap_path}" -r "{pcap_file_path}" -s session -o "{s_dir}"'

        try:
            # 使用 subprocess.run 执行命令
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"执行命令时出错：{e}")


if __name__ == '__main__':

    split_pcap_files(r"D:\数据集\trojan\YouTube+trojan")