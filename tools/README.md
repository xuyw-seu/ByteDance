# Tools 目录说明

## 1. 目录简介

该目录包含了用于网络流量处理、分析和可视化的各种工具脚本，主要用于网络流量特征提取、文本数据处理、时间序列分析、数据可视化和数据转换等任务。

## 2. 目录结构

```
tools/
├── clean_pcap/               # PCAP文件清理和处理工具
├── data_conversion/          # 数据转换工具
├── data_visualization/       # 数据可视化工具
├── feature_extraction/       # 网络流量特征提取工具
├── process_json/             # JSON数据处理工具
├── time_sequence_analysis/   # 时间序列分析工具
└── use_tools/                # 通用工具和测试工具
```

## 3. 子目录功能说明

### 3.1 clean_pcap
用于PCAP文件的清理、过滤和处理

| 文件名 | 功能描述 |
|--------|----------|
| clean_pcap.py | 基础PCAP文件清理工具，用于清理和过滤PCAP文件 |
| clean_pcap_enhanced.py | 增强版PCAP文件清理工具，提供更复杂的过滤和处理功能 |
| get_max_100_pcap.py | 获取最大的100个PCAP文件 |
| splitpcap.py | PCAP文件分割工具，用于将大PCAP文件分割成多个小文件 |

### 3.2 data_conversion
用于不同格式数据之间的转换

| 文件名 | 功能描述 |
|--------|----------|
| hdf5_file_processor.py | HDF5文件处理器，用于处理HDF5格式文件，包括文件结构查看和数据集转换为文本格式 |
| turn_tsv_to_txt_formate.py | 将TSV格式转换为TXT格式 |

### 3.3 data_visualization
用于生成各种图表来可视化网络流量数据

| 文件名 | 功能描述 |
|--------|----------|
| comparative_line_graph_plotter.py | 对比折线图绘制工具，用于绘制两个文本文件数据的对比折线图，特别适用于学术论文图表 |
| traffic_confusion_matrix_generator.py | 流量混淆矩阵生成器，用于生成和显示网络流量数据的混淆矩阵 |
| traffic_line_graph_plotter.py | 网络流量折线图绘制工具，用于绘制网络流量数据的折线图，区分请求和响应数据 |
| traffic_request_response_ratio.py | 网络流量请求响应比例计算器，用于计算网络流量数据中请求和响应的字节比例 |

### 3.4 feature_extraction
用于从PCAP文件中提取网络流量特征，是整个流量处理流程的核心

| 文件名 | 功能描述 |
|--------|----------|
| traffic_feature_extractor.py | 核心特征提取库，提供基础的网络流量特征提取功能 |
| traffic_text_processor.py | 文本数据处理器，支持并行处理、特征提取和数据转换 |
| turn_pcap_to_2_columns.py | 将PCAP文件转换为两列格式（时间戳和字节数） |
| turn_pcap_to_pean_model.py | 将PCAP文件转换为PEAN模型所需的格式 |

### 3.5 process_json
用于处理JSON格式的网络流量数据

| 文件名 | 功能描述 |
|--------|----------|
| process_json_combined_labels.py | 用于合并JSON数据中的标签信息 |
| process_json_feature_extractor.py | 用于从JSON数据中提取网络流量特征 |

### 3.6 time_sequence_analysis
专注于网络流量时间序列数据的分析和处理

| 文件名 | 功能描述 |
|--------|----------|
| time_sequence_analyzer.py | 时间序列分析器，提供基础的时间序列处理功能，包括时间突变检测和子图生成 |
| traffic_time_sequence_burst_detector.py | 突发检测增强版网络流量时间序列处理器，专注于时间突变分析 |
| traffic_time_sequence_enhanced.py | 增强版网络流量时间序列处理器，支持更复杂的数据格式和处理逻辑 |
| traffic_time_sequence_grouped.py | 分组增强版网络流量时间序列处理器，支持带分组信息的复杂数据格式处理 |
| traffic_time_sequence_processor.py | 网络流量时间序列处理器，用于处理网络流量的时间序列数据 |

### 3.7 use_tools
通用工具和测试工具

| 文件名 | 功能描述 |
|--------|----------|
| copy_top_k_pcaps.py | 复制前K个PCAP文件的工具 |
| file_line_count_analyzer.py | 文件行数统计分析工具，用于统计文本文件的行数分布 |
| test.py | 通用测试文件 |
| test_local_attention.py | 测试本地注意力机制的文件 |
| traffic_feature_ratio_calculator.py | 流量特征比例计算器，用于计算网络流量数据中请求和响应的比例特征 |

## 4. 文件命名规则

1. 前缀表示文件类型或功能：`traffic_`（网络流量相关）、`time_`（时间序列相关）等
2. 中间部分描述具体功能：`feature_extractor`、`line_graph_plotter`等
3. 后缀表示文件类型：`.py`

这种命名方式使得文件功能一目了然，便于后续维护和使用。

## 5. 依赖说明

主要依赖库包括：
- matplotlib：用于数据可视化
- numpy：用于数值计算
- h5py：用于处理HDF5文件
- scapy：用于处理PCAP文件

## 6. 使用示例

### 6.1 PCAP文件清理示例
```python
# 使用clean_pcap_enhanced.py清理PCAP文件
from clean_pcap.clean_pcap_enhanced import clean_pcap

input_pcap = "sample.pcap"
output_pcap = "cleaned_sample.pcap"
clean_pcap(input_pcap, output_pcap)
```

### 6.2 特征提取示例
```python
# 使用traffic_feature_extractor.py提取PCAP特征
from feature_extraction.traffic_feature_extractor import extract_features

pcap_file = "sample.pcap"
features = extract_features(pcap_file)
```

### 6.3 数据可视化示例
```python
# 使用traffic_line_graph_plotter.py绘制流量折线图
from data_visualization.traffic_line_graph_plotter import plot_positive_negative_lines

input_file = "traffic_data.txt"
save_path = "traffic_plot.png"
plot_positive_negative_lines(input_file, save_path)
```

### 6.4 时间序列分析示例
```python
# 使用time_sequence_analyzer.py检测时间突变
from time_sequence_analysis.time_sequence_analyzer import find_timestamp_transitions

timestamp_list = [0, 1, 2, 5, 8, 10, 12, 15, 18, 20, 1000, 2000]
threshold = 10
transitions, sequence_lengths = find_timestamp_transitions(timestamp_list, threshold)
print("时间突变位置:", transitions)
print("突变序列长度:", sequence_lengths)
```

## 7. 注意事项

1. 部分脚本可能包含硬编码的文件路径，使用前需要根据实际情况修改
2. 某些脚本可能依赖于特定的数据格式，使用前请确保输入数据格式正确
3. 对于大型数据集，建议使用并行处理功能以提高效率
4. 可视化脚本默认将图表保存为PNG或PDF格式，可根据需要修改保存格式
