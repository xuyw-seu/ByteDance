# Tools Directory Documentation

## 1. Directory Introduction

This directory contains various tool scripts for network traffic processing, analysis, and visualization, mainly used for network traffic feature extraction, text data processing, time series analysis, data visualization, and data conversion tasks.

## 2. Directory Structure

```
tools/
├── clean_pcap/               # PCAP file cleaning and processing tools
├── data_conversion/          # Data conversion tools
├── data_visualization/       # Data visualization tools
├── feature_extraction/       # Network traffic feature extraction tools
├── process_json/             # JSON data processing tools
├── time_sequence_analysis/   # Time sequence analysis tools
└── use_tools/                # General tools and test tools
```

## 3. Subdirectory Function Descriptions

### 3.1 clean_pcap
Used for PCAP file cleaning, filtering, and processing

| File Name | Function Description |
|--------|----------|
| clean_pcap.py | Basic PCAP file cleaning tool for cleaning and filtering PCAP files |
| clean_pcap_enhanced.py | Enhanced PCAP file cleaning tool providing more complex filtering and processing capabilities |
| get_max_100_pcap.py | Tool to get the largest 100 PCAP files |
| splitpcap.py | PCAP file splitting tool for dividing large PCAP files into multiple small files |

### 3.2 data_conversion
Used for conversion between different data formats

| File Name | Function Description |
|--------|----------|
| hdf5_file_processor.py | HDF5 file processor for handling HDF5 format files, including file structure viewing and dataset conversion to text format |
| turn_tsv_to_txt_formate.py | Converts TSV format to TXT format |

### 3.3 data_visualization
Used for generating various charts to visualize network traffic data

| File Name | Function Description |
|--------|----------|
| comparative_line_graph_plotter.py | Comparative line graph plotting tool for drawing comparative line graphs of data from two text files, especially suitable for academic paper charts |
| traffic_confusion_matrix_generator.py | Traffic confusion matrix generator for generating and displaying confusion matrices of network traffic data |
| traffic_line_graph_plotter.py | Network traffic line graph plotting tool for drawing line graphs of network traffic data, distinguishing between request and response data |
| traffic_request_response_ratio.py | Network traffic request-response ratio calculator for calculating the byte ratio of requests and responses in network traffic data |

### 3.4 feature_extraction
Used for extracting network traffic features from PCAP files, which is the core of the entire traffic processing workflow

| File Name | Function Description |
|--------|----------|
| traffic_feature_extractor.py | Core feature extraction library providing basic network traffic feature extraction functionality |
| traffic_text_processor.py | Text data processor supporting parallel processing, feature extraction, and data conversion |
| turn_pcap_to_2_columns.py | Converts PCAP files to two-column format (timestamp and byte count) |
| turn_pcap_to_pean_model.py | Converts PCAP files to the format required by the PEAN model |

### 3.5 process_json
Used for processing JSON format network traffic data

| File Name | Function Description |
|--------|----------|
| process_json_combined_labels.py | Used for merging label information in JSON data |
| process_json_feature_extractor.py | Used for extracting network traffic features from JSON data |

### 3.6 time_sequence_analysis
Focused on the analysis and processing of network traffic time sequence data

| File Name | Function Description |
|--------|----------|
| time_sequence_analyzer.py | Time sequence analyzer providing basic time sequence processing functionality, including time mutation detection and subgraph generation |
| traffic_time_sequence_burst_detector.py | Burst detection enhanced network traffic time sequence processor, focusing on time mutation analysis |
| traffic_time_sequence_enhanced.py | Enhanced network traffic time sequence processor supporting more complex data formats and processing logic |
| traffic_time_sequence_grouped.py | Grouped enhanced network traffic time sequence processor supporting complex data format processing with grouping information |
| traffic_time_sequence_processor.py | Network traffic time sequence processor for processing time sequence data of network traffic |

### 3.7 use_tools
General tools and test tools

| File Name | Function Description |
|--------|----------|
| copy_top_k_pcaps.py | Tool for copying the top K PCAP files |
| file_line_count_analyzer.py | File line count statistical analysis tool for counting the line distribution of text files |
| test.py | General test file |
| test_local_attention.py | File for testing local attention mechanism |
| traffic_feature_ratio_calculator.py | Traffic feature ratio calculator for calculating the ratio features of requests and responses in network traffic data |

## 4. File Naming Conventions

1. Prefix indicates file type or function: `traffic_` (network traffic related), `time_` (time sequence related), etc.
2. Middle part describes specific function: `feature_extractor`, `line_graph_plotter`, etc.
3. Suffix indicates file type: `.py`

This naming convention makes file functions clear at a glance, facilitating subsequent maintenance and use.

## 5. Dependency Description

Main dependency libraries include:
- matplotlib: Used for data visualization
- numpy: Used for numerical calculations
- h5py: Used for processing HDF5 files
- scapy: Used for processing PCAP files

## 6. Usage Examples

### 6.1 PCAP File Cleaning Example
```python
# Use clean_pcap_enhanced.py to clean PCAP files
from clean_pcap.clean_pcap_enhanced import clean_pcap

input_pcap = "sample.pcap"
output_pcap = "cleaned_sample.pcap"
clean_pcap(input_pcap, output_pcap)
```

### 6.2 Feature Extraction Example
```python
# Use traffic_feature_extractor.py to extract PCAP features
from feature_extraction.traffic_feature_extractor import extract_features

pcap_file = "sample.pcap"
features = extract_features(pcap_file)
```

### 6.3 Data Visualization Example
```python
# Use traffic_line_graph_plotter.py to draw traffic line graphs
from data_visualization.traffic_line_graph_plotter import plot_positive_negative_lines

input_file = "traffic_data.txt"
save_path = "traffic_plot.png"
plot_positive_negative_lines(input_file, save_path)
```

### 6.4 Time Sequence Analysis Example
```python
# Use time_sequence_analyzer.py to detect time mutations
from time_sequence_analysis.time_sequence_analyzer import find_timestamp_transitions

timestamp_list = [0, 1, 2, 5, 8, 10, 12, 15, 18, 20, 1000, 2000]
threshold = 10
transitions, sequence_lengths = find_timestamp_transitions(timestamp_list, threshold)
print("Time mutation positions:", transitions)
print("Mutation sequence lengths:", sequence_lengths)
```

## 7. Notes

1. Some scripts may contain hardcoded file paths, which need to be modified according to actual situations before use
2. Certain scripts may depend on specific data formats, please ensure the input data format is correct before use
3. For large datasets, it is recommended to use parallel processing functionality to improve efficiency
4. Visualization scripts default to saving charts in PNG or PDF format, which can be modified as needed