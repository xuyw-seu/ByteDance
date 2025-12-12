# Baseline Methods

This directory contains implementations of several baseline methods for network traffic classification and analysis. The following table lists all baseline methods used in our study, including their sources and availability.

## Baseline Methods Table

| Method | Source |
|--------|--------|
| FS-NET<sup>[1]</sup> | https://github.com/WSPTTH/FS-Net<sup>[1]</sup> |
| RF<sup>[2]</sup> | https://github.com/robust-fingerprinting/RF<sup>[2]</sup> |
| ET-BERT<sup>[3]</sup> | https://github.com/linwhitehat/ET-BERT<sup>[3]</sup> |
| PEAN<sup>[4]</sup> | https://github.com/Lin-Dada/PEAN<sup>[4]</sup> |
| PETNet<sup>[5]</sup> | https://github.com/CN-PETNet/PETNet<sup>[5]</sup> |
| GLADS<sup>[6]</sup> | See baseline folder for details<sup>[6]</sup> |
| MFFusion<sup>[7]</sup> | See baseline folder for details<sup>[7]</sup> |
| DM-HNN<sup>[8]</sup> | See baseline folder for details<sup>[8]</sup> |
| TSCRNN<sup>[9]</sup> | See baseline folder for details<sup>[9]</sup> |

## Code Availability

- Implementations for **GLADS**, **MFF**, **DM-HNN**, and **TSCRNN** are provided in this directory.
- For other baseline methods, please refer to their original repositories via the links provided in the table above.

## Citation

If you use our implementations of GLADS, MFF, DM-HNN, or TSCRNN in your research, please cite our paper:

```
@article{XU2025111843,
    title = {ByteDance: Let bytes perform brilliantly in multi-view encrypted traffic classification},
    author = {Yuwei Xu and Zhiyuan Liang and Xiaotian Fang and Kehui Song and Meng Wang and Qiao Xiang and Guang Cheng},
    journal = {Computer Networks},
    pages = {111843},
    year = {2025},
    issn = {1389-1286},
    doi = {https://doi.org/10.1016/j.comnet.2025.111843},
    url = {https://www.sciencedirect.com/science/article/pii/S1389128625002071},
}
```

## Notes

- All implementations follow a unified interface for easy comparison.
- Each method's code is organized in its respective subdirectory.
- Please refer to the individual README files in each subdirectory for usage instructions.