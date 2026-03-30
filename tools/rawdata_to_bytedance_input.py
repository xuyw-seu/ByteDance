#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert raw traffic data into ByteDance model inputs.

Supported raw input files:
1) Three-column text files (timestamp, signed length, byte sequence)
2) PCAP/PCAPNG files (converted via tools/feature_extraction/traffic_feature_extractor.py)

Output:
- NPZ file containing:
  - len_seq:  [N, max_packet_num] int64
  - byte_seq: [N, max_packet_num, byte_dim] int64
  - labels:   [N] int64
  - files:    [N] object (relative paths)
- label_map.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tools.feature_extraction.traffic_feature_extractor import (
    extract_3_colum_feature_from_pcap,
)


def _parse_byte_token(token: str) -> int:
    token = token.strip()
    if not token:
        return 0
    if token.lower().startswith("0x"):
        return int(token, 16) & 0xFF
    if any(c in "abcdefABCDEF" for c in token):
        return int(token, 16) & 0xFF
    value = int(token)
    if value < 0:
        value = 0
    if value > 255:
        value = 255
    return value


def _signed_len_to_token(signed_len: int) -> int:
    # Keep 0 as padding id. Valid packet-length tokens are [1, 3999].
    value = signed_len + 2001
    if value < 1:
        return 1
    if value > 3999:
        return 3999
    return value


def _parse_three_column_txt(
    file_path: Path,
    max_packet_num: int,
    byte_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    len_seq = np.zeros((max_packet_num,), dtype=np.int64)
    byte_seq = np.zeros((max_packet_num, byte_dim), dtype=np.int64)

    packet_idx = 0
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                signed_len = int(float(parts[1]))
            except ValueError:
                continue

            len_seq[packet_idx] = _signed_len_to_token(signed_len)

            payload_tokens = parts[2:]
            payload_vals = []
            for token in payload_tokens[:byte_dim]:
                try:
                    payload_vals.append(_parse_byte_token(token))
                except ValueError:
                    payload_vals.append(0)
            if payload_vals:
                byte_seq[packet_idx, : len(payload_vals)] = np.asarray(payload_vals, dtype=np.int64)

            packet_idx += 1
            if packet_idx >= max_packet_num:
                break

    return len_seq, byte_seq


def _collect_raw_files(root: Path) -> List[Path]:
    supported = {".txt", ".pcap", ".pcapng"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in supported]
    files.sort()
    return files


def _infer_label(rel_path: Path) -> str:
    if len(rel_path.parts) >= 2:
        return rel_path.parts[0]
    return "default"


def convert_rawdata_to_model_input(
    rawdata_dir: Path,
    output_npz: Path,
    max_packet_num: int = 20,
    byte_dim: int = 80,
    min_packets: int = 1,
) -> Dict[str, int]:
    all_files = _collect_raw_files(rawdata_dir)
    if not all_files:
        raise FileNotFoundError(f"No supported files found under: {rawdata_dir}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_txt_dir = output_npz.parent / "_tmp_three_column_txt"
    tmp_txt_dir.mkdir(parents=True, exist_ok=True)

    label_to_id: Dict[str, int] = {}
    len_list: List[np.ndarray] = []
    byte_list: List[np.ndarray] = []
    label_list: List[int] = []
    file_list: List[str] = []

    for idx, raw_file in enumerate(all_files, start=1):
        rel_path = raw_file.relative_to(rawdata_dir)
        label_name = _infer_label(rel_path)
        if label_name not in label_to_id:
            label_to_id[label_name] = len(label_to_id)

        parse_file = raw_file
        if raw_file.suffix.lower() in {".pcap", ".pcapng"}:
            txt_name = str(rel_path).replace("\\", "_").replace("/", "_") + ".txt"
            parse_file = tmp_txt_dir / txt_name
            data_tuple = f"{raw_file}%%{parse_file}"
            extract_3_colum_feature_from_pcap(data_tuple)
            if not parse_file.exists():
                print(f"[Skip] pcap conversion failed: {raw_file}")
                continue

        len_seq, byte_seq = _parse_three_column_txt(
            parse_file,
            max_packet_num=max_packet_num,
            byte_dim=byte_dim,
        )
        valid_packets = int((len_seq != 0).sum())
        if valid_packets < min_packets:
            print(f"[Skip] too few packets ({valid_packets}): {raw_file}")
            continue

        len_list.append(len_seq)
        byte_list.append(byte_seq)
        label_list.append(label_to_id[label_name])
        file_list.append(str(rel_path))

        if idx % 200 == 0:
            print(f"Processed {idx}/{len(all_files)} files...")

    if not len_list:
        raise RuntimeError("No valid samples produced. Check your raw data format.")

    len_arr = np.stack(len_list, axis=0)
    byte_arr = np.stack(byte_list, axis=0)
    label_arr = np.asarray(label_list, dtype=np.int64)
    file_arr = np.asarray(file_list, dtype=object)

    np.savez_compressed(
        output_npz,
        len_seq=len_arr,
        byte_seq=byte_arr,
        labels=label_arr,
        files=file_arr,
    )

    label_map_path = output_npz.with_suffix(".label_map.json")
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Samples: {len(label_arr)}")
    print(f"len_seq shape:  {len_arr.shape}")
    print(f"byte_seq shape: {byte_arr.shape}")
    print(f"labels shape:   {label_arr.shape}")
    print(f"NPZ: {output_npz}")
    print(f"Label map: {label_map_path}")

    return label_to_id


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert rawdata directory to ByteDance model input npz."
    )
    parser.add_argument(
        "--rawdata-dir",
        type=Path,
        required=True,
        help="Root directory of raw data (.txt/.pcap/.pcapng).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--max-packet-num",
        type=int,
        default=20,
        help="Number of packets kept for each sample.",
    )
    parser.add_argument(
        "--byte-dim",
        type=int,
        default=80,
        help="Number of bytes kept per packet.",
    )
    parser.add_argument(
        "--min-packets",
        type=int,
        default=1,
        help="Minimum non-padding packets required for a valid sample.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    convert_rawdata_to_model_input(
        rawdata_dir=args.rawdata_dir,
        output_npz=args.output,
        max_packet_num=args.max_packet_num,
        byte_dim=args.byte_dim,
        min_packets=args.min_packets,
    )


if __name__ == "__main__":
    main()
