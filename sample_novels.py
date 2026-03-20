#!/usr/bin/env python3
"""从子文件夹中按比例随机抽取JSON文件，组成sample.jsonl。

用法:
    python3 sample_novels.py -n 100                    # 按比例抽取100条
    python3 sample_novels.py -n 10                     # 随机抽取10条（<20不按比例）
    python3 sample_novels.py -n 500 -o my_sample.jsonl # 指定输出文件
    python3 sample_novels.py -n 100 --max-size 512     # 只采样512KB以内的文件
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def scan_folders(data_dir: Path, max_size_kb: Optional[float] = None) -> dict[str, list[Path]]:
    """扫描子文件夹，返回 {文件夹名: [json文件路径列表]}。

    Args:
        data_dir: 数据目录路径
        max_size_kb: 文件大小上限（KB），超过此大小的文件将被排除；None表示不限制
    """
    max_size_bytes = max_size_kb * 1024 if max_size_kb is not None else None
    folder_files: dict[str, list[Path]] = {}
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        json_files = sorted(entry.glob("*.json"))
        if max_size_bytes is not None:
            json_files = [f for f in json_files if f.stat().st_size <= max_size_bytes]
        if json_files:
            folder_files[entry.name] = json_files
    return folder_files


def sample_proportional(folder_files: dict[str, list[Path]], n: int) -> list[tuple[str, Path]]:
    """按子文件夹文件数量比例抽取，返回 [(文件夹名, 文件路径)]。"""
    total = sum(len(files) for files in folder_files.values())

    # 按比例计算每个文件夹的抽取数（向下取整）
    quotas: dict[str, int] = {}
    for folder, files in folder_files.items():
        quotas[folder] = math.floor(n * len(files) / total)

    # 剩余名额按小数部分从大到小分配
    remainder = n - sum(quotas.values())
    if remainder > 0:
        fractional = []
        for folder, files in folder_files.items():
            frac = (n * len(files) / total) - quotas[folder]
            fractional.append((frac, folder))
        fractional.sort(reverse=True)
        for _, folder in fractional[:remainder]:
            quotas[folder] += 1

    # 从每个文件夹中随机抽取
    selected: list[tuple[str, Path]] = []
    for folder, files in folder_files.items():
        k = min(quotas[folder], len(files))
        sampled = random.sample(files, k)
        selected.extend((folder, f) for f in sampled)

    random.shuffle(selected)
    return selected


def sample_random(folder_files: dict[str, list[Path]], n: int) -> list[tuple[str, Path]]:
    """纯随机抽取（不按比例），返回 [(文件夹名, 文件路径)]。"""
    all_files: list[tuple[str, Path]] = []
    for folder, files in folder_files.items():
        all_files.extend((folder, f) for f in files)
    return random.sample(all_files, min(n, len(all_files)))


def main():
    parser = argparse.ArgumentParser(description="从子文件夹中随机抽取JSON文件，组成sample.jsonl")
    parser.add_argument("-n", "--num", type=int, required=True, help="抽取数量")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出文件路径（默认 sample_n{num}_{timestamp}.jsonl）")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="数据目录路径")
    parser.add_argument("--max-size", type=float, default=None,
                        help="文件大小上限（单位KB），超过此大小的文件将被排除；未指定则不限制")
    args = parser.parse_args()

    # 如果未指定输出文件，则自动生成带 num 和时间戳的文件名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"sample_n{args.num}_{timestamp}.jsonl"

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"错误: 数据目录不存在: {data_dir}", file=sys.stderr)
        sys.exit(1)

    folder_files = scan_folders(data_dir, max_size_kb=args.max_size)
    if not folder_files:
        print(f"错误: 数据目录下没有找到包含JSON文件的子文件夹", file=sys.stderr)
        sys.exit(1)

    total = sum(len(files) for files in folder_files.values())
    if args.max_size is not None:
        print(f"文件大小上限: {args.max_size}KB")
    print(f"扫描到 {len(folder_files)} 个子文件夹，共 {total} 个JSON文件")

    n = args.num
    if n > total:
        print(f"警告: 请求数量 {n} 超过总文件数 {total}，将抽取全部文件", file=sys.stderr)
        n = total

    if n < 20:
        print(f"抽取数量 < 20，使用纯随机抽取（不按比例）")
        selected = sample_random(folder_files, n)
    else:
        print(f"按子文件夹比例抽取 {n} 条")
        selected = sample_proportional(folder_files, n)

    # 写入 JSONL
    output_path = Path(args.output)
    written = 0
    errors = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for folder, filepath in selected:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["source_folder"] = folder
                out.write(json.dumps(data, ensure_ascii=False) + "\n")
                written += 1
            except (json.JSONDecodeError, OSError) as e:
                print(f"警告: 读取失败 {filepath}: {e}", file=sys.stderr)
                errors += 1

    print(f"完成! 写入 {written} 条到 {output_path}")
    if errors:
        print(f"（{errors} 个文件读取失败）")

    # 打印各文件夹抽取统计
    from collections import Counter
    counts = Counter(folder for folder, _ in selected)
    print("\n各文件夹抽取统计:")
    for folder in sorted(counts, key=counts.get, reverse=True):
        orig = len(folder_files[folder])
        print(f"  {folder}: {counts[folder]}/{orig} ({counts[folder]/orig*100:.1f}%)")


if __name__ == "__main__":
    main()
