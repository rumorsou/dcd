import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def sample_edges(input_file: str, n: int, output_file: Optional[str] = None, seed: Optional[int] = None):
    if n < 0:
        raise ValueError("n 必须是非负整数。")

    array = np.loadtxt(input_file, dtype=np.int32)
    array = np.atleast_2d(array)

    if array.shape[1] < 2:
        raise ValueError("输入文件至少需要两列，表示一条边的两个端点。")

    edges = array[:, :2]
    edge_num = edges.shape[0]
    sample_size = min(n, edge_num)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(edge_num, size=sample_size, replace=False)
    sampled_edges = edges[sampled_indices]

    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_name(f"{input_path.stem}_{sample_size}{input_path.suffix}"))

    np.savetxt(output_file, sampled_edges, fmt="%d")
    return output_file, sample_size


def main():
    parser = argparse.ArgumentParser(description="从图文件中随机采样 n 条边。")
    parser.add_argument("input_file", help="输入图文件路径")
    parser.add_argument("n", type=int, help="随机采样的边数")
    parser.add_argument("-o", "--output", dest="output_file", help="输出文件路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")
    args = parser.parse_args()

    output_file, sample_size = sample_edges(
        args.input_file,
        args.n,
        output_file=args.output_file,
        seed=args.seed,
    )
    print(f"已输出 {sample_size} 条边到: {output_file}")


if __name__ == "__main__":
    main()
