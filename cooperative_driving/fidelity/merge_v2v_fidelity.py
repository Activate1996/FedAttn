#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path
from export_v2v_fidelity import merge_results

HERE = Path(__file__).resolve().parent
TASK = 'q1'
GROUP = 'A'
NUM_SHARDS = 3
RUN_ROOT = HERE / 'results/v2v_fidelity' / TASK / GROUP
INPUT_ROOTS = [RUN_ROOT / f'shard{i}' for i in range(NUM_SHARDS)]
OUTPUT_ROOT = RUN_ROOT / 'merged'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', '--input-roots', dest='inputs', nargs='+', type=Path, default=INPUT_ROOTS)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    summary = merge_results(args.inputs, args.output_root, make_bundle=False)
    from prepare_plot_data import prepare
    prepare(summary, args.output_root / 'plot_inputs')


if __name__ == '__main__':
    main()
