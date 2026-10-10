"""Re-export the saved Q1/A measurements and compare with their reference outputs."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / 'demo/q1_fidelity_A/input'
EXPECTED_DIR = ROOT / 'demo/q1_fidelity_A/expected_outputs'
OUTPUT_DIR = ROOT / 'results/demo/q1_fidelity_A'
sys.path.insert(0, str(ROOT / 'fidelity'))

from export_v2v_fidelity import export_results
from fidelity_locks import result_lock
from prepare_plot_data import prepare


def main():
    with result_lock(OUTPUT_DIR):
        for source in INPUT_DIR.iterdir():
            if source.is_file():
                shutil.copy2(source, OUTPUT_DIR / source.name)
        summary = export_results(OUTPUT_DIR, make_bundle=False)
        plot_summary = prepare(summary, OUTPUT_DIR / 'plot_inputs')
        for name in ('v2v_fidelity_summary.json', 'v2v_fidelity_points.csv',
                     'v2v_fidelity_samples.csv', 'plot_inputs/v2v_fidelity_plot_points.csv'):
            actual, expected = OUTPUT_DIR / name, EXPECTED_DIR / name
            if name.endswith('.json'):
                equal = json.loads(actual.read_text()) == json.loads(expected.read_text())
            else:
                equal = actual.read_bytes() == expected.read_bytes()
            if not equal:
                raise AssertionError(f'Demo output differs from reference: {name}')
        actual = json.loads(plot_summary.read_text())
        expected = json.loads((EXPECTED_DIR / 'plot_inputs/v2v_fidelity_summary.json').read_text())
        # Paths and the preparation-code fingerprint identify the current export.
        for field in ('points', 'baselines', 'point_counts'):
            if actual[field] != expected[field]:
                raise AssertionError(f'Plot summary data differ: {field}')
    print('PASS: 7 samples, 455 prefill records, 64 complete Q1/A KV points.')
    print(f'Generated outputs: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
