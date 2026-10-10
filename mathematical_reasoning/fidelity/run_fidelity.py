#!/usr/bin/env python3
"""Measure GSM8K prefill representation and KV fidelity."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from prepare_plot_data import prepare_plot_data
from fig23.server_resources import resolve_model_reference

FIDELITY_DIR = Path(__file__).resolve().parent
# Source root containing FedAttn/ or its Wrapper.py and evaluate_gsm8k/ files.
SOURCE_REPO = FIDELITY_DIR.parent
RESULTS_ROOT = FIDELITY_DIR / "results"
PLOT_INPUTS_ROOT = FIDELITY_DIR / "plot_inputs"

RESULTS_ROOT = FIDELITY_DIR / "results_demo" # Measurement outputs for the demo.
PLOT_INPUTS_ROOT = FIDELITY_DIR / "plot_inputs_demo" # Plotting summaries for the demo.

# Select exactly one experiment and one job for each invocation.
EXPERIMENT = "fig23"          # "fig23", "trajectory", or "kv"
FIG23_JOB = "fp32"            # "fp32" or "7b_bf16"
TRAJECTORY_H = 2              # Synchronization interval for the block-trajectory experiment.
KV_JOB = "fp32"               # "fp32" or "7b_bf16"

# Execution controls for preparation, measurement, and export.
args = SimpleNamespace(prepare_only=False, export_only=False, resume=False)

FIG23_SETTINGS = {'fp32': {'source_repo': str(SOURCE_REPO),
          'dataset_source': 'hf_cache',
          'data_jsonl': None,
          'output_root': str(RESULTS_ROOT / 'fidelity_results_fig23_full_fp32'),
          'models': {'Qwen2.5-0.5B': 'Qwen/Qwen2.5-0.5B',
                    #  'Qwen2.5-1.5B': 'Qwen/Qwen2.5-1.5B',
                    #  'Qwen2.5-3B': 'Qwen/Qwen2.5-3B'
                     },
          'figures': [2, 3],
          'splits': ['even', 
                    #  'smart', 
                    #  'even_question_last', 
                    #  'smart_question_last'
                     ],
          'sample_selection': 'stride',
          'sample_start': 0,
          'sample_step': 1,
        #   'sample_count': None,
          'sample_count': 2,  # Select the first two test samples.
          'device': 'cuda:0',
          'dtype': 'float32',
          'seed': 0,
          'results_root': None,
          'plot_grid_json': None,
          'fig2_h_values': {'Qwen2.5-0.5B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                             20, 21, 22, 23, 24],
                            'Qwen2.5-1.5B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                             20, 21, 22, 23, 24, 25, 26, 27, 28],
                            'Qwen2.5-3B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
                            'Qwen2.5-7B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                           21, 22, 23, 24, 25, 26, 27, 28]},
          'fig3_n_values': [2, 3, 4, 5, 6, 7, 8],
          'sample_stop': None,
          'sample_plan_json': None},
 '7b_bf16': {'source_repo': str(SOURCE_REPO),
             'dataset_source': 'hf_cache',
             'data_jsonl': None,
             'output_root': str(RESULTS_ROOT / 'fidelity_results_fig23_full_7b_bf16'),
             'models': {'Qwen2.5-7B': 'Qwen/Qwen2.5-7B'},
             'figures': [2, 3],
             'splits': ['even', 
                        # 'smart', 
                        # 'even_question_last', 
                        # 'smart_question_last'
                        ],
             'sample_selection': 'stride',
             'sample_start': 0,
             'sample_step': 1,
             'sample_count': None,
             'device': 'cuda:0',
             'dtype': 'bfloat16',
             'seed': 0,
             'results_root': None,
             'plot_grid_json': None,
             'fig2_h_values': {'Qwen2.5-0.5B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                19, 20, 21, 22, 23, 24],
                               'Qwen2.5-1.5B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                               'Qwen2.5-3B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                              36],
                               'Qwen2.5-7B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                              20, 21, 22, 23, 24, 25, 26, 27, 28]},
             'fig3_n_values': [2, 3, 4, 5, 6, 7, 8],
             'sample_stop': None,
             'sample_plan_json': None}}

TRAJECTORY_SETTINGS = {'source_repo': str(SOURCE_REPO),
 'dataset_source': 'hf_cache',
 'data_jsonl': None,
 'models': {'Qwen2.5-0.5B': 'Qwen/Qwen2.5-0.5B',
            # 'Qwen2.5-1.5B': 'Qwen/Qwen2.5-1.5B',
            # 'Qwen2.5-3B': 'Qwen/Qwen2.5-3B',
            # 'Qwen2.5-7B': 'Qwen/Qwen2.5-7B'
            },
 'output_root': str(RESULTS_ROOT / f'block_trajectory_results_h{TRAJECTORY_H}_full'),
 'H': TRAJECTORY_H,
 'num_clients': 4,
 'num_shots': 4,
 'splits': ['even', 
            # 'even_question_last', 
            # 'smart', 
            # 'smart_question_last'
            ],
 'sample_selection': 'stride',
 'sample_start': 0,
 'sample_step': 1,
 'sample_count': None,
 'sample_stop': None,
 'sample_plan_json': None,
 'dtype_by_model': {'Qwen2.5-0.5B': 'float32',
                    'Qwen2.5-1.5B': 'float32',
                    'Qwen2.5-3B': 'float32',
                    'Qwen2.5-7B': 'bfloat16'},
 'device': 'cuda:0',
 'seed': 0}

KV_SETTINGS = {'fp32': {'source_repo': str(SOURCE_REPO),
          'dataset_source': 'hf_cache',
          'data_jsonl': None,
          'output_root': str(RESULTS_ROOT / 'fidelity_results_fig4_kv_full_fp32'),
          'models': {'Qwen2.5-0.5B': 'Qwen/Qwen2.5-0.5B',
                    #  'Qwen2.5-1.5B': 'Qwen/Qwen2.5-1.5B',
                    #  'Qwen2.5-3B': 'Qwen/Qwen2.5-3B'
                     },
          'dtype_by_model': {'Qwen2.5-0.5B': 'float32', 'Qwen2.5-1.5B': 'float32', 'Qwen2.5-3B': 'float32'},
          'H_by_model': {'Qwen2.5-0.5B': 8, 'Qwen2.5-1.5B': 8, 'Qwen2.5-3B': 9, 'Qwen2.5-7B': 8},
          'splits': ['even', 
                     # 'even_question_last', 
                     # 'smart', 
                     # 'smart_question_last'
                     ],
          'sample_selection': 'stride',
          'sample_start': 0,
          'sample_step': 1,
          'sample_count': None,
          'device': 'cuda:0',
          'seed': 0,
          'ratio_comp_values': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
          'ratio_comm_values': [0.001, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
          'schemes': ['uni_fr', 'uni_bk', 'inc_gp', 'dec_gp'],
          'sample_stop': None},
  '7b_bf16': {'source_repo': str(SOURCE_REPO),
          'dataset_source': 'hf_cache',
          'data_jsonl': None,
          'output_root': str(RESULTS_ROOT / 'fidelity_results_fig4_kv_full_7b_bf16'),
          'models': {'Qwen2.5-7B': 'Qwen/Qwen2.5-7B'},
          'dtype_by_model': {'Qwen2.5-7B': 'bfloat16'},
          'H_by_model': {'Qwen2.5-0.5B': 8, 'Qwen2.5-1.5B': 8, 'Qwen2.5-3B': 9, 'Qwen2.5-7B': 8},
          'splits': ['even', 
                     # 'even_question_last', 
                     # 'smart', 
                     # 'smart_question_last'
                     ],
          'sample_selection': 'stride',
          'sample_start': 0,
          'sample_step': 1,
          'sample_count': None,
          'device': 'cuda:0',
          'seed': 0,
          'ratio_comp_values': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
          'ratio_comm_values': [0.001, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
          'schemes': ['uni_fr', 'uni_bk', 'inc_gp', 'dec_gp'],
          'sample_stop': None}}

def select_experiment():
    if EXPERIMENT == "fig23":
        settings = deepcopy(FIG23_SETTINGS[FIG23_JOB])
        runner = ("run_fig23_deviation.py" if FIG23_JOB == "fp32"
                  else "run_fig23_deviation_7b_maskfix.py")
        job = FIG23_JOB
    elif EXPERIMENT == "trajectory":
        settings = deepcopy(TRAJECTORY_SETTINGS)
        runner, job = "run_block_trajectory.py", settings["H"]
    elif EXPERIMENT == "kv":
        settings = deepcopy(KV_SETTINGS[KV_JOB])
        runner, job = "run_fig4_fidelity.py", KV_JOB
    else:
        raise ValueError("EXPERIMENT must be fig23, trajectory, or kv")
    return settings, FIDELITY_DIR / EXPERIMENT / runner, job


def resolve_settings_paths(settings):
    # Resolve relative filesystem paths against the entry directory.
    for field in ("source_repo", "output_root", "data_jsonl", "sample_plan_json",
                  "results_root", "plot_grid_json", "dataset_path"):
        value = settings.get(field)
        if value:
            path = Path(value).expanduser()
            settings[field] = str((path if path.is_absolute() else FIDELITY_DIR / path).resolve())
    settings["models"] = {
        model: resolve_model_reference(value, FIDELITY_DIR)
        for model, value in settings["models"].items()
    }
    return settings


def export_existing(root):
    folder = FIDELITY_DIR / EXPERIMENT
    if EXPERIMENT == "fig23":
        command = [str(folder / "export_representation_deviation.py"),
                   "--results-root", str(root), "--plan", str(root / "batch_plan.json"),
                   "--output-dir", str(root / "curves")]
    elif EXPERIMENT == "trajectory":
        command = [str(folder / "trajectory_io.py"), str(root)]
    else:
        command = [str(folder / "export_fig4_fidelity.py"), "--output-root", str(root)]
    subprocess.run([sys.executable, *command], check=True, cwd=str(folder))


def main():
    if args.prepare_only and args.export_only:
        raise ValueError("prepare_only and export_only cannot both be True")
    settings, runner, job = select_experiment()
    settings = resolve_settings_paths(settings)
    root = Path(settings["output_root"])
    if args.export_only:
        # Export saved plans and measurements without updating run settings.
        export_existing(root)
    else:
        source = Path(settings["source_repo"])
        fed = source / "FedAttn" if (source / "FedAttn").is_dir() else source
        for relative in ("Wrapper.py", "utils.py", "evaluate_gsm8k/main_evaluate_gsm8k.py"):
            if not (fed / relative).is_file():
                raise FileNotFoundError(f"Original source file is missing: {fed / relative}")
        root.mkdir(parents=True, exist_ok=True)
        settings_path = root / "fidelity_settings.json"
        if settings_path.exists():
            previous = json.loads(settings_path.read_text(encoding="utf-8"))
            if previous != settings:
                raise ValueError(f"Changed settings at {settings_path}; use a different output_root")
        else:
            settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        command = [sys.executable, str(runner), "--settings", str(settings_path)]
        if args.prepare_only:
            command.append("--prepare-only")
        if args.resume:
            command.append("--resume")
        print(f"Experiment: {EXPERIMENT}; job: {job}; output: {root}", flush=True)
        subprocess.run(command, check=True, cwd=str(runner.parent))
        if args.prepare_only:
            return
    for path in prepare_plot_data(EXPERIMENT, {job: root}, PLOT_INPUTS_ROOT):
        print(f"Plot input: {path}", flush=True)


if __name__ == "__main__":
    main()
