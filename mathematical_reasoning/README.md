# FedAttn: mathematical reasoning on GSM8K

Inference, answer evaluation, analytical system costs and fidelity measurements for GSM8K.

## Structure

| Path | Purpose |
|---|---|
| `FedAttn/Wrapper.py`, `FedAttn/utils.py` | Transformer forward pass, token segmentation and attention masks |
| `FedAttn/evaluate_gsm8k/` | Inference, scoring and experiment settings |
| `FedAttn/evaluate_gsm8k/system_cost/` | Communication, computation and memory estimates |
| `fidelity/run_fidelity.py` | Fidelity experiment entry |
| `fidelity/fig23/`, `fidelity/trajectory/`, `fidelity/kv/` | Three fidelity experiments |
| `fidelity/prepare_plot_data.py` | Validate and export plotting data |

## Setup

Keep `FedAttn/` and `fidelity/` as siblings. Run commands from `mathematical_reasoning/`.

Use the experiment environment with PyTorch, Transformers, Datasets, NumPy and jsonlines. Set the Qwen2.5 model and tokenizer paths in the Python entries. GSM8K loads through Datasets, with a local fallback at `FedAttn/evaluate_gsm8k/data/gsm8k` in `save_to_disk` format.

## Inference and evaluation

Edit `FedAttn/evaluate_gsm8k/run_evaluate_gsm8k.py` and `para_dict.py` to select the model, participants, synchronization interval, segmentation and samples. Update the entry's demo override together with the parameter grid. Give each new experiment a fresh `combinations_file` name and `write_path` to initialize its configuration list and completion record.

```bash
CUDA_VISIBLE_DEVICES=0 python FedAttn/evaluate_gsm8k/run_evaluate_gsm8k.py
```

Answer scoring runs with inference. For system costs, use `preprocess_prompt_and_res.py` to obtain prompt partitions and response-token counts, then `get_exp_stats()` and `get_perf_stat()` in `system_cost/preprocess_data.py` to aggregate the matching results.

## Fidelity

Configure `fidelity/run_fidelity.py` to select one experiment per run.

| `EXPERIMENT` | Selector | Measurement |
|---|---|---|
| `'fig23'` | `FIG23_JOB='fp32'` or `'7b_bf16'` | Final-block representation deviation across H or participant count |
| `'trajectory'` | `TRAJECTORY_H` | Representation deviation across blocks |
| `'kv'` | `KV_JOB='fp32'` or `'7b_bf16'` | KV fidelity across input retention, communication retention, asymmetric synchronization and block selection |

Set models, precision, `device`, segmentation and sample selection in the corresponding settings dictionary. `sample_count=2` selects a small run. Set the active `RESULTS_ROOT` and `PLOT_INPUTS_ROOT` assignments to fresh directories before starting new measurements.

```bash
python fidelity/run_fidelity.py
```

The entry measures fidelity and exports plotting summaries. Set `args.prepare_only=True` to generate plans or `args.export_only=True` to aggregate saved measurements. Select one mode at a time. `args.resume=True` resumes an interrupted run with matching settings.

## Demos and plotting data

Both included demos use Qwen2.5-0.5B, even segmentation and GSM8K test rows 0 and 1.

### Inference, evaluation and cost

The saved run uses four participants, four shots, H=2 and greedy decoding with up to 256 new tokens. Results are under:

```text
FedAttn/evaluate_gsm8k/demo_runs/inference_evaluation/results/main/
```

| Relative path | Content |
|---|---|
| `results_*_client*.jsonl` | Responses and correctness |
| `avg_acc/avg_acc_results_*.json` | Accuracy for quality plots |
| `collected_data/Qwen2.5-0.5B/main_record_*.json` | Analytical communication, computation and memory costs |

Match accuracy and cost records by configuration and timestamp. The saved accuracy is 0.0. Cost calculations assume two bytes per value.

### Final-block fidelity

The FP32 demo contains 64 measurements across 32 configuration points, varying H for Fig. 2 and participant count for Fig. 3. Representations are measured before the final RMSNorm over valid input tokens.

| Path | Content |
|---|---|
| `fidelity/results_demo/fidelity_results_fig23_full_fp32/` | Measurements and run settings |
| `fidelity/plot_inputs_demo/fig23/fp32/representation_deviation_summary.json` | Prepared Fig. 2/3 plotting data |

The summary contains means, standard deviations, standard errors and sample counts. For percentage plots, multiply both relative deviations and their error bars by 100.

Regenerate the saved demo summaries with the default `fig23` and `fp32` selection:

```bash
cd fidelity
python - <<'PY'
import run_fidelity as r
r.args.export_only = True
r.main()
PY
```
