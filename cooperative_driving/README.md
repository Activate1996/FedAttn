# FedAttn: cooperative driving

Inference, evaluation and fidelity experiments for **“Federated attention for collective intelligence in large language models.”** Run commands from `cooperative_driving/`.

| Task | ID | Checkpoint | Frames per CAV | Metric |
|---|---|---:|---:|---|
| Task 1 | q1 | 1387 | 1 | Pooled binary presence F1 |
| Task 2 | q4 | 490 | 1 | Pooled 4 m location coverage F1 |
| Task 3 | q5 | 490 | 2 | Mean waypoint L2 and collision rate over 3 s |

## Setup

Keep `LLaVA/`, `DMSTrack/`, `fidelity/` and `demo/` at the same level.

```bash
conda env create -f LLaVA/environment_llava_v2v.yml
conda env create -f LLaVA/environment_dmstrack3.yml

conda activate llava_v2v
python -m pip install -e ./LLaVA

conda activate dmstrack3
python -m pip install -e ./DMSTrack/V2V4Real
git clone https://github.com/xinshuoweng/AB3DMOT.git DMSTrack/AB3DMOT
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox.git DMSTrack/AB3DMOT/Xinshuo_PyToolbox
```

Download these archives from [V2V-GoT-QA](https://huggingface.co/datasets/eddyhkchiu/V2V-GoT-QA/tree/main):

```text
dataset_processed_features_and_gt.zip
dataset_jsons.zip
dataset_jsons_v2vllm_q1234.zip
model_ckpt.zip
model_ckpt_v2vllm_q1234_f1.zip
```

Place the extracted assets at the following paths. `<task>` is `q1`, `q4` or `q5`, `<cav>` is `ego` or `1`, and frame IDs are zero-padded.

```text
DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/
    v2v4real_3d_grounding_qa_dataset_v2vllm<task>.json
    <cav>/<frame>_regression_map.npy
    <cav>/<frame>_classification_map.npy
    <cav>/<frame>_detection_box_score.npy

LLaVA/checkpoints/llava-v1.5-7b-task-lora/
    llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vllm<task>_10ep_both_shallow_f<frames>/checkpoint-<step>/
```

Use the frame and checkpoint values in the task table. Q5 evaluation also uses the processed ground-truth and LiDAR-pose files. Cache `liuhaotian/llava-v1.5-7b` and `openai/clip-vit-large-patch14-336` for offline inference.

## Inference and evaluation

Select one task and one experiment group per run.

| Group | Configuration sweep |
|---|---|
| H | Synchronization interval |
| A | Input retention |
| B | Communication retention |
| C | Collaborator interval with a fixed publisher interval |
| D | Shallow, Deep, Progressive and Regressive block selection |

In `LLaVA/llava/eval/model_vqa_loader_Fed_v2v_llm.py`, set `ACTIVE_QUESTION`, `EXPERIMENT_GROUP`, `GPU_id` and the corresponding settings dictionary. Set `DEMO_MODE=False` for quality experiments. Q1 selects every 20th QA row from row 0, and Q4/Q5 use all rows.

```bash
conda activate llava_v2v
python LLaVA/llava/eval/model_vqa_loader_Fed_v2v_llm.py
```

In `LLaVA/scripts/eval_v2v4real_3d_grounding_v8.py`, set `QUESTION_ID` to `1`, `4` or `5`, select the matching `EXPERIMENT_GROUP`, and set `DEMO_MODE=False`.

```bash
conda activate dmstrack3
python LLaVA/scripts/eval_v2v4real_3d_grounding_v8.py
```

Evaluation produces per-sample metrics and configuration-level aggregates under `results/evaluation/{per_sample,aggregate}/v2v_llm/<group>/`.

## Fidelity

In `fidelity/run_v2v_fidelity.py`, set `ACTIVE_QUESTION`, `EXPERIMENT_GROUP` (`A`–`D`), `GPU_ID` and the sampling settings. `SUBSAMPLE_EVERY=10` selects every tenth sample from the quality sample set. Use a fresh output directory for each changed experiment definition.

```bash
conda activate llava_v2v
python fidelity/run_v2v_fidelity.py
```

Measurements and prepared plotting data are saved under `fidelity/results/v2v_fidelity/<task>/<group>/`. To combine completed runs, set `SUMMARY_PATHS` in `fidelity/prepare_plot_data.py` and run:

```bash
python fidelity/prepare_plot_data.py
```

## Demo

### Q1 evaluation

Set `DEMO_MODE=True`, `QUESTION_ID=1` and `EXPERIMENT_GROUP='H'` in the evaluation entry, then run its command above. The demo evaluates two QA samples under CenAttn and H=2,3,4,8. Outputs are written to `results/demo/q1_evaluation/`.

### Q1/A fidelity

```bash
python demo/run_fidelity_demo.py
```

This CPU demo processes seven samples with 455 saved prefill records and produces 64 configuration points under `results/demo/q1_fidelity_A/`.

### Plotting data

The processed reference files are included under `demo/`:

| File | Plotting use |
|---|---|
| `q1_evaluation/expected_outputs/aggregate/v2v_llm/H/v2v_llm_q1.json` | Q1 F1 by configuration |
| `q1_fidelity_A/expected_outputs/plot_inputs/v2v_fidelity_summary.json` | KV fidelity means and sample counts |
| `q1_fidelity_A/expected_outputs/plot_inputs/v2v_fidelity_plot_points.csv` | Ordered fidelity points |

The CSV provides `mean_percent = 100 * mean`. Multiply `std` or `sem` by 100 for percentage error bars.
