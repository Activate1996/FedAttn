# FedAttn
Reproducibility code for FedAttn experiments on GSM8K.
# FedAttn

Code for the paper *Federated Attention (FedAttn)*. It reproduces the GSM8K results: FedAttn inference + EM evaluation*, and the system-cost model.
**run the inference experiments first then evaluation**.

## Repository layout

```
FedAttn/
├── Wrapper.py                          # FedAttn inference core: per-layer masked attention (local / KV-sync)
├── utils.py                            # attention masks, block schedules (get_points), token segmentation (token_chunk)
└── evaluate_gsm8k/
    ├── run_evaluate_gsm8k.py               # ENTRY POINT — launches the experiments (reads grids from para_dict.py)
    ├── run_evaluate_gsm8k_multiprocess.py  # per-run driver: builds task list, sets result paths, runs one config
    ├── main_evaluate_gsm8k.py              # FedAttn generation loop + Pass@1 exact-match (EM) scoring
    ├── para_dict.py                        # experiment grids for each comm_policy (models, N, H, shots, ...)
    ├── preprocess_prompt_and_res.py        # builds token-length records the plotting stage needs
    ├── prompt/                             # few-shot chain-of-thought exemplars (gsm8k_prompt_0.txt … _7.txt)
    ├── results/                            # (auto-created) all experiment outputs are written here
    └── plot_figures/
        ├── preprocess_data.py              # reads results/, computes per-run system cost (calls the cost model)
        ├── utils_get_performance_stats.py  # system-cost model: read_shape, summarize_prefill / summarize_decode
        ├── plot_main_figs1.py              # plots communication frontier and computation frontier
        ├── ablapnas1.py                    # plots A sparse local, B sparse KV, C publisher, D block selection
        ├── utils_plot_main_fig_comms.py    # communication-frontier subplot helpers 
        └── utils_plot_main_fig_comps_mem.py# computation-frontier subplot helpers 
```

## Install
```bash
pip install -r requirements.txt   # torch, transformers, datasets, numpy, jsonlines, matplotlib
```
The Qwen2.5 base models and GSM8K download automatically on first use (`from_pretrained` / `load_dataset`). To use local copies, set `checkpoint_path` in `para_dict.py` to a local model directory and place GSM8K under `evaluate_gsm8k/data/gsm8k`.

## Run experiments (inference + EM)

The run is driven by `para_dict.py`, selected by `comm_policy` in `run_evaluate_gsm8k.py`.

```bash
cd FedAttn/evaluate_gsm8k
python run_evaluate_gsm8k.py          # set comm_policy inside the file
```
All models use Qwen2.5 base (0.5B/1.5B/3B/7B), the four segmentation settings (`even`, `smart`, `even_question_last`, `smart_question_last` = TokAg, SemAg, TokEx, SemEx), greedy decoding, 256 max new tokens, and bf16. Greedy decoding makes results deterministic.

**Outputs.** Results are written under `evaluate_gsm8k/results/<comm_policy>/`, including `avg_acc/avg_acc_results_*.json` (the per-configuration EM records). **system cost read these files, so inference exp must be run first.**

## System cost

`plot_figures/preprocess_data.py` reads the inference records and, via the cost model in `utils_get_performance_stats.py`, returns per-participant communication (bytes), FLOPs, and peak memory. `read_shape` pulls each backbone's architecture from its config automatically.
```bash
cd FedAttn/evaluate_gsm8k/plot_figures
python -c "import preprocess_data as p; print(p.get_exp_stats('../results', 'main'))"
```

## Results and Figures 

The plotting scripts read `results/…/avg_acc/avg_acc_results_*.json` and call the cost model for the shaded cost regions. **Run them only after the corresponding experiments have finished.**
```bash
cd FedAttn/evaluate_gsm8k/plot_figures
python plot_main_figs1.py    
python abla.py          
```
