# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:00:32 2026

@author: DengXiumei
"""

# -*- coding: utf-8 -*-
"""
=================
Generates (Fidelity-Utility Gap) figure

Row order (top -> bottom):
    Panel A  Sparse local attention      (comm_policy = "sample_uni_comp")
    Panel B  Sparse KV synchronization   (comm_policy = "sample_uni_comm")
    Panel C  Task publisher sync. freq.  (comm_policy = "uni_extra")
    Panel D  Block-selection schemes     (uni_fr / uni_bk / inc_gp / dec_gp)

Column order (left -> right):
    Qwen2.5-0.5B, 1.5B, 3B, 7B

"""

import os
import sys
import copy
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess_prompt_and_res import generate_res_lens_file
from preprocess_data import get_exp_stats, get_info
from plot_main_figs import load_main_data, generate_xy_axis
from utils_get_performance_stats import read_shape, get_model_weight_byte
from plot_abla_figs_uni_comp import (
    load_main_data_no_perf,
    generate_xy_axis_no_perf,
)
from utils_plot_uni_fr_uni_bk_inc_gp_dec_gp import plot_bar

YAXIS_ARROW_TOP = 1.135
XAXIS_ARROW_RIGHT = 1.065
# ===========================================================================
RCPARAMS = {
    # Font family
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans"],

    # Sizes
    "font.size":             10,   # was 11
    "axes.labelsize":         9,   # was 10
    "axes.titlesize":        10,   # was 11
    "axes.titleweight":      "normal",
    "xtick.labelsize":        9,   # was 10
    "ytick.labelsize":        9,   # was 10
    "legend.fontsize":        9,   # was 11
    "legend.title_fontsize":  9,   # was 11

    # Spines, grid
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.8,

    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,

    "hatch.linewidth":   0.5,

    "pdf.fonttype": 42,
    "ps.fonttype":  42,

    # DPI
    "figure.dpi":  150,
    "savefig.dpi": 300,
}
plt.rcParams.update(RCPARAMS)


# ===========================================================================
# Constants
# ===========================================================================
MODELS = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]

# Segmentation (panels A/B/C lines)
LINE_STYLES = {
    "even":                {"marker": "o", "linestyle": "-",  "color": "navy",
                            "label": "TokAg"},
    "even_question_last":  {"marker": "^", "linestyle": ":",  "color": "#2E7D32",
                            "label": "TokEx"},
    "smart":               {"marker": "s", "linestyle": "--", "color": "red",
                            "label": "SemAg"},
    "smart_question_last": {"marker": "v", "linestyle": "-.", "color": "purple",
                            "label": "SemEx"},
}
SPLIT_WAYS = list(LINE_STYLES.keys())
PROMPT_SEG_DISPLAY = {k: v["label"] for k, v in LINE_STYLES.items()}


SCHEME_ORDER = ["uni_fr", "uni_bk", "inc_gp", "dec_gp"]
SCHEME_LABELS = {
    "uni_fr": "Shallow-Half",
    "uni_bk": "Deep-Half",
    "inc_gp": "Progressive",
    "dec_gp": "Regressive",
}
SCHEME_BAR_COLORS  = ["navy",     # Shallow-Half
                      "red",      # Deep-Half
                      "orange",   # Progressive
                      "#2E7D32"]  # Regressive (dark green)

SCHEME_BAR_HATCHES = ["///",  "///", "...",   "..."]

PANEL_D_XTICK_ROTATION = 25
PANEL_D_HEADROOM = 1.05

# main_keys
MAIN_KEYS_TEMPLATE = {
    "uni_extra": {
        "num_local_forwards_last": {
            "model": None, "file_type": "avg_acc_results", "num_shots": 4,
            "num_clients": 4, "num_local_forwards": None,
            "if_do_sample": False, "max_new_tokens": 256, "split_way": None,
        },
    },
    "sample_uni_comp": {
        "ratio_comp": {
            "model": None, "file_type": "avg_acc_results", "num_shots": 4,
            "num_clients": 4, "num_local_forwards": None,
            "if_do_sample": False, "max_new_tokens": 256, "split_way": None,
        },
    },
    "sample_uni_comm": {
        "ratio_comm": {
            "model": None, "file_type": "avg_acc_results", "num_shots": 4,
            "num_clients": 4, "num_local_forwards": None,
            "if_do_sample": False, "max_new_tokens": 256, "split_way": None,
        },
    },
}

X_LABELS = {
    "num_local_forwards_last": "Synchronization interval",
    "ratio_comp":               "Retention ratio",
    "ratio_comm":               "Retention ratio",
}


# ===========================================================================
# Helpers
# ===========================================================================
def _set_xlim_by_max(ax, x):
    mx = max(x)
    if   mx == 24: ax.set_xlim(min(x) - 1.05, mx + 1.05)
    elif mx == 28: ax.set_xlim(min(x) - 1.18, mx + 1.18)
    elif mx == 36: ax.set_xlim(min(x) - 1.50, mx + 1.50)
    elif mx == 8:  ax.set_xlim(min(x) - 0.32, mx + 0.32)
    elif mx == 9:  ax.set_xlim(min(x) - 0.35, mx + 0.35)


def _y_arrow(ax):
    """Decorative upward arrow at the top of the y-axis (currently unused)."""
    ax.annotate(
        "", xy=(0, 0), xytext=(0, 1.12),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<|-", color="black", lw=0.6),
    )


def _apply_overrides_to_subplot(ax):
    ax.tick_params(axis="both", labelsize=9)
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=9)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)
    if ax.title.get_text():
        ax.title.set_fontsize(10)

    ax.annotate('', xy=(0, 0), xytext=(0, YAXIS_ARROW_TOP),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<|-', color='black'),
                annotation_clip=False)
    ax.annotate('', xy=(0, 0), xytext=(XAXIS_ARROW_RIGHT, 0),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<|-', color='black'),
                annotation_clip=False)
# ---------------------------------------------------------------------------
# Panels A, B, C
# ---------------------------------------------------------------------------
def plot_line_subplot(
    ax, model, comm_policy, x_axis, data, data_path, script_dir,
    model_w_byte, main_keys, loader="no_perf",
):
    """One subplot of Panel A, B, or C."""

    for key in SPLIT_WAYS:
        cfg = main_keys[comm_policy][x_axis]
        cfg["model"]     = model
        cfg["split_way"] = key
        if "num_local_forwards" in cfg:
            cfg["num_local_forwards"] = 9 if model == "Qwen2.5-3B" else 8

        if loader == "no_perf":
            main_data = load_main_data_no_perf(
                data, cfg, data_path, script_dir, comm_policy)
            x, y_acc = generate_xy_axis_no_perf(
                main_data=main_data, x_axis=x_axis,
                model_w_byte=model_w_byte, comm_policy=comm_policy,
            )
        else:
            main_data = load_main_data(
                data, cfg, data_path, script_dir, comm_policy)
            x, y_acc, _ = generate_xy_axis(
                main_data=main_data, x_axis=x_axis,
                model_w_byte=model_w_byte, comm_policy=comm_policy,
            )

        style = LINE_STYLES[key]
        ax.plot(
            x, y_acc["All Participants"] * 100,
            marker=style["marker"], linestyle=style["linestyle"],
            color=style["color"],
            linewidth=1.5, markersize=8,
            markerfacecolor="none", markeredgecolor=style["color"],
            markeredgewidth=1.5,
        )

    ax.set_xlabel(X_LABELS[x_axis])
    ax.set_ylabel("EM Acc. (%)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    _set_xlim_by_max(ax, x)


# ---------------------------------------------------------------------------
# Panel D: bar plot
# ---------------------------------------------------------------------------
def build_panel_D_acc_dict(data_path, script_dir, cache_dir):
    cache_file = os.path.join(cache_dir, "uni_fr_uni_bk_inc_gp_dec_gp.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    keys = ["All Participants",
            r"Participant $N$",
            r"Participants $1, \ldots, N-1$"]

    for cp in SCHEME_ORDER:
        generate_res_lens_file(os.path.join(data_path, cp))

    acc_dict = {}
    for model in MODELS:
        acc_dict[model] = {}
        for split_way in SPLIT_WAYS:
            acc_dict[model][split_way] = {k: {} for k in keys}

            for comm_policy in SCHEME_ORDER:
                data = get_exp_stats(data_path, comm_policy)
                main_keys = {
                    "model":          model,
                    "file_type":      "avg_acc_results",
                    "num_shots":      4,
                    "num_clients":    4,
                    "num_local_forwards": 9 if model == "Qwen2.5-3B" else 8,
                    "if_do_sample":   False,
                    "max_new_tokens": 256,
                    "split_way":      split_way,
                }
                main_data = load_main_data(
                    data, main_keys, data_path, script_dir, comm_policy)
                if not main_data:
                    continue
                record = main_data[0]
                client_field = {
                    "All Participants":
                        [str(i) for i in range(record["num_clients"])],
                    r"Participant $N$":
                        [str(record["num_clients"] - 1)],
                    r"Participants $1, \ldots, N-1$":
                        [str(i) for i in range(max(record["num_clients"] - 1, 1))],
                }
                acc_dict[model][split_way]["All Participants"][comm_policy] = \
                    record["data"].get("avg")
                acc_dict[model][split_way][r"Participant $N$"][comm_policy] = \
                    record["data"].get(client_field[r"Participant $N$"][0])
                vals = [record["data"][k]
                        for k in client_field[r"Participants $1, \ldots, N-1$"]]
                acc_dict[model][split_way][r"Participants $1, \ldots, N-1$"][comm_policy] = \
                    sum(vals) / len(vals)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(acc_dict, f, ensure_ascii=False, indent=2)
    return acc_dict


def call_plot_bar_preserving_rcparams(*args, **kwargs):
    saved = dict(plt.rcParams)
    try:
        plot_bar(*args, **kwargs)
    finally:
        plt.rcParams.update(saved)


def _override_bar_colors(ax, n_methods, n_cats):
    expected = n_methods * n_cats
    if len(ax.patches) < expected:
        return  
    for m_idx in range(n_methods):
        for c_idx in range(n_cats):
            patch = ax.patches[m_idx * n_cats + c_idx]
            patch.set_facecolor(SCHEME_BAR_COLORS[m_idx])
            patch.set_linewidth(0.5)


def _tighten_panel_D_ylim(ax, n_methods, n_cats):
    expected = n_methods * n_cats
    if len(ax.patches) < expected:
        return
    y_max = max(ax.patches[i].get_height() for i in range(expected))
    if y_max > 0:
        ax.set_ylim(0, y_max * PANEL_D_HEADROOM)


def plot_panel_D_subplot(ax, model, acc_dict, metric="All Participants"):
    data = {}
    for split_way in SPLIT_WAYS:
        data[PROMPT_SEG_DISPLAY[split_way]] = {
            SCHEME_LABELS[scheme]: acc_dict[model][split_way][metric].get(scheme, 0.0)
            for scheme in SCHEME_ORDER
        }

    call_plot_bar_preserving_rcparams(
        data,
        split_way=None,         
        per_metric=metric,        
        title=None,              
        xlabel=None,
        ylabel="EM Acc. (%)",
        ax=ax,
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    _override_bar_colors(ax, n_methods=len(SCHEME_ORDER), n_cats=len(SPLIT_WAYS))
    _tighten_panel_D_ylim(ax, n_methods=len(SCHEME_ORDER), n_cats=len(SPLIT_WAYS))
    for ann in [t for t in list(ax.texts) if t.get_text() == ""]:
        ann.remove()

# ===========================================================================
# Main
# ===========================================================================
def main():
    # ---------------- Paths ----------------
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    cache_dir_D = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        r"\figures",
        "uni_fr_uni_bk_inc_gp_dec_gp",
    )

    # ---------------- Preprocess for line panels ----------------
    for cp in ["sample_uni_comp", "sample_uni_comm", "uni_extra"]:
        generate_res_lens_file(os.path.join(data_path, cp))

    # ---------------- Build acc_dict for Panel D ----------------
    print("Building Panel D data ...")
    acc_dict = build_panel_D_acc_dict(data_path, script_dir, cache_dir_D)

    # ---------------- Set up figure ----------------
    fig_width  = 15.2
    fig_height = 9
    PANEL_D_HEIGHT_RATIO = 1.1
    fig, axes = plt.subplots(
        4, 4, figsize=(fig_width, fig_height),
        gridspec_kw={
            "hspace": 0.49,
            "wspace": 0.3,
            "height_ratios": [1, 1, 1, PANEL_D_HEIGHT_RATIO],
        },
    )

    # ============== Panel A: sample_uni_comp ==============
    print("Plotting Panel A (sparse local attention) ...")
    comm_policy, x_axis = "sample_uni_comp", "ratio_comp"
    data_A = get_exp_stats(data_path, comm_policy)
    for col_idx, model in enumerate(MODELS):
        model_shape = read_shape(repo_id="Qwen/" + model)
        model_w_byte = get_model_weight_byte(model_shape)
        plot_line_subplot(
            axes[0, col_idx], model, comm_policy, x_axis, data_A,
            data_path, script_dir, model_w_byte,
            copy.deepcopy(MAIN_KEYS_TEMPLATE), loader="no_perf",
        )

    # ============== Panel B: sample_uni_comm ==============
    print("Plotting Panel B (sparse KV synchronization) ...")
    comm_policy, x_axis = "sample_uni_comm", "ratio_comm"
    data_B = get_exp_stats(data_path, comm_policy)
    for col_idx, model in enumerate(MODELS):
        model_shape = read_shape(repo_id="Qwen/" + model)
        model_w_byte = get_model_weight_byte(model_shape)
        plot_line_subplot(
            axes[1, col_idx], model, comm_policy, x_axis, data_B,
            data_path, script_dir, model_w_byte,
            copy.deepcopy(MAIN_KEYS_TEMPLATE), loader="no_perf",
        )

    # ============== Panel C: uni_extra ==============
    print("Plotting Panel C (task publisher synchronization) ...")
    comm_policy, x_axis = "uni_extra", "num_local_forwards_last"
    data_C = get_exp_stats(data_path, comm_policy)
    for col_idx, model in enumerate(MODELS):
        model_shape = read_shape(repo_id="Qwen/" + model)
        model_w_byte = get_model_weight_byte(model_shape)
        plot_line_subplot(
            axes[2, col_idx], model, comm_policy, x_axis, data_C,
            data_path, script_dir, model_w_byte,
            copy.deepcopy(MAIN_KEYS_TEMPLATE), loader="perf",
        )

    # ============== Panel D: block-selection bars (plot_bar untouched) ==============
    for col_idx, model in enumerate(MODELS):
        plot_panel_D_subplot(axes[3, col_idx], model, acc_dict,
                             metric="All Participants")

    for row in axes:
        for ax in row:
            _apply_overrides_to_subplot(ax)

    # ============== Column headers==============
    for col_idx, model in enumerate(MODELS):
        axes[0, col_idx].set_title(model, fontsize=10, pad=16, fontweight="bold")

    # ============== Panel labels A/B/C/D  ==============
    for row_idx, label in enumerate(["A", "B", "C", "D"]):
        axes[row_idx, 0].text(
            -0.3, 1.12, label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=16, fontweight="bold",
            va="top", ha="left",
        )

    # ============== Shared legends ==
    # --- Row 1 --------------------------------
    seg_handles = [
        Line2D(
            [0], [0],
            marker=s["marker"], linestyle=s["linestyle"], color=s["color"],
            markerfacecolor="none", markeredgecolor=s["color"],
            linewidth=1.5, markersize=8, markeredgewidth=1.5,
        )
        for s in LINE_STYLES.values()
    ]
    seg_labels = [s["label"] for s in LINE_STYLES.values()]   # TokAg/TokEx/SemAg/SemEx

    leg_seg = fig.legend(
        seg_handles, seg_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,                       
        fontsize=9,
        frameon=False,              
        handlelength=2.5, handletextpad=0.5,
        columnspacing=2.5, labelspacing=0.3,
        borderpad=0.2,
    )

    # --- Row 2 ---------
    scheme_handles = [
        Patch(
            facecolor=SCHEME_BAR_COLORS[i],
            edgecolor="black", linewidth=0.5,
            hatch=SCHEME_BAR_HATCHES[i], alpha=0.85,
        )
        for i in range(len(SCHEME_ORDER))
    ]
    scheme_labels = [SCHEME_LABELS[s] for s in SCHEME_ORDER]

    leg_scheme = fig.legend(
        scheme_handles, scheme_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=4,                      
        fontsize=9,
        frameon=False,               
        handlelength=2.2, handletextpad=0.5,
        columnspacing=2.2, labelspacing=0.3,
        borderpad=0.2,
    )

    fig.add_artist(leg_seg)

    # ============== Layout & save ==============
    fig.tight_layout(rect=[0.02, 0.10, 0.96, 0.985])

    out_pdf = os.path.join(save_dir, "fig4.pdf")
    out_png = os.path.join(save_dir, "fig4.png")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight",
                dpi=600, pad_inches=0.15)
    fig.savefig(out_png, format="png", bbox_inches="tight",
                dpi=300, pad_inches=0.15)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()