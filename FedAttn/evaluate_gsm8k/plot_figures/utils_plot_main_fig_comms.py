# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 19:57:30 2025

@author: DengXiumei
"""
import json
import os
import re
import jsonlines
import numpy as np
import numbers

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as patches

from figs_preview import figs_preview
from utils_get_performance_stats import read_shape
from preprocess_data import get_exp_stat

from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "figure.dpi": 150
})


def human_bytes(n):
    """
    将字节数格式化成人类可读的字符串。

    参数:
        n: 原始字节数（int）。

    返回:
        str，例如 "123.45 MB"
    """

    units = ["B","KB","MB","GB","TB","PB"]  # 单位序列
    
    if isinstance(n, np.ndarray):
        x = np.max(n)
        for u in units:
            # 当数值小于 1024 或者已经到最后一个单位时，输出并返回
            if x < 1000 or u == units[-1]:
                break
            x /= 1000  # 否则除以 1024 继续转换到更大单位
        return n/(1000**(units.index(u))), u  
    
    elif isinstance(n, numbers.Number):  # 包括 int, float, complex
        x = float(n)
        for u in units:
            # 当数值小于 1024 或者已经到最后一个单位时，输出并返回
            if x < 1000 or u == units[-1]:
                return x, u
            x /= 1000  # 否则除以 1024 继续转换到更大单位

prompt_seg_map = {
    "even": "TokAg",           # 按token数量平均分10份
    "even_question_last": "TokEx",            # 最后一份是问题，剩下按token平均分
    "smart": "SemAg",        # 平均分完整instruction，问题给最后一个
    "smart_question_last": "SemEx"          # 最后一个只有问题，剩下平均分instruction
}


def plot_main_fig_comm(
        x, y_acc, y_cost, xlabel, main_keys, 
        left_ylim=None, show_grid=True, 
        save_path=None,
        ax=None, figsize=(4.3, 3.3),
        accs_map={ 
            'All Participants': "Average", 
            r'Participant $N$': "Task publisher", 
            r'Participants $1, \ldots, N-1$': "Best collaborator", 
            },
        metrics_map={ 
            'Prefilling phase': 'Prefill', 
            'Decoding phase': 'Decode'
            }
        ):
            
    model = main_keys["model"]
    prompt_seg = main_keys["split_way"]
    
    prompt_seg = prompt_seg_map[prompt_seg]
    
    """
    左轴：多颜色区分
    右轴：单一颜色的不同深浅 + 填充区域
    左右轴图例分开显示，带美化效果
    """
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.titleweight": "bold",
        "axes.grid": show_grid,
        "grid.alpha": 0.25,
        "font.size": 10,
        "figure.dpi": 150
    })
    
    # 左轴三条线的样式配置
    line_styles = {
        'All Participants': {'marker': 's', 'linestyle': '-', 'color': 'navy'},      # 深蓝
        r'Participant $N$': {'marker': 'o', 'linestyle': '--', 'color': 'red'},     # 红色
        r'Participants $1, \ldots, N-1$': {'marker': '^', 'linestyle': ':', 'color': 'orange'}  # 橙色
        }
        
    # 右轴颜色配置（绿色系的不同深浅）
    fill_colors = {
        'darkest': '#013220',     # 极深绿（几乎黑绿）
        'very_dark': '#1B5E20',   # 很深的绿
        'dark': '#2E7D32',        # 深绿（比你原来的更深）    
        'base': '#2E8B57',      # 深绿
        'medium': '#3CB371',    # 中绿  
        'light': '#90EE90',     # 浅绿
        'extra_light': '#F0FFF0' # 极浅绿（用于填充）
        }
    
    
    # 新增两根线的样式（绿色系+填充）
    fill_line_styles = {
        'dark': {
            'color': '#2E7D32',        # 深绿
            'linestyle': '-', 
            'linewidth': 2,            # 稍细的线
            'alpha': 0.8,              # 线条稍透明
            'fill_color': '#2E7D32',
            'fill_alpha': 0.5,
            'fill_hatch': '/////'        # 斜线填充
            },
        'light': {
            'color': '#388E3C',        # 中绿
            'linestyle': '-',
            'linewidth': 2,
            'alpha': 0.8,
            'fill_color': '#388E3C', 
            'fill_alpha': 0.25,
            'fill_hatch': '\\\\\\'        # 点状填充
            }
        }    

    if ax:
        ax1 = ax
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    # 左轴：Accuracy（多种颜色）
    acc_keys = list(accs_map.keys())  
    for key in acc_keys:
        style = line_styles[key]
        ax1.plot(
            x, y_acc[key]*100, 
            marker=style['marker'], 
            linestyle=style['linestyle'], 
            color=style['color'], 
            linewidth=2,
            markersize=8,
            markerfacecolor='none',
            markeredgecolor=style['color'],
            markeredgewidth=2
            )


    if len(list(accs_map.keys())) == 1:
        ax1.set_xlabel("Synchronization interval", fontsize=10)
        ax1.set_ylabel("EM Acc. (%)", color='navy', fontsize=10)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=9)
        ax1.tick_params(axis='y', labelcolor='navy', labelsize=9)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  
    else:
        ax1.set_xlabel("Synchronization interval", fontsize=10)
        ax1.set_ylabel("EM Acc. (%)", color='black', fontsize=10)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=9)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=9)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # 右轴：KV Communication（绿色系 + 填充区域）
    ax2 = ax1.twinx()
    
    # 绘制主线
    comm_lines, unit = human_bytes(y_cost['Prefilling phase']['Key-Value matrices'])
    ax2.plot(
        x, comm_lines, 
        color=fill_line_styles['dark']['color'],
        linestyle=fill_line_styles['dark']['linestyle'],
        linewidth=fill_line_styles['dark']['linewidth'],
        alpha=fill_line_styles['dark']['alpha'])
    
    ax2.fill_between(
        x, comm_lines,
        alpha=fill_line_styles['dark']['fill_alpha'],
        color=fill_line_styles['dark']['fill_color'],
        hatch=fill_line_styles['dark']['fill_hatch'],
        interpolate=True)
    
    ax2.set_ylabel(f"Comm. cost ({unit})", color=fill_colors['very_dark'], fontsize=10)
    ax2.tick_params(axis='y', labelcolor=fill_colors['very_dark'], labelsize=9)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    
    # 右轴倒置
    ax2.invert_yaxis()
    ax2.spines["right"].set_visible(True)
    
    # 设置xy轴显示范围
    # ax2.set_ylim(np.asarray(comm_lines).min()*0.95, np.asarray(comm_lines).max()*1.05)
    if max(x) == 24:
        ax1.set_xlim(min(x)-1.05, max(x)+1.05)  
    elif max(x) == 28:
        ax1.set_xlim(min(x)-1.18, max(x)+1.18)
    elif max(x) == 36:
        ax1.set_xlim(min(x)-1.5, max(x)+1.5)     
    elif max(x) == 8:
        ax1.set_xlim(min(x)-0.32, max(x)+0.32)            
    elif max(x) == 9:
        ax1.set_xlim(min(x)-0.35, max(x)+0.35)         
    
    
    
    # === 添加Y轴箭头 ===
    # 左轴：向上箭头
    ax1.annotate('', xy=(0, 0), xytext=(0, 1.15),
                 xycoords='axes fraction', textcoords='axes fraction', 
                 arrowprops=dict(arrowstyle='<|-', 
                                 color='black', 
                                 # lw=2
                                 )
                 )
    # 右轴：向下箭头
    ax2.annotate('', xy=(1, 1), xytext=(1, -0.1),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='<|-', 
                                 color='black', 
                                 # lw=2
                                 )
                 )
    
    # 创建左轴图例handles
    left_handles = []
    for key in acc_keys:
        style = line_styles[key]
        handle = Line2D(
            [0], [0], 
            marker=style['marker'],
            linestyle=style['linestyle'],
            color=style['color'],
            linewidth=2,
            markersize=6,
            markerfacecolor='none',
            markeredgecolor=style['color'],
            markeredgewidth=2
            )
        left_handles.append(handle)
    
    # 创建右轴图例handle（填充区域样式）
    right_handle = patches.Patch(
        facecolor=fill_line_styles['dark']['fill_color'],  # 填充颜色
        alpha=fill_line_styles['dark']['fill_alpha'],      # 透明度
        hatch=fill_line_styles['dark']['fill_hatch'],      # 填充样式
        edgecolor=fill_line_styles['dark']['color'],       # 边框颜色
        linewidth=fill_line_styles['dark']['linewidth']    # 边框宽度
        )
    
    # === 美化左轴图例 ===
    acc_labels = list(accs_map.values())
    acc_labels_title = 'Response Quality:'
    if ax:
        legend_paras = { 
            "Qwen2.5-0.5B": { 
                "left_bbox_to_anchor": (0.22, 0.7),
                "right_bbox_to_anchor": (0.22, 0.98),
                },
            "Qwen2.5-1.5B": {
                "left_bbox_to_anchor": (0.2, 0.7),
                "right_bbox_to_anchor": (0.2, 0.98),           
                },
            "Qwen2.5-3B": {
                "left_bbox_to_anchor": (0.2, 0.7),
                "right_bbox_to_anchor": (0.2, 0.98),            
                },
            "Qwen2.5-7B": {
                "left_bbox_to_anchor": (0.2, 0.7),
                "right_bbox_to_anchor": (0.2, 0.98),            
                },     
            } 
    else:
        legend_paras = { 
            "Qwen2.5-0.5B": { 
                "left_bbox_to_anchor": (0.3, 0.63),
                "right_bbox_to_anchor": (0.3, 0.88),
                },
            "Qwen2.5-1.5B": {
                "left_bbox_to_anchor": (0.3, 0.63),
                "right_bbox_to_anchor": (0.3, 0.88),           
                },
            "Qwen2.5-3B": {
                "left_bbox_to_anchor": (0.3, 0.63),
                "right_bbox_to_anchor": (0.3, 0.88),            
                },
            "Qwen2.5-7B": {
                "left_bbox_to_anchor": (0.3, 0.63),
                "right_bbox_to_anchor": (0.3, 0.88),            
                },     
            }        
              
    left_legend = ax1.legend(left_handles, acc_labels, 
                             loc='upper left',
                             title=acc_labels_title,
                            
                             # 字体设置
                             fontsize=9,
                             title_fontsize=9,
                            
                             # 布局设置
                             ncol=1,                         # 1列垂直排列
                             labelspacing=0.2,               # 条目间距
                             handlelength=2,               # 图例标记长度
                             handletextpad=0.6,              # 标记与文字间距
                             columnspacing=1.0,              # 列间距
                            
                             # 美化效果
                             frameon=True,                   # 显示边框
                             fancybox=True,                  # 圆角边框
                             shadow=False,                    # 阴影效果
                             framealpha=1,                # 背景透明度
                             facecolor="None",         
                             edgecolor='black',      
                             borderpad=0.6,
                             
                             # 位置设置
                             bbox_to_anchor=legend_paras[model]["left_bbox_to_anchor"],
                             bbox_transform=ax1.transAxes,
                            
                             # 其他美化
                             markerscale=1.0,                # 标记缩放比例
                             markerfirst=True,
                             )               # 标记在前，文字在后
    
    # === 美化右轴图例 ===
    right_legend = ax2.legend([right_handle], ['Key-Value Transmission'], 
                              loc='upper left',
                              title='Communication Cost:',
                             
                              # 字体设置
                              fontsize=9,
                              title_fontsize=9,
    
                              # 布局设置
                              ncol=1,                         # 1列垂直排列
                              labelspacing=0.2,               # 条目间距
                              handlelength=2,               # 图例标记长度
                              handletextpad=0.6,              # 标记与文字间距
                              columnspacing=1.0,              # 列间距
                                
                              # 美化效果
                              frameon=True,                   # 显示边框
                              fancybox=True,                  # 圆角边框
                              shadow=False,                    # 阴影效果
                              framealpha=1,                # 背景透明度
                              facecolor="None",          
                              edgecolor=fill_colors['very_dark'],          
                              borderpad=0.6,
                             
                              # 位置设置
                              bbox_to_anchor=legend_paras[model]["right_bbox_to_anchor"],
                              bbox_transform=ax2.transAxes
                              )
    
    left_legend._legend_box.align = "left"
    right_legend._legend_box.align = "left"

    left_legend.get_frame().set_linewidth(1.5)      # 边框粗细
    right_legend.get_frame().set_linewidth(1.5)     # 边框粗细  

    
    # 确保两个图例都显示
    ax1.add_artist(left_legend)

    # 设置标题
    if ax:
        ax1.set_title(model+'\n'+prompt_seg, fontsize=10)
        return ax1, ax2  
    else:
        fig.suptitle(prompt_seg, fontsize=10, y=0.9)
        fig.tight_layout()
        plt.show()
        
        if save_path:
            save_path = os.path.join(save_path, "indiv")
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"comm_{model}_{main_keys["split_way"]}.pdf")
            fig.savefig(file_path, format="pdf", bbox_inches="tight")
            plt.close()  
            

def plot_main_fig_comm_bottom_legends(
        x, y_acc, y_cost, xlabel, main_keys, 
        left_ylim=None, show_grid=True, 
        save_path=None,
        ax=None, figsize=(4.3, 3.3),
        accs_map={ 
            'All Participants': "Average",  
            r'Participant $N$': "Task publisher", 
            r'Participants $1, \ldots, N-1$': "Best collaborator", 
            },
        metrics_map={ 
            'Prefilling phase': 'Prefill', 
            'Decoding phase': 'Decode'
            }
        ):
            
    model = main_keys["model"]
    prompt_seg = main_keys["split_way"]
    
    prompt_seg = prompt_seg_map[prompt_seg]
    
    """
    左轴：多颜色区分
    右轴：单一颜色的不同深浅 + 填充区域
    左右轴图例分开显示，带美化效果
    """
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.titleweight": "bold",
        "axes.grid": show_grid,
        "grid.alpha": 0.25,
        "font.size": 10,
        "figure.dpi": 150
    })
    
    # 左轴三条线的样式配置
    line_styles = {
        'All Participants': {'marker': 's', 'linestyle': '-', 'color': 'navy'},      # 深蓝
        r'Participant $N$': {'marker': 'o', 'linestyle': '--', 'color': 'red'},     # 红色
        r'Participants $1, \ldots, N-1$': {'marker': '^', 'linestyle': ':', 'color': 'orange'}  # 橙色
        }
        
    # 右轴颜色配置（绿色系的不同深浅）
    fill_colors = {
        'darkest': '#013220',     # 极深绿（几乎黑绿）
        'very_dark': '#1B5E20',   # 很深的绿
        'dark': '#2E7D32',        # 深绿（比你原来的更深）    
        'base': '#2E8B57',      # 深绿
        'medium': '#3CB371',    # 中绿  
        'light': '#90EE90',     # 浅绿
        'extra_light': '#F0FFF0' # 极浅绿（用于填充）
        }
    
    
    # 新增两根线的样式（绿色系+填充）
    fill_line_styles = {
        'dark': {
            'color': '#2E7D32',        # 深绿
            'linestyle': '-', 
            'linewidth': 2,            # 稍细的线
            'alpha': 0.8,              # 线条稍透明
            'fill_color': '#2E7D32',
            'fill_alpha': 0.5,
            'fill_hatch': '/////'        # 斜线填充
            },
        'light': {
            'color': '#388E3C',        # 中绿
            'linestyle': '-',
            'linewidth': 2,
            'alpha': 0.8,
            'fill_color': '#388E3C', 
            'fill_alpha': 0.25,
            'fill_hatch': '\\\\\\'        # 点状填充
            }
        }    

    if ax:
        ax1 = ax
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    fig = ax1.figure   # ✅ 保证 fig 一定可用

    # 左轴：Accuracy（多种颜色）
    acc_keys = list(accs_map.keys())  
    for key in acc_keys:
        style = line_styles[key]
        ax1.plot(
            x, y_acc[key]*100, 
            marker=style['marker'], 
            linestyle=style['linestyle'], 
            color=style['color'], 
            linewidth=2,
            markersize=8,
            markerfacecolor='none',
            markeredgecolor=style['color'],
            markeredgewidth=2
            )

    if len(list(accs_map.keys())) == 1:
        ax1.set_xlabel("Synchronization interval", fontsize=10)
        ax1.set_ylabel("EM Acc. (%)", color='navy', fontsize=10)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=9)
        ax1.tick_params(axis='y', labelcolor='navy', labelsize=9)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  
    else:
        ax1.set_xlabel("Synchronization interval", fontsize=10)
        ax1.set_ylabel("EM Acc. (%)", color='black', fontsize=10)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=9)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=9)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # 右轴：KV Communication（绿色系 + 填充区域）
    ax2 = ax1.twinx()
    
    # 绘制主线
    comm_lines, unit = human_bytes(y_cost['Prefilling phase']['Key-Value matrices'])
    ax2.plot(
        x, comm_lines, 
        color=fill_line_styles['dark']['color'],
        linestyle=fill_line_styles['dark']['linestyle'],
        linewidth=fill_line_styles['dark']['linewidth'],
        alpha=fill_line_styles['dark']['alpha'])
    
    ax2.fill_between(
        x, comm_lines,
        alpha=fill_line_styles['dark']['fill_alpha'],
        color=fill_line_styles['dark']['fill_color'],
        hatch=fill_line_styles['dark']['fill_hatch'],
        interpolate=True)
    
    ax2.set_ylabel(f"Comm. cost ({unit})", color=fill_colors['very_dark'], fontsize=10)
    ax2.tick_params(axis='y', labelcolor=fill_colors['very_dark'], labelsize=9)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    
    # 右轴倒置
    ax2.invert_yaxis()
    ax2.spines["right"].set_visible(True)
    
    # 设置xy轴显示范围
    # ax2.set_ylim(np.asarray(comm_lines).min()*0.95, np.asarray(comm_lines).max()*1.05)
    if max(x) == 24:
        ax1.set_xlim(min(x)-1.05, max(x)+1.05)  
    elif max(x) == 28:
        ax1.set_xlim(min(x)-1.18, max(x)+1.18)
    elif max(x) == 36:
        ax1.set_xlim(min(x)-1.5, max(x)+1.5)     
    elif max(x) == 8:
        ax1.set_xlim(min(x)-0.32, max(x)+0.32)            
    elif max(x) == 9:
        ax1.set_xlim(min(x)-0.35, max(x)+0.35)         
    
    
    
    # === 添加Y轴箭头 ===
    # 左轴：向上箭头
    ax1.annotate('', xy=(0, 0), xytext=(0, 1.15),
                 xycoords='axes fraction', textcoords='axes fraction', 
                 arrowprops=dict(arrowstyle='<|-', 
                                 color='black', 
                                 # lw=2
                                 )
                 )
    # 右轴：向下箭头
    ax2.annotate('', xy=(1, 1), xytext=(1, -0.1),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='<|-', 
                                 color='black', 
                                 # lw=2
                                 )
                 )

    if model == "Qwen2.5-7B":
    
        # 创建左轴图例handles
        left_handles = []
        for key in acc_keys:
            style = line_styles[key]
            handle = Line2D(
                [0], [0], 
                marker=style['marker'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=2,
                markersize=6,
                markerfacecolor='none',
                markeredgecolor=style['color'],
                markeredgewidth=2
                )
            left_handles.append(handle)
        
        # 创建右轴图例handle（填充区域样式）
        right_handle = patches.Patch(
            facecolor=fill_line_styles['dark']['fill_color'],  # 填充颜色
            alpha=fill_line_styles['dark']['fill_alpha'],      # 透明度
            hatch=fill_line_styles['dark']['fill_hatch'],      # 填充样式
            edgecolor=fill_line_styles['dark']['color'],       # 边框颜色
            linewidth=fill_line_styles['dark']['linewidth']    # 边框宽度
            )
        
        # === 美化左轴图例 ===
        acc_labels = list(accs_map.values())
        acc_labels_title = 'Response quality:'

        left_legend = ax1.legend(left_handles, acc_labels, 
                                 loc='upper left',
                                 title=acc_labels_title,
                                
                                 # 字体设置
                                 fontsize=9,
                                 title_fontsize=9,
                                
                                 # 布局设置
                                 ncol=3,                         # 1列垂直排列
                                 labelspacing=0.2,               # 条目间距
                                 handlelength=2,               # 图例标记长度
                                 handletextpad=0.6,              # 标记与文字间距
                                 columnspacing=1.5,              # 列间距
                                
                                 # 美化效果
                                 frameon=True,                   # 显示边框
                                 fancybox=True,                  # 圆角边框
                                 shadow=False,                    # 阴影效果
                                 framealpha=1,                # 背景透明度
                                 facecolor="None",         
                                 edgecolor='black',      
                                 borderpad=0.6,                               
                                
                                 # 其他美化
                                 markerscale=1.0,                # 标记缩放比例
                                 markerfirst=True,
                                 )               # 标记在前，文字在后
        
        # === 美化右轴图例 ===
        right_legend = ax2.legend([right_handle], ['Key-Value Exchange'], 
                                  loc='upper left',
                                  title='Communication Cost:',
                                 
                                  # 字体设置
                                  fontsize=9,
                                  title_fontsize=9,
        
                                  # 布局设置
                                  ncol=1,                         # 1列垂直排列
                                  labelspacing=0.2,               # 条目间距
                                  handlelength=2,               # 图例标记长度
                                  handletextpad=0.6,              # 标记与文字间距
                                  columnspacing=1.0,              # 列间距
                                    
                                  # 美化效果
                                  frameon=True,                   # 显示边框
                                  fancybox=True,                  # 圆角边框
                                  shadow=False,                    # 阴影效果
                                  framealpha=1,                # 背景透明度
                                  facecolor="None",          
                                  edgecolor=fill_colors['very_dark'],          
                                  borderpad=0.6,
                                  
                                  )
        
        left_legend._legend_box.align = "left"
        right_legend._legend_box.align = "left"
    
        left_legend.get_frame().set_linewidth(1)      # 边框粗细
        right_legend.get_frame().set_linewidth(1)     # 边框粗细  

        
        left_legend.set_bbox_to_anchor((0.28, 0.22), transform=fig.transFigure)
        left_legend.set_visible(True)
        fig.add_artist(left_legend)
        
        right_legend.set_bbox_to_anchor((0.595, 0.22), transform=fig.transFigure)
        right_legend.set_visible(True)
        fig.add_artist(right_legend)


    # 设置标题
    if ax:
        if model == "Qwen2.5-0.5B":
            ax1.set_title(prompt_seg, fontsize=10, pad=10)
        # ax1.set_title(model+'×'+prompt_seg, fontsize=10)
        return ax1, ax2  
    else:
        fig.suptitle(prompt_seg, fontsize=10, y=0.9)
        fig.tight_layout()
        plt.show()
        
        if save_path:
            save_path = os.path.join(save_path, "indiv")
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"comm_{model}_{main_keys["split_way"]}.pdf")
            fig.savefig(file_path, format="pdf", bbox_inches="tight")
            plt.close()              
            
            
            
            
            