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
        x = np.min(n)
        for u in units:
            # 当数值小于 1024 或者已经到最后一个单位时，输出并返回
            if x < 1000 or u == units[-1]:
                break
            x /= 1000  # 否则除以 1024 继续转换到更大单位
        return n/(1000**(units.index(u))), u, 1/(1000**(units.index(u)))
    
    elif isinstance(n, numbers.Number):  # 包括 int, float, complex
        x = float(n)
        for u in units:
            # 当数值小于 1024 或者已经到最后一个单位时，输出并返回
            if x < 1000 or u == units[-1]:
                return x, u, 1/(1000**(units.index(u)))
            x /= 1000  # 否则除以 1024 继续转换到更大单位



def human_flops(n):
    """
    将 FLOPs 数量格式化成人类可读的字符串。

    参数:
        n: 原始 FLOPs 数量（int）。

    返回:
        str，例如 "1.234 GFLOPs"
    """       
    units = ["FLOPs",
             "FLOPs\n" + r'$\left(\times 10^1\right)$',
             "FLOPs\n" + r'$\left(\times 10^2\right)$',
             "FLOPs\n" + r'$\left(\times 10^3\right)$',
             "FLOPs\n" + r'$\left(\times 10^4\right)$',
             "FLOPs\n" + r'$\left(\times 10^5\right)$',
             "FLOPs\n" + r'$\left(\times 10^6\right)$',
             "FLOPs\n" + r'$\left(\times 10^7\right)$',
             "FLOPs\n" + r'$\left(\times 10^8\right)$',
             "FLOPs\n" + r'$\left(\times 10^9\right)$',
             "FLOPs\n" + r'$\left(\times 10^{10}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{11}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{12}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{13}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{14}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{15}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{16}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{17}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{18}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{19}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{20}\right)$',    
             "FLOPs\n" + r'$\left(\times 10^{21}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{22}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{23}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{24}\right)$',
             "FLOPs\n" + r'$\left(\times 10^{25}\right)$',               
             ]
    if isinstance(n, np.ndarray):
        x = float(np.max(n))
        for u in units:
            if x < 100 or u == units[-1]:
                # 当数值小于 1000 或者已经到最后一个单位时，输出并返回
                break                
            x /= 10  # 否则除以 1000 继续转换到更大单位    
        return n/(10**(units.index(u))), u, 1/(10**(units.index(u)))           
            
    elif isinstance(n, numbers.Number):  # 包括 int, float, complex           
        x = float(n)
        for u in units:
            if x < 100 or u == units[-1]:
                # 当数值小于 1000 或者已经到最后一个单位时，输出并返回
                return x, u, 1/(10**(units.index(u)))
            x /= 10  # 否则除以 1000 继续转换到更大单位


prompt_seg_map = {
    "even": "TokAg",           # 按token数量平均分10份
    "even_question_last": "TokEx",            # 最后一份是问题，剩下按token平均分
    "smart": "SemAg",        # 平均分完整instruction，问题给最后一个
    "smart_question_last": "SemEx"          # 最后一个只有问题，剩下平均分instruction
}

def plot_main_fig_comp_mem(
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
        'Prefilling phase': {
            'color': '#2E7D32',        # 深绿
            'linestyle': '-', 
            'linewidth': 1.5,            # 稍细的线
            'alpha': 0.8,              # 线条稍透明
            'fill_color': 'none',
            'fill_alpha': 0.5,
            'fill_hatch': '/////',        # 斜线填充
            },
        'Decoding phase': {
            'color': '#388E3C',        # 中绿
            'linestyle': '--',
            'linewidth': 1.5,
            'alpha': 0.8,
            'fill_color': 'none', 
            'fill_alpha': 0.5,
            'fill_hatch': '\\\\\\',        # 点状填充
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
        ax1.set_xlabel(xlabel, fontsize=9)
        ax1.set_ylabel("EM Acc. (%)", color='navy', fontsize=9)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=9)
        ax1.tick_params(axis='y', labelcolor='navy', labelsize=9)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  
    else:
        ax1.set_xlabel(xlabel, fontsize=9)
        ax1.set_ylabel("EM Acc. (%)", color='black', fontsize=9)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=9)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=9)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # 右轴：KV Communication（绿色系 + 填充区域）
    # 在 ax1 的上半部分创建右轴
    ax2 = ax1.inset_axes([0.0, 0.5, 1.0, 0.5], transform=ax1.transAxes)
    ax2.set_facecolor('none')   # 背景透明


    
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    for sp in ["left", "bottom", "top"]:
        ax2.spines[sp].set_visible(False)

    
    # 绘制主线
    FLOPs_lines = {'Prefilling phase': None, 'Decoding phase': None}
    FLOPs_lines['Prefilling phase'], unit, scale_para_FLOPs = human_flops(y_cost['Prefilling phase']['FLOPs'])
    FLOPs_lines['Decoding phase'] = y_cost['Decoding phase']['FLOPs']*scale_para_FLOPs
    
    metric_keys = list(metrics_map.keys())  
    for key in metric_keys:
        style = fill_line_styles[key]

        ax2.plot(x, FLOPs_lines[key], 
                 color=style['color'],
                 linestyle=style['linestyle'],
                 linewidth=style['linewidth'],
                 alpha=style['alpha'])
        
        ax2.fill_between(x, FLOPs_lines[key],
                         alpha=style['fill_alpha'],
                         color=style['fill_color'],
                         hatch=style['fill_hatch'],
                         facecolor=(style['color'] if style['fill_color'] != 'none' else 'none'),
                         edgecolor=style['color'],       # hatch 用这个颜色
                         interpolate=True)


    ax2.set_ylabel(unit, color=fill_colors['very_dark'], fontsize=9)
    ax2.tick_params(axis='y', labelcolor=fill_colors['very_dark'], labelsize=9)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    ax2.tick_params(axis='x',      # 只对 x 轴设置
                bottom=False,  # 关掉底部刻度
                top=False,     # 关掉顶部刻度
                labelbottom=False,  # 关掉底部标签
                labeltop=False)     # 关掉顶部标签

    # 右轴倒置
    ax2.invert_yaxis()
    
    ax2.spines["right"].set_visible(True)

    x_pos, y_pos = ax2.yaxis.label.get_position()
    ax2.yaxis.set_label_coords(1.13, y_pos+0.15)  # (x, y)，相对于坐标轴范围
 
    for label in ax2.get_yticklabels():
        label.set_rotation(90)   # 旋转 90°
        label.set_va('center')   # 垂直居中对齐
        label.set_ha('left')    # 右对齐，让数字紧贴坐标轴    
 
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
    ax2.annotate('', xy=(1, 1), xytext=(1, -0.08),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='<|-', 
                                 color=fill_colors['very_dark'], 
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
        right_handles = []
        for key in metric_keys:
            style = fill_line_styles[key]
            right_handle = patches.Patch(
                facecolor=(style['fill_color'] if style['fill_color'] != 'none' else 'none'),  # 填充颜色
                alpha=style['fill_alpha'],      # 透明度
                hatch=style['fill_hatch'],      # 填充样式
                edgecolor=style['color'],  # 边框颜色+透明度
                linewidth=style['linewidth'],    # 边框宽度
                linestyle=style['linestyle']       # 边框线型            
                
            )     
            right_handles.append(right_handle)        
        
        
        # === 美化的左轴图例 ===
               
        acc_labels = list(accs_map.values())
        acc_labels_title = 'Response Quality:'
        metric_labels = list(metrics_map.values())
        metric_labels_title = 'Floating Point Operations:'
      
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
        
        # === 美化的右轴图例 ===
        right_legend = ax2.legend(right_handles, metric_labels, 
                                  loc='upper left',
                                  title=metric_labels_title,
                                 
                                  # 字体设置
                                  fontsize=9,
                                  title_fontsize=9,
        
                                  # 布局设置
                                  ncol=2,                         # 1列垂直排列
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
                                  edgecolor="black",          
                                  borderpad=0.6,
                                 
                                  )
        
        left_legend._legend_box.align = "left"
        right_legend._legend_box.align = "left"
    
        left_legend.get_frame().set_linewidth(1)      # 边框粗细
        right_legend.get_frame().set_linewidth(1)     # 边框粗细  
    




    # 新增两根线的样式（绿色系+填充）
    fill_line_styles = {
        'Prefilling phase': {
            'color': 'pink',        # 深绿
            'linestyle': '-', 
            'linewidth': 1.5,            # 稍细的线
            'alpha': 0.8,              # 线条稍透明
            'fill_color': 'none',
            'fill_alpha': 0.5,
            'fill_hatch': '/////',        # 斜线填充
            'fill': False
            },
        'Decoding phase': {
            'color': 'lightpink',        # 中绿
            'linestyle': '--',
            'linewidth': 1.5,
            'alpha': 0.8,
            'fill_color': 'none', 
            'fill_alpha': 0.5,
            'fill_hatch': '\\\\\\',        # 点状填充
            'fill': True
            }
        }    



    # 在 ax1 的上半部分创建右轴
    ax3 = ax1.inset_axes([0.0, 0.0, 1.0, 0.5], transform=ax1.transAxes)
    ax3.set_facecolor('none')   # 背景透明
    
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    for sp in ["left", "bottom", "top"]:
        ax3.spines[sp].set_visible(False)

    
    # 绘制主线
    mem_lines = {'Prefilling phase': None, 'Decoding phase': None}
    mem_lines['Prefilling phase'], unit, scale_para = human_bytes(y_cost['Prefilling phase']["Memory usage"])
    mem_lines['Decoding phase'] = y_cost['Decoding phase']["Memory usage"]*scale_para


    metric_keys = list(metrics_map.keys()) 
    for key in metric_keys:
        style = fill_line_styles[key]

        ax3.plot(
            x, mem_lines[key], 
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth'],
            alpha=style['alpha'],
            
            )
        
        ax3.fill_between(
            x, mem_lines[key],
            alpha=style['fill_alpha'],
            color=style['fill_color'],
            hatch=style['fill_hatch'],
            facecolor=(style['color'] if style['fill_color'] != 'none' else 'none'),
            edgecolor=style['color'],       # hatch 用这个颜色
            interpolate=True
            )

    
    ax3.set_ylabel(f"Peak Mem. \n({unit})", color="#E75480", fontsize=9)
    ax3.tick_params(axis='y', labelcolor="#E75480", labelsize=9)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    ax3.tick_params(axis='x',      # 只对 x 轴设置
                bottom=False,  # 关掉底部刻度
                top=False,     # 关掉顶部刻度
                labelbottom=False,  # 关掉底部标签
                labeltop=False)     # 关掉顶部标签    
    # 右轴倒置
    ax3.invert_yaxis()
    ax3.spines["right"].set_visible(True)
    
    x_pos, y_pos = ax3.yaxis.label.get_position()
    ax3.yaxis.set_label_coords(1.13, y_pos-0.15)  # (x, y)，相对于坐标轴范围
 
    for label in ax3.get_yticklabels():
        label.set_rotation(90)   # 旋转 90°
        label.set_va('center')   # 垂直居中对齐
        label.set_ha('left')    # 右对齐，让数字紧贴坐标轴 
    
    
    # 右轴：向下箭头
    ax3.annotate('', xy=(1, 1), xytext=(1, -0.08),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='<|-', 
                                 color="#E75480", 
                                 # lw=2
                                 )
                 )

    
    if model == "Qwen2.5-7B":
                
        # 创建右轴图例handle（填充区域样式）   
        right_handles_appendix = []
        for key in metric_keys:
            style = fill_line_styles[key]
            right_handle = patches.Patch(
                facecolor=(style['fill_color'] if style['fill'] else 'none'),
                alpha=style['fill_alpha'],      # 透明度
                hatch=style['fill_hatch'],      # 填充样式
                edgecolor=style['color'],       # 边框颜色
                linewidth=style['linewidth'],    # 边框宽度
                linestyle=style['linestyle']       # 边框线型  
            )     
            right_handles_appendix.append(right_handle)        
        
        

        metric_labels = list(metrics_map.values())
        metric_labels_title = 'Peak Memory Usage:'
    
        
        # === 美化的右轴图例 ===
        right_legend_appendix = ax3.legend(right_handles_appendix, metric_labels, 
                                  loc='upper left',
                                  title=metric_labels_title,
                                 
                                  # 字体设置
                                  fontsize=9,
                                  title_fontsize=9,
        
                                  # 布局设置
                                  ncol=2,                         # 1列垂直排列
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
                                  edgecolor="black",          
                                  borderpad=0.6
    
                                  
                                  )
        
    
        right_legend_appendix._legend_box.align = "left"
        right_legend_appendix.get_frame().set_linewidth(1)     # 边框粗细  
    
        
        left_legend.set_bbox_to_anchor((0.22, 0.22), transform=fig.transFigure)
        left_legend.set_visible(True)
        fig.add_artist(left_legend)
        
        right_legend.set_bbox_to_anchor((0.535, 0.22), transform=fig.transFigure)
        right_legend.set_visible(True)
        fig.add_artist(right_legend)
        
        right_legend_appendix.set_bbox_to_anchor((0.69, 0.22), transform=fig.transFigure)
        right_legend_appendix.set_visible(True)
        fig.add_artist(right_legend_appendix)



    # # 确保两个图例都显示
    # ax1.add_artist(left_legend)
    
    # 设置标题
    if ax:
        if model == "Qwen2.5-0.5B":
            ax1.set_title(prompt_seg, fontsize=10, pad=10)
        
        
        # ax1.set_title(model+'×'+prompt_seg, fontsize=10)
        return ax1, ax2, ax3  
    else:
        fig.suptitle(prompt_seg, fontsize=10, y=0.9)
        fig.tight_layout()
        plt.show()
        
        if save_path:
            save_path = os.path.join(save_path, "indiv")
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"comp_{model}_{main_keys["split_way"]}.pdf")
            fig.savefig(file_path, format="pdf", bbox_inches="tight")
            plt.close()  
            





























