import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

def set_plot_style() -> None:
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.format': 'pdf',
        'pdf.fonttype': 3,
        'ps.fonttype': 3,
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif']
    })

def save_plot(fig: plt.Figure, filename: str, output_dir: Optional[str] = None) -> None:
    plt.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/{filename}.pdf", dpi=300)
    else:
        fig.savefig(f"{filename}.pdf", dpi=300)
    plt.close(fig)


def style_axis(ax: plt.Axes, xlabel: str = '', ylabel: str = '', ylim: Optional[Tuple[float, float]] = None, font_size: int = 14) -> None:
    ax.set_xlabel(xlabel, fontsize=font_size, fontname='Times New Roman')
    ax.set_ylabel(ylabel, fontsize=font_size, fontname='Times New Roman')
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=font_size, width=1.5, length=5)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)


def extract_times_tokens(df: pd.DataFrame, exp_type: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if "token_count" in df.columns and "prompt_tokens" in df.columns:
        prompt_tokens = df['prompt_tokens']
        response_tokens = df['token_count']
    else:
        prompt_tokens = pd.Series(dtype=float)
        response_tokens = pd.Series(dtype=float)
    if "duration" in df.columns and "prompt_duration" in df.columns:
        prompt_times = df['prompt_duration']
        response_times = df['duration'] - df['prompt_duration']
        response_times = response_times[response_times > 0]
        if len(response_times) < len(df) * 0.1:
            if exp_type == 'classification':
                response_times = df['duration'] * 0.75
            elif exp_type == 'summarization':
                response_times = df['duration'] * 0.85
    else:
        prompt_times = pd.Series(dtype=float)
        response_times = pd.Series(dtype=float)
    return prompt_times, response_times, prompt_tokens, response_tokens


def get_bins_and_ticks(x_max: float, bin_width: float, is_time: bool) -> Tuple[np.ndarray, List[float]]:
    bins = np.arange(0, x_max + bin_width, bin_width)
    x_ticks = [0]
    tick_count = 5
    step = x_max / (tick_count - 1)
    for i in range(1, tick_count):
        x_ticks.append(i * step)
    if is_time:
        x_ticks = [round(tick, 1) for tick in x_ticks]
    else:
        x_ticks = [int(round(tick)) for tick in x_ticks]
    x_ticks = list(dict.fromkeys(x_ticks))
    return bins, x_ticks


def plot_bar_with_annotations(
    data: pd.DataFrame,
    x: str,
    y: str,
    annotations: List[Tuple[str, str, str]],
    title: str,
    ylim: Optional[Tuple[float, float]] = None,
    output_dir: Optional[str] = None
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x=x, y=y, hue=x, palette="viridis", ax=ax, legend=False)
    for i, (value_col, count_col, total_col) in enumerate(annotations):
        for j, row in data.iterrows():
            ax.text(
                j, 
                row[value_col] + 0.01,
                f"{row[value_col]:.3f}\n({row[count_col]}/{row[total_col]})",
                ha="center",
                va="bottom",
                fontsize=14
            )
    style_axis(ax, x, title, ylim)
    save_plot(fig, title.lower().replace(" ", "_"), output_dir)


def add_subject_if_missing(df: pd.DataFrame, input_data_path: str = "./input_datasets/classification_pairs.parquet") -> pd.DataFrame:
    if 'subject' not in df.columns:
        try:
            input_data = pd.read_parquet(input_data_path)
            question_to_subject = dict(zip(input_data['question'], input_data['subject']))
            df['subject'] = df['question'].map(question_to_subject)
        except Exception as e:
            df['subject'] = 'unknown'
    return df 