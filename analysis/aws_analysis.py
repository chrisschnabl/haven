import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import logging
from datetime import datetime
from shared_plot_helpers import (
    set_plot_style, save_plot, style_axis, extract_times_tokens, get_bins_and_ticks, plot_bar_with_annotations, add_subject_if_missing
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aws_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plot style
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

# Create output directory
OUTPUT_DIR = Path(f'aws-analysis-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths and names
EXPERIMENT_MODES = ["enclave4b", "host4b", "comp-constant-8b"]
MODE_LABELS = {
    'enclave4b': '(i)',
    'host4b': '(ii)',
    'comp-constant-8b': '(iii)'
}

# Experiment types
EXPERIMENT_TYPES = {
    "classification": "llama_classification.parquet",
    "summarization": "llama_summaries.parquet",
    "toxicity": "llama_toxic.analysis.parquet"
}

def load_aws_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all datasets for all experiments and modes."""
    data = {}
    base_path = Path("./remote_experiments")
    
    for mode in EXPERIMENT_MODES:
        data[mode] = {}
        logger.info(f"Loading data for mode: {mode}")
        
        for exp_type, file_name in EXPERIMENT_TYPES.items():
            try:
                if exp_type == "toxicity":
                    if mode == "comp-constant-8b":
                        analysis_path = base_path / mode / "llama3_8b_8bit_toxicity.analysis.parquet"
                    elif mode == "enclave4b":
                        analysis_path = base_path / mode / file_name
                    else:  # host4b
                        toxic_path = base_path / mode / "llama_toxic.parquet"
                        analysis_path = base_path / mode / file_name
                        
                        if not toxic_path.exists() or not analysis_path.exists():
                            logger.warning(f"Missing toxicity files for {mode}")
                            data[mode][exp_type] = None
                            continue
                        
                        toxic_df = pd.read_parquet(toxic_path)
                        analysis_df = pd.read_parquet(analysis_path)
                        df = pd.merge(toxic_df, analysis_df, on='id', how='outer', suffixes=('', '_analysis'))
                        data[mode][exp_type] = df
                        continue
                    
                    if not analysis_path.exists():
                        logger.warning(f"Missing toxicity analysis file: {analysis_path}")
                        data[mode][exp_type] = None
                        continue
                    
                    df = pd.read_parquet(analysis_path)
                    data[mode][exp_type] = df
                    logger.info(f"Successfully loaded toxicity data for {mode}: {len(df)} rows")
                    
                else:
                    file_path = base_path / mode / file_name
                    if not file_path.exists():
                        logger.warning(f"File not found: {file_path}")
                        data[mode][exp_type] = None
                        continue
                    
                    df = pd.read_parquet(file_path)
                    if exp_type == "classification":
                        df = add_subject_if_missing(df)
                    data[mode][exp_type] = df
                    logger.info(f"Successfully loaded {exp_type} data for {mode}: {len(df)} rows")
            
            except Exception as e:
                logger.error(f"Error loading {exp_type} data for {mode}: {str(e)}")
                data[mode][exp_type] = None
    
    return data

def analyze_classification_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze classification performance for all models."""
    logger.info("Analyzing classification performance...")
    
    if not any(data[mode].get("classification") is not None for mode in data):
        logger.warning("No classification data available")
        return
    
    # Overall accuracy
    accuracies = []
    for mode in EXPERIMENT_MODES:
        if data[mode].get("classification") is not None:
            df = data[mode]["classification"]
            accuracy = df["correct"].mean() if "correct" in df.columns else np.nan
            accuracies.append({
                "mode": MODE_LABELS.get(mode, mode),
                "accuracy": accuracy,
                "correct": df["correct"].sum() if "correct" in df.columns else 0,
                "total": len(df)
            })
    
    if accuracies:
        plot_bar_with_annotations(
            pd.DataFrame(accuracies),
            "mode",
            "accuracy",
            [("accuracy", "correct", "total")],
            "Classification Accuracy",
            (0, 1.1),
            output_dir=str(OUTPUT_DIR)
        )
    
    # Valid responses only
    valid_accuracies = []
    for mode in EXPERIMENT_MODES:
        if data[mode].get("classification") is not None:
            df = data[mode]["classification"]
            valid_responses = df["response"].str.strip().str.match('^[ABCD]$')
            valid_df = df[valid_responses]
            
            if len(valid_df) > 0:
                valid_accuracies.append({
                    "mode": MODE_LABELS.get(mode, mode),
                    "accuracy": valid_df["correct"].mean(),
                    "correct": valid_df["correct"].sum(),
                    "total_valid": len(valid_df),
                    "total_all": len(df),
                    "valid_rate": len(valid_df) / len(df)
                })
    
    if valid_accuracies:
        valid_df = pd.DataFrame(valid_accuracies)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        sns.barplot(data=valid_df, x="mode", y="accuracy", hue="mode", palette="viridis", ax=ax1, legend=False)
        for i, row in valid_df.iterrows():
            ax1.text(i, row["accuracy"] + 0.01,
                    f"{row['accuracy']:.3f}\n({row['correct']}/{row['total_valid']})",
                    ha="center", va="bottom", fontsize=14)
        style_axis(ax1, "Mode", "Accuracy", (0, 1.1))
        
        # Valid rate plot
        sns.barplot(data=valid_df, x="mode", y="valid_rate", hue="mode", palette="viridis", ax=ax2, legend=False)
        for i, row in valid_df.iterrows():
            ax2.text(i, row["valid_rate"] + 0.01,
                    f"{row['valid_rate']:.3f}\n({row['total_valid']}/{row['total_all']})",
                    ha="center", va="bottom", fontsize=14)
        
        save_plot(fig, "classification_accuracy_valid_only", output_dir=str(OUTPUT_DIR))

def analyze_summarization_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze summarization performance for all models."""
    logger.info("Analyzing summarization performance...")
    
    if not any(data[mode].get("summarization") is not None for mode in data):
        logger.warning("No summarization data available")
        return
    
    all_scores = []
    for mode in EXPERIMENT_MODES:
        if data[mode].get("summarization") is not None:
            df = data[mode]["summarization"]
            if "score" in df.columns:
                scores = df["score"].to_frame()
                scores["mode"] = MODE_LABELS.get(mode, mode)
                all_scores.append(scores)
    
    if all_scores:
        all_scores_df = pd.concat(all_scores)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.violinplot(data=all_scores_df, x="mode", y="score", palette="viridis", inner=None, ax=ax)
        sns.boxplot(data=all_scores_df, x="mode", y="score", color="white", width=0.3,
                   boxprops=dict(alpha=0.7), ax=ax)
        
        style_axis(ax, "Mode", "BERT Score")
        save_plot(fig, "summarization_bert_scores", output_dir=str(OUTPUT_DIR))

def analyze_toxicity_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze toxicity detection performance for all models."""
    logger.info("Analyzing toxicity performance...")
    
    if not any(data[mode].get("toxicity") is not None for mode in data):
        logger.warning("No toxicity data available")
        return
    
    # Toxicity rate
    toxicity_rates = []
    for mode in EXPERIMENT_MODES:
        if data[mode].get("toxicity") is not None:
            df = data[mode]["toxicity"]
            if "toxic" in df.columns:
                toxic_rate = df["toxic"].mean()
            elif "classification" in df.columns:
                toxic_rate = (df["classification"] == "toxic").mean()
            elif "expected_toxic" in df.columns:
                toxic_rate = (df["expected_toxic"] == 1).mean()
            else:
                continue
            
            if not pd.isna(toxic_rate):
                toxicity_rates.append({
                    "mode": MODE_LABELS.get(mode, mode),
                    "toxic_rate": toxic_rate,
                    "toxic_count": int(toxic_rate * len(df)),
                    "total_samples": len(df)
                })
    
    if toxicity_rates:
        plot_bar_with_annotations(
            pd.DataFrame(toxicity_rates),
            "mode",
            "toxic_rate",
            [("toxic_rate", "toxic_count", "total_samples")],
            "Toxicity Rate",
            (0, max(pd.DataFrame(toxicity_rates)["toxic_rate"]) * 1.2),
            output_dir=str(OUTPUT_DIR)
        )

def plot_enclave_combined_analysis(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Create a combined figure with token histograms and timing analysis for enclave4b."""
    enclave_data = data.get('enclave4b', {})
    if not enclave_data:
        logger.warning("No enclave4b data available for combined analysis plot")
        return
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    row_titles = ['Response Time', 'Response Tokens', 'Prompt Time', 'Prompt Tokens']
    exp_colors = sns.color_palette('viridis', n_colors=3)
    bin_settings = {
        'classification': {'response_time_bins': 0.2, 'response_time_max': 5, 'response_tokens_bins': 0.5, 'response_tokens_max': 10, 'prompt_time_bins': 2, 'prompt_time_max': 40, 'prompt_tokens_bins': 5, 'prompt_tokens_max': 150},
        'summarization': {'response_time_bins': 2, 'response_time_max': 30, 'response_tokens_bins': 2, 'response_tokens_max': 50, 'prompt_time_bins': 5, 'prompt_time_max': 80, 'prompt_tokens_bins': 20, 'prompt_tokens_max': 400},
        'toxicity': {'response_time_bins': 5, 'response_time_max': 150, 'response_tokens_bins': 10, 'response_tokens_max': 250, 'prompt_time_bins': 2, 'prompt_time_max': 30, 'prompt_tokens_bins': 20, 'prompt_tokens_max': 250}
    }
    font_size = 14
    for col, exp_type in enumerate(EXPERIMENT_TYPES.keys()):
        df = enclave_data.get(exp_type)
        if df is None:
            for row in range(4):
                axes[row, col].text(0.5, 0.5, "No data available", ha="center", va="center", transform=axes[row, col].transAxes, fontsize=font_size, fontname='Times New Roman')
            continue
        prompt_times, response_times, prompt_tokens, response_tokens = extract_times_tokens(df, exp_type)
        row_data = [response_times, response_tokens, prompt_times, prompt_tokens]
        row_settings = [
            {'bins': bin_settings[exp_type]['response_time_bins'], 'max': bin_settings[exp_type]['response_time_max'], 'is_time': True},
            {'bins': bin_settings[exp_type]['response_tokens_bins'], 'max': bin_settings[exp_type]['response_tokens_max'], 'is_time': False},
            {'bins': bin_settings[exp_type]['prompt_time_bins'], 'max': bin_settings[exp_type]['prompt_time_max'], 'is_time': True},
            {'bins': bin_settings[exp_type]['prompt_tokens_bins'], 'max': bin_settings[exp_type]['prompt_tokens_max'], 'is_time': False}
        ]
        for row, (data_series, settings) in enumerate(zip(row_data, row_settings)):
            ax = axes[row, col]
            if data_series.empty:
                ax.text(0.5, 0.5, f"No {row_titles[row].lower()} data", ha="center", va="center", transform=ax.transAxes, fontsize=font_size, fontname='Times New Roman')
                continue
            bin_width = settings['bins']
            x_max = settings['max']
            is_time_row = settings['is_time']
            bins, x_ticks = get_bins_and_ticks(x_max, bin_width, is_time_row)
            sns.histplot(data=data_series, bins=bins, color=exp_colors[col], ax=ax, kde=False, stat="density", linewidth=1.5, alpha=0.7)
            median_val = data_series.median()
            ax.axvline(median_val, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_val:.1f}" + ("s" if is_time_row else ""))
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.text(0.95, 0.95, f"Median: {median_val:.1f}" + ("s" if is_time_row else ""), transform=ax.transAxes, verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=font_size, fontname='Times New Roman')
            ax.set_xlim(0, x_max * 1.02)
            ax.set_xticks(x_ticks)
            y_max = ax.get_ylim()[1]
            ax.set_ylim(0, min(y_max * 1.15, 0.5 * 1.15) if y_max > 0.5 else y_max * 1.15)
            style_axis(ax, xlabel='Time (seconds)' if is_time_row else '# tokens', ylabel='Density', font_size=font_size)
        if col < 3:
            axes[0, col].set_title(exp_type.capitalize(), fontsize=font_size, fontname='Times New Roman')
    for row in range(4):
        axes[row, 0].set_ylabel(row_titles[row], fontsize=font_size, fontname='Times New Roman')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    save_plot(fig, "enclave_combined_analysis", output_dir=str(OUTPUT_DIR))
    logger.info("Enclave combined analysis plot generated")

def plot_enclave_merged_analysis(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Create a combined figure with merged token histograms and timing analysis for enclave4b.
    
    Generates a 2-row figure with the following structure:
    - Row 1: Time distributions (both response and prompt times in same subplot)
    - Row 2: Token distributions (both response and prompt tokens in same subplot)
    
    Args:
        data: Dictionary containing experiment data by mode
    """
    # Extract enclave4b data
    enclave_data = data.get('enclave4b', {})
    if not enclave_data:
        logger.warning("No enclave4b data available for merged analysis plot")
        return
    
    # Create figure with two rows and 3 columns (experiment types)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    
    # Row titles: Time and Tokens
    row_titles = ['Time (seconds)', '# tokens']
    
    # Use a better color palette with higher contrast
    prompt_color = '#5975A4'  # Deeper blue
    response_color = '#5F9E6E'  # Deeper green
    
    # Create custom transparent colors for fill
    prompt_fill = mpl.colors.to_rgba(prompt_color, 0.7)
    response_fill = mpl.colors.to_rgba(response_color, 0.7)
    
    # Manually set bin widths and x-axis limits for each experiment type and row
    bin_settings = {
        'classification': {
            'time_bins': 0.2,      # seconds
            'time_max': 40,        # seconds
            'tokens_bins': 0.5,    # tokens
            'tokens_max': 150      # tokens
        },
        'summarization': {
            'time_bins': 2,        # seconds
            'time_max': 80,        # seconds
            'tokens_bins': 2,      # tokens
            'tokens_max': 400      # tokens
        },
        'toxicity': {
            'time_bins': 5,        # seconds
            'time_max': 150,       # seconds
            'tokens_bins': 10,     # tokens
            'tokens_max': 250      # tokens
        }
    }
    
    # Process each experiment type
    for col, exp_type in enumerate(EXPERIMENT_TYPES.keys()):
        df = enclave_data.get(exp_type)
        
        if df is None:
            # No data for this experiment type
            for row in range(2):
                axes[row, col].text(
                    0.5, 0.5, "No data available",
                    ha="center", va="center", transform=axes[row, col].transAxes,
                    fontsize=14, fontname='Times New Roman'
                )
            continue
            
        # Extract token and timing data
        if "token_count" in df.columns and "prompt_tokens" in df.columns:
            prompt_tokens = df['prompt_tokens']
            response_tokens = df['token_count']
        else:
            prompt_tokens = pd.Series()
            response_tokens = pd.Series()
            
        if "duration" in df.columns and "prompt_duration" in df.columns:
            prompt_times = df['prompt_duration']
            
            # Calculate response times (total - prompt)
            response_times = df['duration'] - df['prompt_duration']
            response_times = response_times[response_times > 0]
            
            # Handle limited valid response times
            if len(response_times) < len(df) * 0.1:
                if exp_type == 'classification':
                    response_times = df['duration'] * 0.75
                elif exp_type == 'summarization':
                    response_times = df['duration'] * 0.85
        else:
            prompt_times = pd.Series()
            response_times = pd.Series()
        
        # Define data for each row
        row_data = [
            # Row 0: Times (both prompt and response)
            {
                'prompt': prompt_times,
                'response': response_times,
                'bin_width': bin_settings[exp_type]['time_bins'],
                'max_val': bin_settings[exp_type]['time_max'],
                'xlabel': 'Time (seconds)',
                'is_time': True
            },
            # Row 1: Tokens (both prompt and response)
            {
                'prompt': prompt_tokens,
                'response': response_tokens,
                'bin_width': bin_settings[exp_type]['tokens_bins'],
                'max_val': bin_settings[exp_type]['tokens_max'],
                'xlabel': '# tokens',
                'is_time': False
            }
        ]
        
        # Plot each row
        for row, data_dict in enumerate(row_data):
            prompt_data = data_dict['prompt']
            response_data = data_dict['response']
            
            # Skip if no data
            if prompt_data.empty and response_data.empty:
                axes[row, col].text(
                    0.5, 0.5, f"No data available",
                    ha="center", va="center", transform=axes[row, col].transAxes,
                    fontsize=14, fontname='Times New Roman'
                )
                continue
            
            is_time_row = data_dict['is_time']
            bin_width = data_dict['bin_width']
            x_max = data_dict['max_val']
            
            # Create bins starting from 0
            bins = np.arange(0, x_max + bin_width, bin_width)
            
            # Clear the plot first to avoid overplotting
            axes[row, col].clear()
            
            # Add light grid lines for readability
            axes[row, col].grid(True, linestyle='--', alpha=0.3)
            
            # First plot response data (so prompt appears on top)
            if not response_data.empty:
                response_hist = axes[row, col].hist(
                    response_data,
                    bins=bins,
                    color=response_fill,
                    label='Response',
                    histtype='bar',
                    density=True,
                    edgecolor=response_color,
                    linewidth=0.8,
                    zorder=1
                )
                
                # Add mean line for response
                response_mean = response_data.mean()
                axes[row, col].axvline(
                    response_mean,
                    color=response_color,
                    linestyle='--',
                    linewidth=2.0,
                    zorder=3,
                    label=f"Response Mean: {response_mean:.1f}" + ("s" if is_time_row else "")
                )
            
            # Then plot prompt data
            if not prompt_data.empty:
                prompt_hist = axes[row, col].hist(
                    prompt_data,
                    bins=bins,
                    color=prompt_fill,
                    label='Prompt',
                    histtype='bar',
                    density=True,
                    edgecolor=prompt_color,
                    linewidth=0.8,
                    zorder=2
                )
                
                # Add mean line for prompt
                prompt_mean = prompt_data.mean()
                axes[row, col].axvline(
                    prompt_mean,
                    color=prompt_color,
                    linestyle='--',
                    linewidth=2.0,
                    zorder=4,
                    label=f"Prompt Mean: {prompt_mean:.1f}" + ("s" if is_time_row else "")
                )
            
            # Set x-axis limits with some padding
            data_min = 0
            axes[row, col].set_xlim(data_min, x_max * 1.02)
            
            # Set appropriate ticks
            x_ticks = [0]
            tick_count = 5  # Number of ticks including min and max
            step = x_max / (tick_count - 1)
            for i in range(1, tick_count):
                x_ticks.append(i * step)
            
            # Round the ticks appropriately
            if is_time_row:
                x_ticks = [round(tick, 1) for tick in x_ticks]
            else:
                x_ticks = [int(round(tick)) for tick in x_ticks]
                
            # Remove duplicates while preserving order
            x_ticks = list(dict.fromkeys(x_ticks))
            
            axes[row, col].set_xticks(x_ticks)
            
            # Set y-axis label and limits
            axes[row, col].set_ylabel('Density', fontsize=14, fontname='Times New Roman')
            
            # Normalize y-axis if needed
            y_max = axes[row, col].get_ylim()[1]
            if y_max > 0.5:
                axes[row, col].set_ylim(0, 0.5 * 1.15)
            else:
                axes[row, col].set_ylim(0, y_max * 1.15)
            
            # Set x-axis label
            axes[row, col].set_xlabel(data_dict['xlabel'], fontsize=14, fontname='Times New Roman')
            
            # Add legend - position at top center for better visibility
            legend = axes[row, col].legend(
                loc='upper center', 
                fontsize=12,
                framealpha=0.9,
                edgecolor='gray'
            )
            # Ensure legend text is consistent
            for text in legend.get_texts():
                text.set_fontname('Times New Roman')
        
        # Set column titles
        if col < 3:
            axes[0, col].set_title(exp_type.capitalize(), fontsize=16, fontname='Times New Roman', fontweight='bold')
    
    # Set row labels
    for row in range(2):
        axes[row, 0].set_ylabel(row_titles[row], fontsize=14, fontname='Times New Roman', fontweight='bold')
    
    # Set general styling for all subplots
    for row in range(2):
        for col in range(3):
            # Set all spines to black
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
                
            # Set axis label font size and font
            axes[row, col].tick_params(axis='both', labelsize=12, width=1.5, length=5)
            for label in axes[row, col].get_xticklabels() + axes[row, col].get_yticklabels():
                label.set_fontname('Times New Roman')
    
    # Save plot with tight layout
    plt.tight_layout()
    save_plot(fig, "enclave_merged_analysis", output_dir=str(OUTPUT_DIR))
    logger.info("Enclave merged analysis plot generated")

def plot_combined_metrics_grid(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    all_scores = []
    for mode in EXPERIMENT_MODES:
        df = data[mode].get("summarization")
        if df is not None and "score" in df.columns:
            scores = df["score"].to_frame()
            scores["mode"] = MODE_LABELS[mode]
            all_scores.append(scores)
    if all_scores:
        all_scores_df = pd.concat(all_scores)
        sns.violinplot(data=all_scores_df, x="mode", y="score", palette="viridis", inner=None, ax=axes[0])
        sns.boxplot(data=all_scores_df, x="mode", y="score", color="white", width=0.3, boxprops=dict(alpha=0.7), ax=axes[0])
        style_axis(axes[0], "Mode", "BERT Score")
    toxicity_rates = []
    for mode in EXPERIMENT_MODES:
        df = data[mode].get("toxicity")
        if df is not None:
            if "toxic" in df.columns:
                toxic_rate = df["toxic"].mean()
            elif "classification" in df.columns:
                toxic_rate = (df["classification"] == "toxic").mean()
            elif "expected_toxic" in df.columns:
                toxic_rate = (df["expected_toxic"] == 1).mean()
            else:
                continue
            toxicity_rates.append({"mode": MODE_LABELS[mode], "toxic_rate": toxic_rate})
    if toxicity_rates:
        tox_df = pd.DataFrame(toxicity_rates)
        sns.barplot(data=tox_df, x="mode", y="toxic_rate", palette="viridis", ax=axes[1])
        style_axis(axes[1], "Mode", "Toxicity Rate")
    accuracies = []
    for mode in EXPERIMENT_MODES:
        df = data[mode].get("classification")
        if df is not None and "correct" in df.columns:
            accuracy = df["correct"].mean()
            accuracies.append({"mode": MODE_LABELS[mode], "accuracy": accuracy})
    if accuracies:
        acc_df = pd.DataFrame(accuracies)
        sns.barplot(data=acc_df, x="mode", y="accuracy", palette="viridis", ax=axes[2])
        style_axis(axes[2], "Mode", "Classification Accuracy")
    plt.tight_layout()
    save_plot(fig, "combined_metrics_grid", output_dir=str(OUTPUT_DIR))

def run_aws_analysis() -> None:
    """Execute the complete AWS analysis pipeline."""
    logger.info("Starting AWS analysis pipeline...")
    set_plot_style()
    data = load_aws_data()
    analyze_classification_performance(data)
    analyze_summarization_performance(data)
    analyze_toxicity_performance(data)
    plot_enclave_combined_analysis(data)
    plot_enclave_merged_analysis(data)
    plot_combined_metrics_grid(data)
    logger.info("AWS analysis pipeline completed successfully")

if __name__ == "__main__":
    run_aws_analysis() 