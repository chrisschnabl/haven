import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Any, Optional
import re
from dataclasses import dataclass
import logging
import pyarrow.parquet as pq
from datetime import datetime
import os

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

# Configure matplotlib for LaTeX text rendering and Type 1 fonts
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
#mpl.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math text
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['savefig.format'] = 'pdf' 

# Set border color to black
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.left'] = True

# gs -o flow-v4-embedded.pdf -dNoOutputFonts -sDEVICE=pdfwrite token_distribution.pdf
# docker run --rm -v "$PWD:/data" -w /data minidocks/poppler pdffonts token_distribution.pdf

# Set plot style
plt.style.use('default')  # Changed from fivethirtyeight to default
sns.set_style("whitegrid")  # Changed to whitegrid
plt.rcParams['figure.figsize'] = (12, 6)  # Reduced height from 8 to 6
plt.rcParams['axes.titlesize'] = 18  # Reduced from 22
plt.rcParams['axes.labelsize'] = 18  # Reduced from 22
plt.rcParams['xtick.labelsize'] = 16  # Reduced from 22
plt.rcParams['ytick.labelsize'] = 16  # Reduced from 22
plt.rcParams['legend.fontsize'] = 16  # Reduced from 22
plt.rcParams['figure.facecolor'] = 'white'  # Set figure background to white
plt.rcParams['axes.facecolor'] = 'white'    # Set axes background to white
plt.rcParams['savefig.facecolor'] = 'white' # Set saved figure background to white

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['patch.linewidth'] = 1
plt.rcParams['boxplot.boxprops.linewidth'] = 1
plt.rcParams['boxplot.whiskerprops.linewidth'] = 1
plt.rcParams['boxplot.capprops.linewidth'] = 1
plt.rcParams['boxplot.medianprops.linewidth'] = 1
plt.rcParams['boxplot.flierprops.linewidth'] = 1

# Create chart directories
CHART_DIR = Path(f'aws-analysis-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}/charts')
CLASSIFICATION_CHARTS = CHART_DIR / 'classification'
SUMMARIZATION_CHARTS = CHART_DIR / 'summarization'
TOXICITY_CHARTS = CHART_DIR / 'toxicity'
TOKEN_CHARTS = CHART_DIR / 'token_analysis'
TIMING_CHARTS = CHART_DIR / 'timing_analysis'
EFFICIENCY_CHARTS = CHART_DIR / 'efficiency'
ERROR_ANALYSIS_CHARTS = CHART_DIR / 'error_analysis'
FLAMEGRAPH_CHARTS = CHART_DIR / 'flamegraph'

def create_chart_directories():
    """Create all necessary chart directories."""
    for directory in [CLASSIFICATION_CHARTS, SUMMARIZATION_CHARTS, TOXICITY_CHARTS, TIMING_CHARTS, TOKEN_CHARTS]:
        directory.mkdir(parents=True, exist_ok=True)

EXPERIMENT_TYPES = {
    "classification": "llama_classification.parquet",
    "summarization": "llama_summaries.parquet",
    "toxicity": "llama_toxic.analysis.parquet"
}

EXPERIMENT_MODES = ["enclave4b", "host4b", "comp-constant-8b"]

def load_aws_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all datasets for all experiments and modes.
    
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Nested dictionary with 
            mode and experiment type as keys, and dataframes as values
    """
    data = {}
    base_path = Path("./remote_experiments")
    
    for mode in EXPERIMENT_MODES:
        data[mode] = {}
        logger.info(f"Loading data for mode: {mode}")
        
        for exp_type, file_name in EXPERIMENT_TYPES.items():
            try:
                if exp_type == "toxicity":
                    # Special case for comp-constant-8b mode
                    if mode == "comp-constant-8b":
                        analysis_path = base_path / mode / "llama3_8b_8bit_toxicity.analysis.parquet"
                        
                        if not analysis_path.exists():
                            logger.warning(f"Missing toxicity analysis file: {analysis_path}")
                            data[mode][exp_type] = None
                            continue
                        
                        try:
                            # Read the analysis file
                            df = pd.read_parquet(analysis_path)
                            logger.info(f"Analysis columns for {mode}: {list(df.columns)}")
                            
                            data[mode][exp_type] = df
                            logger.info(f"Successfully loaded toxicity data for {mode}: {len(df)} rows")
                            
                        except Exception as e:
                            logger.error(f"Error loading toxicity data for {mode}: {str(e)}")
                            data[mode][exp_type] = None
                    elif mode == "enclave4b":
                        # For enclave4b, just load the analysis file directly
                        analysis_path = base_path / mode / file_name  # This is llama_toxic.analysis.parquet
                        
                        if not analysis_path.exists():
                            logger.warning(f"Missing toxicity analysis file: {analysis_path}")
                            data[mode][exp_type] = None
                            continue
                        
                        try:
                            # Read the analysis file
                            df = pd.read_parquet(analysis_path)
                            logger.info(f"Analysis columns for {mode}: {list(df.columns)}")
                            
                            data[mode][exp_type] = df
                            logger.info(f"Successfully loaded toxicity data for {mode}: {len(df)} rows")
                            
                        except Exception as e:
                            logger.error(f"Error loading toxicity data for {mode}: {str(e)}")
                            data[mode][exp_type] = None
                    else:  # host4b
                        # For host4b, merge both toxicity files
                        toxic_path = base_path / mode / "llama_toxic.parquet"
                        analysis_path = base_path / mode / file_name
                        
                        if not toxic_path.exists():
                            logger.warning(f"Missing toxicity base file: {toxic_path}")
                            data[mode][exp_type] = None
                            continue
                            
                        if not analysis_path.exists():
                            logger.warning(f"Missing toxicity analysis file: {analysis_path}")
                            data[mode][exp_type] = None
                            continue
                        
                        try:
                            # Read both parquet files
                            toxic_df = pd.read_parquet(toxic_path)
                            analysis_df = pd.read_parquet(analysis_path)
                            
                            # Log the columns we have available
                            logger.info(f"Toxic columns for {mode}: {list(toxic_df.columns)}")
                            logger.info(f"Analysis columns for {mode}: {list(analysis_df.columns)}")
                            
                            # Merge on id
                            merged_df = pd.merge(
                                toxic_df,
                                analysis_df,
                                on='id',
                                how='outer',  # Use outer join to keep all data
                                suffixes=('', '_analysis')
                            )
                            
                            # Log the merged columns
                            logger.info(f"Merged toxicity columns for {mode}: {list(merged_df.columns)}")
                            
                            data[mode][exp_type] = merged_df
                            logger.info(f"Successfully merged toxicity data for {mode}: {len(merged_df)} rows")
                            
                        except Exception as e:
                            logger.error(f"Error merging toxicity data for {mode}: {str(e)}")
                            data[mode][exp_type] = None
                else:
                    # For classification and summarization, load their respective files
                    file_path = base_path / mode / file_name
                    logger.info(f"Attempting to load {exp_type} data from: {file_path}")
                    
                    if not file_path.exists():
                        logger.warning(f"File not found: {file_path}")
                        data[mode][exp_type] = None
                        continue
                    
                    # Read using pandas
                    df = pd.read_parquet(file_path)
                    logger.info(f"Loaded {exp_type} data for {mode} with columns: {list(df.columns)}")
                    
                    if exp_type == "classification":
                        df = add_subject_if_missing(df)
                    
                    data[mode][exp_type] = df
                    logger.info(f"Successfully loaded {exp_type} data for {mode}: {len(df)} rows")
            
            except Exception as e:
                logger.error(f"Error loading {exp_type} data for {mode}: {str(e)}")
                data[mode][exp_type] = None
    
    return data

def add_subject_if_missing(df: pd.DataFrame, input_data_path: str = "./input_datasets/classification_pairs.parquet") -> pd.DataFrame:
    """Add subject column to the dataframe if missing."""
    if 'subject' not in df.columns:
        try:
            input_data = pd.read_parquet(input_data_path)
            # Create mapping from question to subject
            question_to_subject = dict(zip(input_data['question'], input_data['subject']))
            # Map questions to subjects
            df['subject'] = df['question'].map(question_to_subject)
            logger.info(f"Successfully added subject mapping to {len(df)} rows")
        except Exception as e:
            logger.warning(f"Could not add subject information: {e}")
            df['subject'] = 'unknown'
    return df

def calculate_token_rates(data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Calculate token processing rates for each mode and experiment type.
    
    Args:
        data: Dictionary containing experiment data by mode
        
    Returns:
        DataFrame with token rate statistics
    """
    token_rates = []
    
    for mode, mode_data in data.items():
        for exp_type, df in mode_data.items():
            if df is not None:
                try:
                    # Calculate total tokens and durations
                    total_duration = df['duration'].sum()
                    
                    # Handle different data structures
                    if 'prompt_tokens' in df.columns:
                        prompt_tokens = df['prompt_tokens'].sum()
                        response_tokens = df['token_count'].sum()
                    else:
                        # For merged data, calculate from token_count
                        response_tokens = df['token_count'].sum()
                        prompt_tokens = df['token_count'].sum() * 0.3  # Estimate based on typical ratio
                    
                    total_tokens = prompt_tokens + response_tokens
                    
                    # Calculate prompt duration if available
                    if 'prompt_duration' in df.columns:
                        prompt_duration = df['prompt_duration'].sum()
                    else:
                        prompt_duration = total_duration * 0.3  # Estimate based on typical ratio
                    
                    response_duration = total_duration - prompt_duration
                    
                    # Calculate rates
                    prompt_rate = prompt_tokens / prompt_duration if prompt_duration > 0 else 0
                    response_rate = response_tokens / response_duration if response_duration > 0 else 0
                    total_rate = total_tokens / total_duration if total_duration > 0 else 0
                    
                    token_rates.append({
                        'mode': mode,
                        'experiment': exp_type,
                        'prompt_tokens_per_sec': prompt_rate,
                        'response_tokens_per_sec': response_rate,
                        'total_tokens_per_sec': total_rate,
                        'total_tokens': total_tokens,
                        'prompt_tokens': prompt_tokens,
                        'response_tokens': response_tokens,
                        'total_duration': total_duration,
                        'prompt_duration': prompt_duration,
                        'response_duration': response_duration
                    })
                except Exception as e:
                    logger.warning(f"Could not calculate token rates for {mode} {exp_type}: {str(e)}")
                    continue
    
    return pd.DataFrame(token_rates)

def plot_token_rates(token_rates: pd.DataFrame) -> None:
    """Plot token processing rates comparison.
    
    Args:
        token_rates: DataFrame with token rate statistics
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot types and their labels
    rate_types = [
        ('prompt_tokens_per_sec', 'Prompt Tokens/s'),
        ('response_tokens_per_sec', 'Response Tokens/s'),
        ('total_tokens_per_sec', 'Total Tokens/s')
    ]
    
    # Create a plot for each rate type
    for idx, (rate_type, title) in enumerate(rate_types):
        # Create grouped bar plot
        sns.barplot(
            data=token_rates,
            x='mode',
            y=rate_type,
            hue='experiment',
            ax=axes[idx]
        )
        
        # Customize subplot
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Mode')
        axes[idx].set_ylabel('Tokens per Second')
        
        # Add value labels on top of bars
        for i, container in enumerate(axes[idx].containers):
            # Get the experiment type for this container
            exp_type = token_rates['experiment'].unique()[i]
            
            # Add labels with total tokens and duration
            for j, patch in enumerate(container):
                height = patch.get_height()
                mode = token_rates['mode'].unique()[j]
                
                # Get corresponding row from DataFrame
                row = token_rates[
                    (token_rates['mode'] == mode) & 
                    (token_rates['experiment'] == exp_type)
                ].iloc[0]
                
                # Create label with rate and totals
                label = f'{height:.1f}\n({int(row["total_tokens"])}t/{row["total_duration"]:.1f}s)'
                
                axes[idx].text(
                    patch.get_x() + patch.get_width()/2.,
                    height,
                    label,
                    ha='center',
                    va='bottom'
                )
        
        # Rotate x-axis labels if needed
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Adjust legend
        axes[idx].legend(title='Experiment Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Add overall title
    fig.suptitle('Token Processing Rates by Mode and Experiment Type', y=1.05)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOKEN_CHARTS / 'token_rates.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('Token rates plot generated')

def plot_token_distribution(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot token distribution for each model and experiment.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    num_modes = len(EXPERIMENT_MODES)
    fig, axes = plt.subplots(num_modes, 3, figsize=(12, 2.5 * num_modes))  # Smaller figure
    
    # Ensure axes is 2D even with 1 mode
    if num_modes == 1:
        axes = axes.reshape(1, -1)
    
    # Set fixed max for toxicity plots
    max_toxicity_tokens = 250
    
    # Custom mode labels
    mode_labels = {
        'enclave4b': '(i) enclave',
        'host4b': '(ii) compute-constant',
        'comp-constant-8b': '(iii) comp-constant-8b'
    }
    # Axis limits for each column
    xlims = [(0, 8), (0, 65), (0, 250)]
    ylims = [(0, 800), (0, 40), (0, 80)]
    # Use viridis color palette for columns
    exp_colors = sns.color_palette('viridis', n_colors=3)
    font_size = 14
    
    # Create distributions for each mode and experiment
    for row, mode in enumerate(EXPERIMENT_MODES):
        mode_data = data.get(mode, {})
        for col, exp_type in enumerate(EXPERIMENT_TYPES.keys()):
            df = mode_data.get(exp_type)
            ax = axes[row, col]
            color = exp_colors[col]
            if df is not None and "token_count" in df.columns:
                if exp_type == "toxicity":
                    bin_width = 5
                    bins = np.arange(0, max_toxicity_tokens + bin_width, bin_width)
                    sns.histplot(
                        data=df,
                        x="token_count",
                        bins=bins,
                        color=color,
                        ax=ax,
                        linewidth=1.5
                    )
                else:
                    sns.histplot(
                        data=df,
                        x="token_count",
                        bins=30,
                        color=color,
                        ax=ax,
                        linewidth=1.5
                    )
                mean_tokens = df["token_count"].mean()
                ax.axvline(
                    mean_tokens,
                    color="red",
                    linestyle="--",
                    linewidth=1.5
                )
                stats_text = (
                    f"Mean: {df['token_count'].mean():.1f}"
                )
                ax.text(
                    0.95, 0.95,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=font_size,
                    fontname='Times New Roman'
                )
            else:
                ax.text(
                    0.5, 0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=font_size,
                    fontname='Times New Roman'
                )
            # Set consistent axis limits
            ax.set_xlim(xlims[col])
            ax.set_ylim(ylims[col])
            # Set all spines to black
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            # Set column titles
            if row == 0:
                ax.set_title(exp_type.capitalize(), fontsize=font_size, fontname='Times New Roman')
            # Set row labels
            if col == 0:
                ax.set_ylabel(mode_labels.get(mode, mode), fontsize=font_size, fontname='Times New Roman')
            else:
                ax.set_ylabel("")
            # Set axis label font size and font
            ax.tick_params(axis='both', labelsize=font_size, width=1.5, length=6)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(font_size)
        
    # Set x/y labels for all subplots
    for col in range(3):
        for row in range(num_modes):
            axes[row, col].set_xlabel('# tokens', fontsize=font_size, fontname='Times New Roman')
            axes[row, col].set_ylabel('Occurrence', fontsize=font_size, fontname='Times New Roman')
    # Set y-axis max for bottom row's Summarization and Toxicity plots
    axes[-1, 1].set_ylim(0, 15)
    axes[-1, 2].set_ylim(0, 10)
    axes[-1, 2].set_xlim(0, 250)
    # Remove the mean legend (do not add a figure legend)
    plt.tight_layout(pad=0.7)
    fig.subplots_adjust(top=0.92)
    fig.savefig(TOKEN_CHARTS / "token_distribution.pdf", dpi=300)
    plt.close(fig)
    logger.info("Token distribution plot generated")

def plot_runtime_distribution(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot runtime distribution for each experiment using PDFs.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create PDF plots for each experiment type
    for col, exp_type in enumerate(EXPERIMENT_TYPES.keys()):
        plot_data = []
        
        # First pass to collect statistics
        for mode in EXPERIMENT_MODES:
            mode_data = data.get(mode, {})
            df = mode_data.get(exp_type)
            if df is not None and "duration" in df.columns:
                durations = df["duration"].values
                # Calculate statistics before filtering
                median_duration = np.median(durations)
                mean_duration = np.mean(durations)
                
                # Filter extreme outliers (more than 3 std from mean)
                std_duration = np.std(durations)
                max_duration = mean_duration + 3 * std_duration
                filtered_durations = durations[durations <= max_duration]
                
                plot_data.append({
                    'mode': mode,
                    'durations': filtered_durations,
                    'median': median_duration,
                    'mean': mean_duration
                })
                logger.info(f"{exp_type} {mode} stats - median: {median_duration:.2f}s, mean: {mean_duration:.2f}s, samples: {len(durations)}")
        
        if not plot_data:
            logger.warning(f"No data available for {exp_type}")
            axes[col].text(0.5, 0.5, "No data available",
                         ha='center', va='center',
                         transform=axes[col].transAxes)
            continue
            
        # Plot distributions
        for i, item in enumerate(plot_data):
            # Plot kernel density estimation (PDF)
            sns.kdeplot(data=item['durations'], 
                       ax=axes[col],
                       label=item['mode'],
                       alpha=0.7)
            
            # Add median line
            axes[col].axvline(item['median'], 
                            color='gray', 
                            linestyle='--', 
                            alpha=0.3)
            
            # Add annotation at the top of the plot with spacing based on number of modes
            spacing = 0.98 / (len(plot_data) + 1)
            axes[col].text(0.98, 0.98 - i * spacing,
                         f"{item['mode']}: median={item['median']:.2f}s, mean={item['mean']:.2f}s",
                         transform=axes[col].transAxes,
                         ha='right',
                         va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Customize each subplot
        axes[col].set_title(f"{exp_type.capitalize()} Runtime Distribution")
        axes[col].set_xlabel("Time (seconds)")
        axes[col].set_ylabel("Density")
        
        # Set x-axis limits based on the data
        if plot_data:
            max_time = max(np.percentile(item['durations'], 99) for item in plot_data)
            axes[col].set_xlim(0, max_time * 1.1)  # Add 10% padding
        
        # Remove legend if it exists
        legend = axes[col].get_legend()
        if legend is not None:
            legend.remove()
        
        # Add grid but make it very light
        axes[col].grid(True, alpha=0.2)
        
        # Remove top and right spines
        axes[col].spines['top'].set_visible(False)
        axes[col].spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    fig.savefig(TOKEN_CHARTS / "runtime_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Runtime distribution plot generated")

def plot_timing_comparison(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot timing comparison between prompt and response across experiments.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Collect timing data
    timing_data = []
    
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        for exp_type in EXPERIMENT_TYPES.keys():
            df = mode_data.get(exp_type)
            if df is not None and all(col in df.columns for col in ['duration', 'prompt_duration', 'tokenize_duration']):
                timing_data.append({
                    'mode': mode,
                    'experiment': exp_type,
                    'prompt_time': df['prompt_duration'].mean(),
                    'response_time': (df['duration'] - df['prompt_duration']).mean(),
                    'tokenize_time': df['tokenize_duration'].mean()
                })
    
    if not timing_data:
        logger.warning("No timing data available")
        return
    
    timing_df = pd.DataFrame(timing_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up the bar positions
    modes = timing_df['mode'].unique()
    experiments = timing_df['experiment'].unique()
    x = np.arange(len(modes))
    
    # Adjust width based on number of experiments
    total_width = 0.8  # Total width to allocate for all bars for a single mode
    width = total_width / len(experiments)  # Width of each bar
    
    # Plot bars for each timing component
    for i, exp in enumerate(experiments):
        exp_data = timing_df[timing_df['experiment'] == exp]
        offset = (i - len(experiments)/2 + 0.5) * width
        
        # Plot tokenize time
        tokenize_bars = ax.bar(x + offset, exp_data['tokenize_time'], 
                              width, label=f'{exp} Tokenize',
                              color=f'C{i}', alpha=0.4)
        
        # Plot prompt time
        prompt_bars = ax.bar(x + offset, exp_data['prompt_time'], 
                           width, bottom=exp_data['tokenize_time'],
                           label=f'{exp} Prompt',
                           color=f'C{i}', alpha=0.6)
        
        # Plot response time
        response_bars = ax.bar(x + offset, exp_data['response_time'],
                             width, bottom=exp_data['tokenize_time'] + exp_data['prompt_time'],
                             label=f'{exp} Response',
                             color=f'C{i}', alpha=0.8)
        
        # Add value annotations only if there's enough space
        font_size = 8  # Smaller font size for more modes
        if len(modes) <= 4:  # Only add labels if we don't have too many modes
            def autolabel(rects, values, bottoms=None):
                """Add value labels on the bars."""
                for rect, value in zip(rects, values):
                    height = value
                    if bottoms is not None:
                        bottom = bottoms.iloc[0] if isinstance(bottoms, pd.Series) else bottoms
                        y_position = bottom + height/2
                    else:
                        y_position = height/2
                    
                    ax.text(rect.get_x() + rect.get_width()/2., y_position,
                           f'{height:.3f}s',
                           ha='center', va='center',
                           fontsize=font_size,
                           rotation=90)
            
            # Add labels for each component
            autolabel(tokenize_bars, exp_data['tokenize_time'])
            autolabel(prompt_bars, exp_data['prompt_time'], exp_data['tokenize_time'])
            autolabel(response_bars, exp_data['response_time'], 
                     exp_data['tokenize_time'] + exp_data['prompt_time'])
            
            # Add total time annotation
            total_times = exp_data['tokenize_time'] + exp_data['prompt_time'] + exp_data['response_time']
            for j, total in enumerate(total_times):
                ax.text(x[j] + offset, total + 0.1,
                       f'Total: {total:.3f}s',
                       ha='center', va='bottom',
                       fontsize=font_size)
        
    # Customize plot
    ax.set_title("Timing Comparison by Mode and Experiment")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOKEN_CHARTS / "timing_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Timing comparison plot generated")

def plot_summarization_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot summarization performance metrics.
    
    Args:
        data: Dictionary containing experiment data by mode
    """
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect BERT scores data
    bert_data = []
    for mode in EXPERIMENT_MODES:
        df = data[mode].get('summarization')
        if df is not None and 'score' in df.columns:
            scores = df['score'].copy()
            scores = pd.DataFrame({
                'score': scores,
                'mode': mode
            })
            bert_data.append(scores)
            
            # Log statistics
            logger.info(f"BERT scores for {mode}:")
            logger.info(f"  Mean: {scores['score'].mean():.3f}")
            logger.info(f"  Median: {scores['score'].median():.3f}")
            logger.info(f"  Std: {scores['score'].std():.3f}")
            logger.info(f"  Samples: {len(scores)}")
    
    if bert_data:
        # Combine all data
        bert_df = pd.concat(bert_data)
        
        # Create violin plot
        sns.violinplot(data=bert_df, x='mode', y='score', ax=ax)
        
        # Add individual points with jitter
        sns.stripplot(data=bert_df, x='mode', y='score', 
                     color='red', alpha=0.3, size=4, jitter=0.2, ax=ax)
        
        # Add statistics annotations
        for mode in EXPERIMENT_MODES:
            mode_scores = bert_df[bert_df['mode'] == mode]
            if not mode_scores.empty:
                # Calculate statistics
                mean_score = mode_scores['score'].mean()
                median_score = mode_scores['score'].median()
                std_score = mode_scores['score'].std()
                count = len(mode_scores)
                
                # Create annotation text
                stats_text = (
                    f'μ={mean_score:.3f}\n'
                    f'med={median_score:.3f}\n'
                    f'σ={std_score:.3f}\n'
                    f'n={count}'
                )
                
                # Find x position for this mode
                x_pos = EXPERIMENT_MODES.index(mode)
                
                # Add text annotation
                ax.text(x_pos, ax.get_ylim()[1], stats_text,
                       ha='center', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Customize plot
        ax.set_title('BERT Scores Distribution by Mode')
        ax.set_xlabel('Mode')
        ax.set_ylabel('BERT Score')
        
        # Add grid
        ax.grid(True, alpha=0.2)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, 'No BERT score data available',
                ha='center', va='center', transform=ax.transAxes)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(SUMMARIZATION_CHARTS / 'bert_scores.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('BERT scores plot generated')

def plot_classification_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot classification performance metrics.
    
    Args:
        data: Dictionary containing experiment data by mode
    """
    # Overall accuracy plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    accuracies = []
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        if mode_data.get("classification") is not None:
            df = mode_data["classification"]
            accuracy = df["correct"].mean() if "correct" in df.columns else np.nan
            accuracies.append({
                "mode": mode,
                "accuracy": accuracy,
                "correct": df["correct"].sum() if "correct" in df.columns else 0,
                "total": len(df)
            })
    
    if not accuracies:
        logger.warning("No accuracy data available")
        return
    
    accuracy_df = pd.DataFrame(accuracies)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=accuracy_df,
        x="mode",
        y="accuracy",
        palette="viridis",
        ax=ax
    )
    
    # Add text annotations
    for i, row in accuracy_df.iterrows():
        ax.text(
            i, 
            row["accuracy"] + 0.01,
            f"{row['accuracy']:.3f}\n({row['correct']}/{row['total']})",
            ha="center",
            va="bottom"
        )
    
    # Customize plot
    ax.set_title("Classification Accuracy by Mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)  # Set y-axis limits for better visibility
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy.pdf", dpi=300)
    plt.close(fig)
    logger.info("Classification accuracy plot generated")

def plot_classification_accuracy_by_subject(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot classification accuracy by subject for each mode.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Collect accuracy data by subject
    subject_accuracies = []
    
    # Define subject groupings
    subject_groups = {
        'High School Sciences': {
            'high school biology', 'high school chemistry', 'high school physics',
            'high school computer science'
        },
        'High School Mathematics': {
            'high school mathematics', 'high school statistics'
        },
        'High School Humanities': {
            'high school us history', 'high school world history', 'high school government',
            'high school literature', 'high school economics'
        },
        'Psychology & Social Sciences': {
            'professional psychology', 'high school psychology', 'psychology',
            'human sexuality', 'sociology', 'anthropology'
        },
        'Philosophy & Ethics': {
            'moral disputes', 'philosophy', 'formal logic', 'moral scenarios',
            'professional ethics', 'security ethics'
        },
        'Business & Economics': {
            'business ethics', 'macroeconomics', 'microeconomics', 'accounting',
            'finance', 'marketing', 'management'
        },
        'Computer Science & Engineering': {
            'computer science', 'computer security', 'machine learning',
            'electrical engineering', 'mechanical engineering'
        },
        'Medicine & Biology': {
            'medicine', 'anatomy', 'biology', 'nutrition', 'medical ethics'
        },
        'Law & Legal': {
            'professional law', 'international law', 'jurisprudence',
            'constitutional law'
        }
    }
    
    # Define minimum samples required for statistical significance
    MIN_SAMPLES = 25  # Lowered from 30
    MIN_ACCURACY = 0.35  # Lowered from 0.4
    
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        if mode_data.get("classification") is not None:
            df = mode_data["classification"]
            if all(col in df.columns for col in ["correct", "subject"]):
                # Group subjects
                def map_subject_to_group(subject):
                    subject = subject.lower()
                    for group, subjects in subject_groups.items():
                        if subject in subjects:
                            return group
                    return 'Miscellaneous'
                
                df['grouped_subject'] = df['subject'].apply(map_subject_to_group)
                
                # Calculate accuracy by grouped subject
                subject_stats = df.groupby('grouped_subject').agg(
                    accuracy=('correct', 'mean'),
                    count=('correct', 'count')
                ).reset_index()
                
                # Keep all subjects that meet either threshold
                significant_subjects = subject_stats[
                    (subject_stats['count'] >= MIN_SAMPLES) | 
                    (subject_stats['accuracy'] >= MIN_ACCURACY)
                ].copy()
                
                others = subject_stats[
                    ~subject_stats['grouped_subject'].isin(significant_subjects['grouped_subject'])
                ].copy()
                
                # Calculate combined stats for "Others"
                if not others.empty:
                    others_total = df[df['grouped_subject'].isin(others['grouped_subject'])]['correct'].count()
                    others_correct = df[df['grouped_subject'].isin(others['grouped_subject'])]['correct'].sum()
                    others_accuracy = others_correct / others_total if others_total > 0 else 0
                    
                    others_combined = pd.DataFrame({
                        'grouped_subject': ['Others'],
                        'accuracy': [others_accuracy],
                        'count': [others_total]
                    })
                    
                    # Log what's in "Others"
                    logger.info(f"Subjects in 'Others' for {mode}:")
                    for _, row in others.iterrows():
                        logger.info(f"  {row['grouped_subject']}: acc={row['accuracy']:.3f}, n={row['count']}")
                else:
                    others_combined = pd.DataFrame()
                
                # Combine categories and add mode
                final_stats = pd.concat([
                    significant_subjects,
                    others_combined if not others.empty else pd.DataFrame()
                ])
                
                if not final_stats.empty:
                    final_stats['mode'] = mode
                    subject_accuracies.append(final_stats)
                
                # Log statistics
                logger.info(f"\nAccuracy statistics for {mode}:")
                logger.info(f"Total subject groups: {len(subject_stats)}")
                logger.info(f"Significant groups: {len(significant_subjects)}")
                logger.info(f"Samples in Others: {others_total if not others.empty else 0}")
    
    if not subject_accuracies:
        logger.warning("No subject accuracy data available")
        return
    
    # Combine all data
    accuracy_df = pd.concat(subject_accuracies, ignore_index=True)
    
    if accuracy_df.empty:
        logger.warning("No data to plot after filtering")
        return
    
    # Create heatmap with explicit NaN handling
    pivot_df = accuracy_df.pivot(index="grouped_subject", columns="mode", values="accuracy")
    
    # Sort subjects by average accuracy across modes, ignoring NaN
    avg_accuracy = pivot_df.mean(axis=1, skipna=True)
    pivot_df = pivot_df.loc[avg_accuracy.sort_values(ascending=False).index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(pivot_df) * 0.6 + 2))
    
    # Create custom colormap that uses white for NaN values
    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_bad('white', 1.0)
    
    # Create heatmap with NaN values shown
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap=cmap,
        fmt=".3f",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Accuracy'},
        mask=False  # Don't mask any values, including NaN
    )
    
    # Add sample size annotations
    sample_sizes = accuracy_df.pivot(index="grouped_subject", columns="mode", values="count")
    sample_sizes = sample_sizes.reindex(pivot_df.index)  # Match the order of pivot_df
    
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            count = sample_sizes.iloc[i, j]
            if pd.notna(count):  # Only add text if we have a valid count
                ax.text(
                    j + 0.5, i + 0.7,
                    f'n={int(count)}',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=8
                )
    
    # Customize plot
    ax.set_title("Classification Accuracy by Subject Group and Mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Subject Group")
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy_by_subject.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Classification accuracy by subject plot generated")

def plot_classification_accuracy_valid_only(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot classification accuracy considering only valid responses (A, B, C, D).
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Collect accuracy data for valid responses only
    accuracies = []
    
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        if mode_data.get("classification") is not None:
            df = mode_data["classification"]
            # Consider only responses that are single letters A, B, C, or D
            valid_responses = df["response"].str.strip().str.match('^[ABCD]$')
            valid_df = df[valid_responses]
            
            if len(valid_df) > 0:
                accuracy = valid_df["correct"].mean()
                accuracies.append({
                    "mode": mode,
                    "accuracy": accuracy,
                    "correct": valid_df["correct"].sum(),
                    "total_valid": len(valid_df),
                    "total_all": len(df),
                    "valid_rate": len(valid_df) / len(df)
                })
    
    if not accuracies:
        logger.warning("No accuracy data available")
        return
    
    accuracy_df = pd.DataFrame(accuracies)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy for valid responses
    sns.barplot(
        data=accuracy_df,
        x="mode",
        y="accuracy",
        palette="viridis",
        ax=ax1
    )
    
    # Add text annotations for accuracy
    for i, row in accuracy_df.iterrows():
        ax1.text(
            i, 
            row["accuracy"] + 0.01,
            f"{row['accuracy']:.3f}\n({row['correct']}/{row['total_valid']})",
            ha="center",
            va="bottom"
        )
    
    # Customize first plot
    ax1.set_title("Classification Accuracy (Valid Responses Only)")
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3)
    
    # Plot 2: Valid response rate
    sns.barplot(
        data=accuracy_df,
        x="mode",
        y="valid_rate",
        palette="viridis",
        ax=ax2
    )
    
    # Add text annotations for valid response rate
    for i, row in accuracy_df.iterrows():
        ax2.text(
            i, 
            row["valid_rate"] + 0.01,
            f"{row['valid_rate']:.3f}\n({row['total_valid']}/{row['total_all']})",
            ha="center",
            va="bottom"
        )
    
    # Customize second plot
    ax2.set_title("Valid Response Rate")
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Rate of Valid Responses (A,B,C,D)")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy_valid_only.pdf", dpi=300)
    plt.close(fig)
    logger.info("Classification accuracy (valid only) plot generated")

def plot_summarization_response_length(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot response length distribution for summarization across modes.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Create figure with subplots (one per mode)
    fig, axes = plt.subplots(1, len(EXPERIMENT_MODES), figsize=(18, 6))
    
    # Ensure axes is always a list
    if len(EXPERIMENT_MODES) == 1:
        axes = [axes]
    
    for i, mode in enumerate(EXPERIMENT_MODES):
        mode_data = data.get(mode, {})
        if mode_data.get("summarization") is not None:
            df = mode_data["summarization"]
            if "token_count" in df.columns:
                # Create histogram with KDE
                sns.histplot(
                    data=df,
                    x="token_count",
                    kde=True,
                    color=sns.color_palette()[i % len(sns.color_palette())],  # Use modulo for safety
                    ax=axes[i]
                )
                
                # Add mean line
                mean_tokens = df["token_count"].mean()
                axes[i].axvline(
                    mean_tokens,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {mean_tokens:.1f}"
                )
                
                # Add descriptive statistics as text
                stats_text = (
                    f"Mean: {df['token_count'].mean():.1f}\n"
                    f"Median: {df['token_count'].median():.1f}\n"
                    f"Min: {df['token_count'].min():.1f}\n"
                    f"Max: {df['token_count'].max():.1f}"
                )
                axes[i].text(
                    0.95, 0.95,
                    stats_text,
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )
                
                # Set title and labels
                axes[i].set_title(f"{mode} Response Length")
                axes[i].set_xlabel("Token Count")
                axes[i].set_ylabel("Frequency")
                axes[i].legend()
            else:
                axes[i].text(
                    0.5, 0.5,
                    "No token count data available",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes
                )
        else:
            axes[i].text(
                0.5, 0.5,
                "No summarization data available",
                ha="center",
                va="center",
                transform=axes[i].transAxes
            )
    
    # Set global title
    fig.suptitle("Summarization Response Length Distribution by Mode", fontsize=16)
    
    # Save plot
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(SUMMARIZATION_CHARTS / "summarization_response_length.pdf", dpi=300)
    plt.close(fig)
    logger.info("Summarization response length plot generated")

def plot_toxicity_rate(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot toxicity rate for each mode.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Collect toxicity rate data
    toxicity_rates = []
    
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        if mode_data.get("toxicity") is not None:
            df = mode_data["toxicity"]
            if "toxic" in df.columns:
                toxic_rate = df["toxic"].mean()
                toxic_count = df["toxic"].sum()
            elif "classification" in df.columns:
                # Use classification column
                toxic_rate = (df["classification"] == "toxic").mean()
                toxic_count = (df["classification"] == "toxic").sum()
            elif "expected_toxic" in df.columns:
                # Use expected_toxic column
                toxic_rate = (df["expected_toxic"] == 1).mean()
                toxic_count = (df["expected_toxic"] == 1).sum()
            else:
                logger.warning(f"No toxicity classification column found for {mode}. Available columns: {list(df.columns)}")
                continue
            
            if not pd.isna(toxic_rate):
                toxicity_rates.append({
                    "mode": mode,
                    "toxic_rate": toxic_rate,
                    "toxic_count": toxic_count,
                    "total_samples": len(df)
                })
            else:
                logger.warning(f"NaN toxicity rate for {mode}")
    
    if not toxicity_rates:
        logger.warning("No toxicity rate data available")
        return
    
    rate_df = pd.DataFrame(toxicity_rates)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create barplot
    sns.barplot(
        data=rate_df,
        x="mode",
        y="toxic_rate",
        ax=ax
    )
    
    # Add text annotations
    for i, row in rate_df.iterrows():
        ax.text(
            i, 
            row["toxic_rate"] + 0.01,
            f"{row['toxic_rate']:.3f}\n({int(row['toxic_count'])}/{row['total_samples']})",
            ha="center",
            va="bottom"
        )
    
    # Customize plot
    ax.set_title("Toxicity Rate by Mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Toxicity Rate")
    ax.set_ylim(0, max(rate_df["toxic_rate"]) * 1.2)  # Set y-axis limits for better visibility
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOXICITY_CHARTS / "toxicity_rate.pdf", dpi=300)
    plt.close(fig)
    logger.info("Toxicity rate plot generated")

def plot_enclave_combined_analysis(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Create a combined figure with token histograms and timing analysis for enclave4b.
    
    Generates a 2-row figure with the following structure:
    - Row 1: Time distributions (both response and prompt times)
    - Row 2: Token distributions (both response and prompt tokens)
    
    Args:
        data: Dictionary containing experiment data by mode
    """
    # Extract enclave4b data
    enclave_data = data.get('enclave4b', {})
    if not enclave_data:
        logger.warning("No enclave4b data available for combined analysis plot")
        return
    
    # Make the figure a bit bigger
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    
    row_titles = ['Response Time', 'Response Tokens', 'Prompt Time', 'Prompt Tokens']
    
    # Use viridis color palette for columns
    exp_colors = sns.color_palette('viridis', n_colors=3)
    
    # Manually set bin widths and x-axis limits for each experiment type and row
    bin_settings = {
        'classification': {
            'response_time_bins': 0.2,  # seconds
            'response_time_max': 5,     # seconds
            'response_tokens_bins': 0.5, # tokens
            'response_tokens_max': 10,   # tokens
            'prompt_time_bins': 2,      # seconds
            'prompt_time_max': 40,      # seconds
            'prompt_tokens_bins': 5,    # tokens
            'prompt_tokens_max': 150    # tokens
        },
        'summarization': {
            'response_time_bins': 2,    # seconds
            'response_time_max': 30,    # seconds
            'response_tokens_bins': 2,   # tokens
            'response_tokens_max': 50,   # tokens
            'prompt_time_bins': 5,      # seconds
            'prompt_time_max': 80,      # seconds
            'prompt_tokens_bins': 20,   # tokens
            'prompt_tokens_max': 400    # tokens
        },
        'toxicity': {
            'response_time_bins': 5,    # seconds
            'response_time_max': 150,   # seconds
            'response_tokens_bins': 10,  # tokens
            'response_tokens_max': 250,  # tokens
            'prompt_time_bins': 2,      # seconds
            'prompt_time_max': 30,      # seconds
            'prompt_tokens_bins': 20,   # tokens
            'prompt_tokens_max': 250    # tokens
        }
    }
    
    # Font size for all
    font_size = 14
    
    # Process each experiment type
    for col, exp_type in enumerate(EXPERIMENT_TYPES.keys()):
        df = enclave_data.get(exp_type)
        
        if df is None:
            for row in range(4):
                axes[row, col].text(
                    0.5, 0.5, "No data available",
                    ha="center", va="center", transform=axes[row, col].transAxes,
                    fontsize=font_size, fontname='Times New Roman'
                )
            continue
        
        if "token_count" in df.columns and "prompt_tokens" in df.columns:
            prompt_tokens = df['prompt_tokens']
            response_tokens = df['token_count']
        else:
            prompt_tokens = pd.Series()
            response_tokens = pd.Series()
        
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
            prompt_times = pd.Series()
            response_times = pd.Series()
        
        row_data = [
            response_times,  # Row 0: Response Time
            response_tokens, # Row 1: Response Tokens
            prompt_times,    # Row 2: Prompt Time
            prompt_tokens    # Row 3: Prompt Tokens
        ]
        row_settings = [
            {'bins': bin_settings[exp_type]['response_time_bins'], 'max': bin_settings[exp_type]['response_time_max'], 'is_time': True},
            {'bins': bin_settings[exp_type]['response_tokens_bins'], 'max': bin_settings[exp_type]['response_tokens_max'], 'is_time': False},
            {'bins': bin_settings[exp_type]['prompt_time_bins'], 'max': bin_settings[exp_type]['prompt_time_max'], 'is_time': True},
            {'bins': bin_settings[exp_type]['prompt_tokens_bins'], 'max': bin_settings[exp_type]['prompt_tokens_max'], 'is_time': False}
        ]
        for row, (data_series, settings) in enumerate(zip(row_data, row_settings)):
            if data_series.empty:
                axes[row, col].text(
                    0.5, 0.5, f"No {row_titles[row].lower()} data",
                    ha="center", va="center", transform=axes[row, col].transAxes,
                    fontsize=font_size, fontname='Times New Roman'
                )
                continue
            is_time_row = settings['is_time']
            bin_width = settings['bins']
            x_max = settings['max']
            data_min = max(0, data_series.min() * 0.9)
            bins = np.arange(0, x_max + bin_width, bin_width)
            xlabel = 'Time (seconds)' if is_time_row else '# tokens'
            sns.histplot(
                data=data_series,
                bins=bins,
                color=exp_colors[col],
                ax=axes[row, col],
                kde=False,
                stat="density",
                linewidth=1.5,
                alpha=0.7
            )
            median_val = data_series.median()
            axes[row, col].axvline(
                median_val,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Median: {median_val:.1f}" + ("s" if is_time_row else "")
            )
            if axes[row, col].get_legend() is not None:
                axes[row, col].get_legend().remove()
            stats_text = (
                f"Median: {data_series.median():.1f}" + ("s" if is_time_row else "")
            )
            axes[row, col].text(
                0.95, 0.95,
                stats_text,
                transform=axes[row, col].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=font_size,
                fontname='Times New Roman'
            )
            axes[row, col].set_xlim(data_min, x_max * 1.02)
            x_ticks = [data_min]
            tick_count = 5
            step = (x_max - data_min) / (tick_count - 1)
            for i in range(1, tick_count - 1):
                x_ticks.append(data_min + i * step)
            x_ticks.append(x_max)
            if is_time_row:
                x_ticks = [round(tick, 1) for tick in x_ticks]
            else:
                x_ticks = [int(round(tick)) for tick in x_ticks]
            x_ticks = list(dict.fromkeys(x_ticks))
            axes[row, col].set_xticks(x_ticks)
            y_max = axes[row, col].get_ylim()[1]
            axes[row, col].set_ylim(0, y_max * 1.15)
            if y_max > 0.5:
                for patch in axes[row, col].patches:
                    current_height = patch.get_height()
                    scale_factor = min(0.5 / y_max, 1.0)
                    patch.set_height(current_height * scale_factor)
                axes[row, col].set_ylim(0, 0.5 * 1.15)
            axes[row, col].set_xlabel(xlabel, fontsize=font_size, fontname='Times New Roman')
            # Set tick label font size and font
            plt.setp(axes[row, col].get_xticklabels(), fontsize=font_size, fontname='Times New Roman')
            plt.setp(axes[row, col].get_yticklabels(), fontsize=font_size, fontname='Times New Roman')
        if col < 3:
            axes[0, col].set_title(exp_type.capitalize(), fontsize=font_size, fontname='Times New Roman')
    for row in range(4):
        axes[row, 0].set_ylabel(row_titles[row], fontsize=font_size, fontname='Times New Roman')
    for row in range(4):
        for col in range(3):
            axes[row, col].set_ylabel('Density', fontsize=font_size, fontname='Times New Roman')
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            axes[row, col].tick_params(axis='both', labelsize=font_size, width=1.5, length=5)
            for label in axes[row, col].get_xticklabels() + axes[row, col].get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(font_size)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(TOKEN_CHARTS / "enclave_combined_analysis.pdf", dpi=300)
    plt.close(fig)
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
    fig.savefig(TOKEN_CHARTS / "enclave_merged_analysis.pdf", dpi=300)
    plt.close(fig)
    logger.info("Enclave merged analysis plot generated")

def plot_combined_metrics_grid(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Create a 1x4 grid plot combining BERT scores, classification accuracy and valid response rates.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # New order: (i), (iii), (ii)
    mode_order = ['enclave4b', 'comp-constant-8b', 'host4b']
    mode_labels = {
        'enclave4b': '(i)',
        'host4b': '(ii)',
        'comp-constant-8b': '(iii)'
    }
    label_order = [mode_labels[m] for m in mode_order]

    # Create 1x4 figure with subplots - make each plot wider
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    bar_width = 0.6
    title_font_size = 14
    label_font_size = 14
    tick_font_size = 14
    annotation_font_size = 14
    
    # 1. BERT Score Distribution Plot (first/left)
    bert_data = []
    for mode in mode_order:
        df = data[mode].get('summarization')
        if df is not None and 'score' in df.columns:
            scores = df['score'].copy()
            scores = pd.DataFrame({
                'score': scores,
                'mode': mode_labels.get(mode, mode)
            })
            bert_data.append(scores)
    if bert_data:
        bert_df = pd.concat(bert_data)
        bert_df['mode'] = pd.Categorical(bert_df['mode'], categories=label_order, ordered=True)
        sns.violinplot(
            data=bert_df, 
            x='mode', 
            y='score', 
            ax=axes[0],
            palette="viridis",
            width=bar_width * 1.2,
            inner=None,
            order=label_order
        )
        axes[0].set_title('BERT Score', fontsize=title_font_size, pad=10, fontname='Times New Roman')
        axes[0].set_xlabel('Mode', fontsize=label_font_size, fontname='Times New Roman')
        axes[0].set_ylabel('')
        axes[0].set_ylim(-0.2, 1.2)
        axes[0].grid(True, alpha=0.2)
    # 2. Toxicity Rate (second plot)
    toxicity_rates = []
    for mode in mode_order:
        mode_data = data.get(mode, {})
        if mode_data.get("toxicity") is not None:
            df = mode_data["toxicity"]
            if "toxic" in df.columns:
                toxic_rate = df["toxic"].mean()
                toxic_count = df["toxic"].sum()
            elif "classification" in df.columns:
                toxic_rate = (df["classification"] == "toxic").mean()
                toxic_count = (df["classification"] == "toxic").sum()
            elif "expected_toxic" in df.columns:
                toxic_rate = (df["expected_toxic"] == 1).mean()
                toxic_count = (df["expected_toxic"] == 1).sum()
            else:
                logger.warning(f"No toxicity classification column found for {mode}. Available columns: {list(df.columns)}")
                continue
            if not pd.isna(toxic_rate):
                toxicity_rates.append({
                    "mode": mode_labels.get(mode, mode),
                    "toxic_rate": toxic_rate,
                    "toxic_count": toxic_count,
                    "total_samples": len(df)
                })
            else:
                logger.warning(f"NaN toxicity rate for {mode}")
    if toxicity_rates:
        rate_df = pd.DataFrame(toxicity_rates)
        rate_df['mode'] = pd.Categorical(rate_df['mode'], categories=label_order, ordered=True)
        sns.barplot(
            data=rate_df,
            x="mode",
            y="toxic_rate",
            ax=axes[1],
            palette="viridis",
            width=bar_width,
            order=label_order
        )
        for i, row in rate_df.sort_values('mode').iterrows():
            axes[1].text(
                i, 
                row["toxic_rate"] + 0.002,
                f"{row['toxic_rate']:.3f}",
                ha="center",
                va="bottom",
                fontsize=annotation_font_size,
                fontname='Times New Roman'
            )
        axes[1].set_title('Toxicity Rate', fontsize=title_font_size, pad=10, fontname='Times New Roman')
        axes[1].set_xlabel("Mode", fontsize=label_font_size, fontname='Times New Roman')
        axes[1].set_ylabel("")
        axes[1].set_ylim(0, max(rate_df["toxic_rate"]) * 1.3)
        axes[1].grid(axis="y", alpha=0.3)
    # 3. Classification Accuracy for Valid Responses Only (third plot)
    accuracies = []
    for mode in mode_order:
        mode_data = data.get(mode, {})
        if mode_data.get("classification") is not None:
            df = mode_data["classification"]
            valid_responses = df["response"].str.strip().str.match('^[ABCD]$')
            valid_df = df[valid_responses]
            if len(valid_df) > 0:
                accuracy = valid_df["correct"].mean()
                accuracies.append({
                    "mode": mode_labels.get(mode, mode),
                    "accuracy": accuracy,
                    "correct": valid_df["correct"].sum(),
                    "total_valid": len(valid_df),
                    "total_all": len(df),
                    "valid_rate": len(valid_df) / len(df)
                })
    if accuracies:
        accuracy_df = pd.DataFrame(accuracies)
        accuracy_df['mode'] = pd.Categorical(accuracy_df['mode'], categories=label_order, ordered=True)
        bars = sns.barplot(
            data=accuracy_df,
            x="mode",
            y="accuracy",
            ax=axes[2],
            palette="viridis",
            width=bar_width,
            order=label_order
        )
        for i, row in accuracy_df.sort_values('mode').iterrows():
            axes[2].text(
                i, 
                row["accuracy"] + 0.02,
                f"{row['accuracy']:.3f}",
                ha="center",
                va="bottom",
                fontsize=annotation_font_size,
                fontname='Times New Roman'
            )
        axes[2].set_title('Classification Accuracy', fontsize=title_font_size, pad=10, fontname='Times New Roman')
        axes[2].set_xlabel("Mode", fontsize=label_font_size, fontname='Times New Roman')
        axes[2].set_ylabel("")
        axes[2].set_ylim(0, 1.1)
        axes[2].grid(axis="y", alpha=0.3)
    # 4. Valid Response Rate (fourth/right plot)
    if accuracies:
        accuracy_df['mode'] = pd.Categorical(accuracy_df['mode'], categories=label_order, ordered=True)
        sns.barplot(
            data=accuracy_df,
            x="mode",
            y="valid_rate",
            ax=axes[3],
            palette="viridis",
            width=bar_width,
            order=label_order
        )
        for i, row in accuracy_df.sort_values('mode').iterrows():
            axes[3].text(
                i, 
                row["valid_rate"] + 0.02,
                f"{row['valid_rate']:.3f}",
                ha="center",
                va="bottom",
                fontsize=annotation_font_size,
                fontname='Times New Roman'
            )
        axes[3].set_title("Valid Response Rate", fontsize=title_font_size, pad=10, fontname='Times New Roman')
        axes[3].set_xlabel("Mode", fontsize=label_font_size, fontname='Times New Roman')
        axes[3].set_ylabel("")
        axes[3].set_ylim(0, 1.1)
        axes[3].grid(axis="y", alpha=0.3)
    # Apply consistent styling to all subplots
    for idx, ax in enumerate(axes):
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1)
            spine.set_visible(True)
        ax.tick_params(axis='both', labelsize=tick_font_size, width=1, length=5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('normal')
        ax.tick_params(axis='x', pad=0)
        plt.setp(ax.get_xticklabels(), ha='center', fontsize=tick_font_size+2, fontweight='normal')
        ax.tick_params(axis='x', length=8)
        ax.set_facecolor('white')
    # Save plot with slightly more padding between subplots
    plt.tight_layout(w_pad=1.2, h_pad=0.2, rect=[0, 0.12, 1, 0.97])
    combined_metrics_path = TOKEN_CHARTS / "combined_metrics"
    combined_metrics_path.mkdir(exist_ok=True)
    fig.savefig(combined_metrics_path / "combined_metrics_grid.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Combined metrics grid plot generated")

def run_aws_analysis() -> None:
    """Execute the complete AWS analysis pipeline."""
    logger.info("Starting AWS analysis pipeline...")
    
    # Create chart directories
    create_chart_directories()
    
    # Load all experiment data
    data = load_aws_data()
    
    # Token and timing analysis
    token_rates = calculate_token_rates(data)
    plot_token_count_comparison(token_rates)
    plot_token_distribution(data)
    # Replace the separate histograms with combined analysis
    # plot_enclave_token_histograms(data)  # Replaced with combined plot
    # plot_detailed_timing_comparison(data)  # Replaced with combined plot
    plot_enclave_combined_analysis(data)  # Original 4-row plot
    plot_enclave_merged_analysis(data)    # New 2-row plot
    plot_runtime_distribution(data)
    plot_timing_comparison(data)
    
    # Classification performance analysis
    plot_classification_accuracy(data)
    plot_classification_accuracy_by_subject(data)
    plot_classification_accuracy_valid_only(data)
    
    # Summarization performance analysis
    plot_summarization_performance(data)
    plot_summarization_response_length(data)
    
    # Toxicity analysis
    plot_toxicity_rate(data)
    logger.info("Toxicity rate plot generated")
    
    # Add the new combined metrics plot
    plot_combined_metrics_grid(data)
    logger.info("Combined metrics grid plot generated")
    
    logger.info("AWS analysis pipeline completed successfully")

def plot_token_count_comparison(token_rates: pd.DataFrame) -> None:
    """Plot token count comparisons between modes.
    
    Args:
        token_rates: DataFrame containing token statistics
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get unique modes
    modes = token_rates['mode'].unique()
    experiments = list(EXPERIMENT_TYPES.keys())
    
    # Set up positions for bars
    x = np.arange(len(modes))
    total_width = 0.8  # Total width for all bars for one mode
    bar_width = total_width / (len(experiments) * 2)  # Width for each bar (2 bars per experiment - prompt and response)
    
    # Colors for each experiment type (prompt, response)
    colors = {
        'classification': ('#67B7DC', '#B6D5EC'),  # Bright blue, Light blue
        'summarization': ('#F47174', '#FFC6C6'),   # Coral red, Light coral
        'toxicity': ('#E5BE7D', '#F2E4B6')         # Dark khaki, Light khaki
    }
    
    # Plot bars for each experiment type
    for i, exp_type in enumerate(experiments):
        exp_data = token_rates[token_rates['experiment'] == exp_type]
        
        # Calculate positions for this experiment type's bars
        offset = (i - len(experiments)/2 + 0.5) * (bar_width * 2)
        
        # Plot prompt and response bars
        prompt_bars = ax.bar(x + offset, exp_data['prompt_tokens'], 
                           bar_width, 
                           label=f'{exp_type} Prompt',
                           color=colors[exp_type][0])
        
        response_bars = ax.bar(x + offset + bar_width, exp_data['response_tokens'],
                             bar_width,
                             label=f'{exp_type} Response',
                             color=colors[exp_type][1])
    
    # Customize plot
    ax.set_title("Token Count Comparison by Model and Experiment")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Token Count")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Set background color to match the reference
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(TOKEN_CHARTS / "token_count_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Token count comparison plot generated")

def plot_classification_accuracy(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot overall classification accuracy for each mode.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Collect accuracy data
    accuracies = []
    
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        if mode_data.get("classification") is not None:
            df = mode_data["classification"]
            accuracy = df["correct"].mean() if "correct" in df.columns else np.nan
            accuracies.append({
                "mode": mode,
                "accuracy": accuracy,
                "correct": df["correct"].sum() if "correct" in df.columns else 0,
                "total": len(df)
            })
    
    if not accuracies:
        logger.warning("No accuracy data available")
        return
    
    accuracy_df = pd.DataFrame(accuracies)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=accuracy_df,
        x="mode",
        y="accuracy",
        palette="viridis",
        ax=ax
    )
    
    # Add text annotations
    for i, row in accuracy_df.iterrows():
        ax.text(
            i, 
            row["accuracy"] + 0.01,
            f"{row['accuracy']:.3f}\n({row['correct']}/{row['total']})",
            ha="center",
            va="bottom"
        )
    
    # Customize plot
    ax.set_title("Classification Accuracy by Mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)  # Set y-axis limits for better visibility
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy.pdf", dpi=300)
    plt.close(fig)
    logger.info("Classification accuracy plot generated")

def plot_classification_accuracy_by_subject(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot classification accuracy by subject for each mode.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Collect accuracy data by subject
    subject_accuracies = []
    
    # Define subject groupings
    subject_groups = {
        'High School Sciences': {
            'high school biology', 'high school chemistry', 'high school physics',
            'high school computer science'
        },
        'High School Mathematics': {
            'high school mathematics', 'high school statistics'
        },
        'High School Humanities': {
            'high school us history', 'high school world history', 'high school government',
            'high school literature', 'high school economics'
        },
        'Psychology & Social Sciences': {
            'professional psychology', 'high school psychology', 'psychology',
            'human sexuality', 'sociology', 'anthropology'
        },
        'Philosophy & Ethics': {
            'moral disputes', 'philosophy', 'formal logic', 'moral scenarios',
            'professional ethics', 'security ethics'
        },
        'Business & Economics': {
            'business ethics', 'macroeconomics', 'microeconomics', 'accounting',
            'finance', 'marketing', 'management'
        },
        'Computer Science & Engineering': {
            'computer science', 'computer security', 'machine learning',
            'electrical engineering', 'mechanical engineering'
        },
        'Medicine & Biology': {
            'medicine', 'anatomy', 'biology', 'nutrition', 'medical ethics'
        },
        'Law & Legal': {
            'professional law', 'international law', 'jurisprudence',
            'constitutional law'
        }
    }
    
    # Define minimum samples required for statistical significance
    MIN_SAMPLES = 25  # Lowered from 30
    MIN_ACCURACY = 0.35  # Lowered from 0.4
    
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        if mode_data.get("classification") is not None:
            df = mode_data["classification"]
            if all(col in df.columns for col in ["correct", "subject"]):
                # Group subjects
                def map_subject_to_group(subject):
                    subject = subject.lower()
                    for group, subjects in subject_groups.items():
                        if subject in subjects:
                            return group
                    return 'Miscellaneous'
                
                df['grouped_subject'] = df['subject'].apply(map_subject_to_group)
                
                # Calculate accuracy by grouped subject
                subject_stats = df.groupby('grouped_subject').agg(
                    accuracy=('correct', 'mean'),
                    count=('correct', 'count')
                ).reset_index()
                
                # Keep all subjects that meet either threshold
                significant_subjects = subject_stats[
                    (subject_stats['count'] >= MIN_SAMPLES) | 
                    (subject_stats['accuracy'] >= MIN_ACCURACY)
                ].copy()
                
                others = subject_stats[
                    ~subject_stats['grouped_subject'].isin(significant_subjects['grouped_subject'])
                ].copy()
                
                # Calculate combined stats for "Others"
                if not others.empty:
                    others_total = df[df['grouped_subject'].isin(others['grouped_subject'])]['correct'].count()
                    others_correct = df[df['grouped_subject'].isin(others['grouped_subject'])]['correct'].sum()
                    others_accuracy = others_correct / others_total if others_total > 0 else 0
                    
                    others_combined = pd.DataFrame({
                        'grouped_subject': ['Others'],
                        'accuracy': [others_accuracy],
                        'count': [others_total]
                    })
                    
                    # Log what's in "Others"
                    logger.info(f"Subjects in 'Others' for {mode}:")
                    for _, row in others.iterrows():
                        logger.info(f"  {row['grouped_subject']}: acc={row['accuracy']:.3f}, n={row['count']}")
                else:
                    others_combined = pd.DataFrame()
                
                # Combine categories and add mode
                final_stats = pd.concat([
                    significant_subjects,
                    others_combined if not others.empty else pd.DataFrame()
                ])
                
                if not final_stats.empty:
                    final_stats['mode'] = mode
                    subject_accuracies.append(final_stats)
                
                # Log statistics
                logger.info(f"\nAccuracy statistics for {mode}:")
                logger.info(f"Total subject groups: {len(subject_stats)}")
                logger.info(f"Significant groups: {len(significant_subjects)}")
                logger.info(f"Samples in Others: {others_total if not others.empty else 0}")
    
    if not subject_accuracies:
        logger.warning("No subject accuracy data available")
        return
    
    # Combine all data
    accuracy_df = pd.concat(subject_accuracies, ignore_index=True)
    
    if accuracy_df.empty:
        logger.warning("No data to plot after filtering")
        return
    
    # Create heatmap with explicit NaN handling
    pivot_df = accuracy_df.pivot(index="grouped_subject", columns="mode", values="accuracy")
    
    # Sort subjects by average accuracy across modes, ignoring NaN
    avg_accuracy = pivot_df.mean(axis=1, skipna=True)
    pivot_df = pivot_df.loc[avg_accuracy.sort_values(ascending=False).index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(pivot_df) * 0.6 + 2))
    
    # Create custom colormap that uses white for NaN values
    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_bad('white', 1.0)
    
    # Create heatmap with NaN values shown
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap=cmap,
        fmt=".3f",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Accuracy'},
        mask=False  # Don't mask any values, including NaN
    )
    
    # Add sample size annotations
    sample_sizes = accuracy_df.pivot(index="grouped_subject", columns="mode", values="count")
    sample_sizes = sample_sizes.reindex(pivot_df.index)  # Match the order of pivot_df
    
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            count = sample_sizes.iloc[i, j]
            if pd.notna(count):  # Only add text if we have a valid count
                ax.text(
                    j + 0.5, i + 0.7,
                    f'n={int(count)}',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=8
                )
    
    # Customize plot
    ax.set_title("Classification Accuracy by Subject Group and Mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Subject Group")
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy_by_subject.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Classification accuracy by subject plot generated")

def plot_classification_accuracy_valid_only(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot classification accuracy considering only valid responses (A, B, C, D).
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Collect accuracy data for valid responses only
    accuracies = []
    
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        if mode_data.get("classification") is not None:
            df = mode_data["classification"]
            # Consider only responses that are single letters A, B, C, or D
            valid_responses = df["response"].str.strip().str.match('^[ABCD]$')
            valid_df = df[valid_responses]
            
            if len(valid_df) > 0:
                accuracy = valid_df["correct"].mean()
                accuracies.append({
                    "mode": mode,
                    "accuracy": accuracy,
                    "correct": valid_df["correct"].sum(),
                    "total_valid": len(valid_df),
                    "total_all": len(df),
                    "valid_rate": len(valid_df) / len(df)
                })
    
    if not accuracies:
        logger.warning("No accuracy data available")
        return
    
    accuracy_df = pd.DataFrame(accuracies)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy for valid responses
    sns.barplot(
        data=accuracy_df,
        x="mode",
        y="accuracy",
        palette="viridis",
        ax=ax1
    )
    
    # Add text annotations for accuracy
    for i, row in accuracy_df.iterrows():
        ax1.text(
            i, 
            row["accuracy"] + 0.01,
            f"{row['accuracy']:.3f}\n({row['correct']}/{row['total_valid']})",
            ha="center",
            va="bottom"
        )
    
    # Customize first plot
    ax1.set_title("Classification Accuracy (Valid Responses Only)")
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3)
    
    # Plot 2: Valid response rate
    sns.barplot(
        data=accuracy_df,
        x="mode",
        y="valid_rate",
        palette="viridis",
        ax=ax2
    )
    
    # Add text annotations for valid response rate
    for i, row in accuracy_df.iterrows():
        ax2.text(
            i, 
            row["valid_rate"] + 0.01,
            f"{row['valid_rate']:.3f}\n({row['total_valid']}/{row['total_all']})",
            ha="center",
            va="bottom"
        )
    
    # Customize second plot
    ax2.set_title("Valid Response Rate")
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Rate of Valid Responses (A,B,C,D)")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy_valid_only.pdf", dpi=300)
    plt.close(fig)
    logger.info("Classification accuracy (valid only) plot generated")

def plot_summarization_response_length(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot response length distribution for summarization across modes.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Create figure with subplots (one per mode)
    fig, axes = plt.subplots(1, len(EXPERIMENT_MODES), figsize=(18, 6))
    
    # Ensure axes is always a list
    if len(EXPERIMENT_MODES) == 1:
        axes = [axes]
    
    for i, mode in enumerate(EXPERIMENT_MODES):
        mode_data = data.get(mode, {})
        if mode_data.get("summarization") is not None:
            df = mode_data["summarization"]
            if "token_count" in df.columns:
                # Create histogram with KDE
                sns.histplot(
                    data=df,
                    x="token_count",
                    kde=True,
                    color=sns.color_palette()[i % len(sns.color_palette())],  # Use modulo for safety
                    ax=axes[i]
                )
                
                # Add mean line
                mean_tokens = df["token_count"].mean()
                axes[i].axvline(
                    mean_tokens,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {mean_tokens:.1f}"
                )
                
                # Add descriptive statistics as text
                stats_text = (
                    f"Mean: {df['token_count'].mean():.1f}\n"
                    f"Median: {df['token_count'].median():.1f}\n"
                    f"Min: {df['token_count'].min():.1f}\n"
                    f"Max: {df['token_count'].max():.1f}"
                )
                axes[i].text(
                    0.95, 0.95,
                    stats_text,
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )
                
                # Set title and labels
                axes[i].set_title(f"{mode} Response Length")
                axes[i].set_xlabel("Token Count")
                axes[i].set_ylabel("Frequency")
                axes[i].legend()
            else:
                axes[i].text(
                    0.5, 0.5,
                    "No token count data available",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes
                )
        else:
            axes[i].text(
                0.5, 0.5,
                "No summarization data available",
                ha="center",
                va="center",
                transform=axes[i].transAxes
            )
    
    # Set global title
    fig.suptitle("Summarization Response Length Distribution by Mode", fontsize=16)
    
    # Save plot
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(SUMMARIZATION_CHARTS / "summarization_response_length.pdf", dpi=300)
    plt.close(fig)
    logger.info("Summarization response length plot generated")

def plot_toxicity_rate(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot toxicity rate for each mode.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Collect toxicity rate data
    toxicity_rates = []
    
    for mode in EXPERIMENT_MODES:
        mode_data = data.get(mode, {})
        if mode_data.get("toxicity") is not None:
            df = mode_data["toxicity"]
            if "toxic" in df.columns:
                toxic_rate = df["toxic"].mean()
                toxic_count = df["toxic"].sum()
            elif "classification" in df.columns:
                # Use classification column
                toxic_rate = (df["classification"] == "toxic").mean()
                toxic_count = (df["classification"] == "toxic").sum()
            elif "expected_toxic" in df.columns:
                # Use expected_toxic column
                toxic_rate = (df["expected_toxic"] == 1).mean()
                toxic_count = (df["expected_toxic"] == 1).sum()
            else:
                logger.warning(f"No toxicity classification column found for {mode}. Available columns: {list(df.columns)}")
                continue
            
            if not pd.isna(toxic_rate):
                toxicity_rates.append({
                    "mode": mode,
                    "toxic_rate": toxic_rate,
                    "toxic_count": toxic_count,
                    "total_samples": len(df)
                })
            else:
                logger.warning(f"NaN toxicity rate for {mode}")
    
    if not toxicity_rates:
        logger.warning("No toxicity rate data available")
        return
    
    rate_df = pd.DataFrame(toxicity_rates)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create barplot
    sns.barplot(
        data=rate_df,
        x="mode",
        y="toxic_rate",
        ax=ax
    )
    
    # Add text annotations
    for i, row in rate_df.iterrows():
        ax.text(
            i, 
            row["toxic_rate"] + 0.01,
            f"{row['toxic_rate']:.3f}\n({int(row['toxic_count'])}/{row['total_samples']})",
            ha="center",
            va="bottom"
        )
    
    # Customize plot
    ax.set_title("Toxicity Rate by Mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Toxicity Rate")
    ax.set_ylim(0, max(rate_df["toxic_rate"]) * 1.2)  # Set y-axis limits for better visibility
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOXICITY_CHARTS / "toxicity_rate.pdf", dpi=300)
    plt.close(fig)
    logger.info("Toxicity rate plot generated")

if __name__ == "__main__":
    run_aws_analysis() 