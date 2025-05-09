import pandas as pd
import numpy as np
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

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_context("notebook")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Create chart directories
CHART_DIR = Path('analysis/aws_charts')
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
    for directory in [CLASSIFICATION_CHARTS, SUMMARIZATION_CHARTS, TOXICITY_CHARTS, TOKEN_CHARTS]:
        directory.mkdir(parents=True, exist_ok=True)

# Experiment types and locations
EXPERIMENT_TYPES = {
    "classification": "llama_classification.parquet",
    "summarization": "llama_summaries.parquet",
    "toxicity": "llama_toxic.analysis.parquet"
}

EXPERIMENT_MODES = ["enclave4b", "host4b"]

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
                    if mode == "enclave4b":
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
    fig.savefig(TOKEN_CHARTS / 'token_rates.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('Token rates plot generated')

def plot_token_distribution(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot token distribution for each model and experiment.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 modes x 3 experiment types
    
    # Set fixed max for toxicity plots
    max_toxicity_tokens = 250
    
    # Create distributions for each mode and experiment
    for row, mode in enumerate(EXPERIMENT_MODES):
        mode_data = data.get(mode, {})
        for col, exp_type in enumerate(EXPERIMENT_TYPES.keys()):
            df = mode_data.get(exp_type)
            if df is not None and "token_count" in df.columns:
                if exp_type == "toxicity":
                    # For toxicity plots, use small fixed-width bins up to 250
                    bin_width = 5  # 5 tokens per bin for granularity
                    bins = np.arange(0, max_toxicity_tokens + bin_width, bin_width)
                    
                    # Create histogram with KDE
                    sns.histplot(
                        data=df,
                        x="token_count",
                        kde=True,
                        bins=bins,
                        color=sns.color_palette()[col],
                        ax=axes[row, col]
                    )
                    
                    # Set x-axis limit to exactly 250
                    axes[row, col].set_xlim(0, max_toxicity_tokens)
                else:
                    # For other plots, use default binning
                    sns.histplot(
                        data=df,
                        x="token_count",
                        kde=True,
                        bins=30,
                        color=sns.color_palette()[col],
                        ax=axes[row, col]
                    )
                
                # Add mean line
                mean_tokens = df["token_count"].mean()
                axes[row, col].axvline(
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
                axes[row, col].text(
                    0.95, 0.95,
                    stats_text,
                    transform=axes[row, col].transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )
            else:
                axes[row, col].text(
                    0.5, 0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=axes[row, col].transAxes
                )
            
            # Set titles and labels
            if row == 0:
                axes[row, col].set_title(exp_type.capitalize())
            if col == 0:
                axes[row, col].set_ylabel(mode)
    
    # Add global title
    fig.suptitle("Token Count Distribution by Mode and Experiment", fontsize=20)
    
    # Save plot
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(TOKEN_CHARTS / "token_distribution.png", dpi=300)
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
        for item in plot_data:
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
            
            # Add annotation at the top of the plot
            axes[col].text(0.98, 0.98 - plot_data.index(item) * 0.15,
                         f"{item['mode']}: median={item['median']:.2f}s, mean={item['mean']:.2f}s",
                         transform=axes[col].transAxes,
                         ha='right',
                         va='top',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
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
    fig.savefig(TOKEN_CHARTS / "runtime_distribution.png", dpi=300, bbox_inches='tight')
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
    width = 0.25
    
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
        
        # Add value annotations
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
                   ha='center', va='bottom')
    
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
    fig.savefig(TOKEN_CHARTS / "timing_comparison.png", dpi=300, bbox_inches='tight')
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
    fig.savefig(SUMMARIZATION_CHARTS / 'bert_scores.png', dpi=300, bbox_inches='tight')
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
        df = data[mode].get('classification')
        if df is not None and 'correct' in df.columns:
            accuracy = df['correct'].mean()
            accuracies.append({
                'mode': mode,
                'accuracy': accuracy,
                'correct': df['correct'].sum(),
                'total': len(df)
            })
    
    if accuracies:
        accuracy_df = pd.DataFrame(accuracies)
        sns.barplot(
            data=accuracy_df,
            x='mode',
            y='accuracy',
            ax=axes[0]
        )
        
        # Add annotations
        for i, row in accuracy_df.iterrows():
            axes[0].text(
                i,
                row["accuracy"] + 0.01,
                f"{row['accuracy']:.3f}\n({row['correct']}/{row['total']})",
                ha="center",
                va="bottom"
            )
    
    axes[0].set_title('Classification Accuracy by Mode')
    axes[0].set_ylabel('Accuracy')
    
    # Accuracy by subject
    subject_accuracies = []
    for mode in EXPERIMENT_MODES:
        df = data[mode].get('classification')
        if df is not None and all(col in df.columns for col in ['correct', 'subject']):
            grouped = df.groupby('subject')['correct'].agg(['mean', 'count']).reset_index()
            grouped['mode'] = mode
            subject_accuracies.append(grouped)
    
    if subject_accuracies:
        subject_df = pd.concat(subject_accuracies)
        sns.boxplot(
            data=subject_df,
            x='mode',
            y='mean',
            ax=axes[1]
        )
        axes[1].set_title('Accuracy Distribution by Subject')
        axes[1].set_ylabel('Accuracy')
    
    # Save plot
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / 'classification_performance.png', dpi=300)
    plt.close(fig)
    logger.info('Classification performance plots generated')

def plot_toxicity_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot toxicity performance metrics.
    
    Args:
        data: Dictionary containing experiment data by mode
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Toxicity rate (using expected_toxic)
    toxicity_rates = []
    for mode in EXPERIMENT_MODES:
        df = data[mode].get('toxicity')
        if df is not None and 'expected_toxic' in df.columns:
            toxic_rate = (df['expected_toxic'] == 1).mean()
            toxic_count = (df['expected_toxic'] == 1).sum()
            toxicity_rates.append({
                'mode': mode,
                'toxic_rate': toxic_rate,
                'toxic_count': toxic_count,
                'total_samples': len(df)
            })
    
    if toxicity_rates:
        rate_df = pd.DataFrame(toxicity_rates)
        sns.barplot(
            data=rate_df,
            x='mode',
            y='toxic_rate',
            ax=axes[0, 0]
        )
        
        # Add annotations
        for i, row in rate_df.iterrows():
            ax = axes[0, 0]
            ax.text(
                i, 
                row["toxic_rate"] + 0.01,
                f"{row['toxic_rate']:.3f}\n({int(row['toxic_count'])}/{row['total_samples']})",
                ha="center",
                va="bottom"
            )
    
    axes[0, 0].set_title('Expected Toxicity Rate by Mode')
    axes[0, 0].set_ylabel('Expected Toxicity Rate')
    
    # Response length distribution
    for mode in EXPERIMENT_MODES:
        df = data[mode].get('toxicity')
        if df is not None and 'token_count' in df.columns:
            sns.kdeplot(
                data=df['token_count'],
                ax=axes[0, 1],
                label=mode
            )
    
    axes[0, 1].set_title('Response Length Distribution')
    axes[0, 1].set_xlabel('Token Count')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # Token rate analysis
    token_rates = []
    for mode in EXPERIMENT_MODES:
        df = data[mode].get('toxicity')
        if df is not None and all(col in df.columns for col in ['token_count', 'duration']):
            token_rate = df['token_count'].sum() / df['duration'].sum()
            token_rates.append({
                'mode': mode,
                'token_rate': token_rate,
                'total_tokens': df['token_count'].sum(),
                'total_duration': df['duration'].sum()
            })
    
    if token_rates:
        rate_df = pd.DataFrame(token_rates)
        sns.barplot(
            data=rate_df,
            x='mode',
            y='token_rate',
            ax=axes[1, 0]
        )
        
        # Add annotations
        for i, row in rate_df.iterrows():
            axes[1, 0].text(
                i,
                row['token_rate'] + 0.01,
                f"{row['token_rate']:.1f}\n({int(row['total_tokens'])}t/{row['total_duration']:.1f}s)",
                ha='center',
                va='bottom'
            )
    
    axes[1, 0].set_title('Token Processing Rate')
    axes[1, 0].set_ylabel('Tokens per Second')
    
    # Duration distribution
    for mode in EXPERIMENT_MODES:
        df = data[mode].get('toxicity')
        if df is not None and 'duration' in df.columns:
            sns.kdeplot(
                data=df['duration'],
                ax=axes[1, 1],
                label=mode
            )
            
            # Add statistics
            stats_text = (
                f"{mode}:\n"
                f"Mean: {df['duration'].mean():.2f}s\n"
                f"Median: {df['duration'].median():.2f}s\n"
                f"Std: {df['duration'].std():.2f}s"
            )
            y_pos = 0.95 - EXPERIMENT_MODES.index(mode) * 0.15
            axes[1, 1].text(
                0.95, y_pos,
                stats_text,
                transform=axes[1, 1].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    axes[1, 1].set_title('Response Duration Distribution')
    axes[1, 1].set_xlabel('Duration (seconds)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOXICITY_CHARTS / 'toxicity_performance.png', dpi=300)
    plt.close(fig)
    logger.info('Toxicity performance plots generated')

def plot_accuracy_heatmap(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Generate accuracy heatmap by subject across different modes."""
    has_classification_data = any('classification' in mode_data for mode_data in data.values())
    
    if not has_classification_data:
        logger.warning("No classification data available")
        return
    
    accuracy_data = {}
    
    for mode, mode_data in data.items():
        if 'classification' in mode_data:
            df = mode_data['classification']
            
            # Calculate accuracy by subject
            accuracy_by_subject = df.groupby('subject')['correct'].agg(
                accuracy='mean',
                count='count'
            ).reset_index()
            
            # Add to the dictionary
            for _, row in accuracy_by_subject.iterrows():
                subject = row['subject']
                accuracy = row['accuracy']
                
                if subject not in accuracy_data:
                    accuracy_data[subject] = {}
                    
                accuracy_data[subject][mode] = accuracy
    
    # Convert to DataFrame for heatmap
    heatmap_df = pd.DataFrame(accuracy_data).T
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.3f', 
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title('Accuracy by Subject across Different Modes')
    
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / 'accuracy_heatmap.png', dpi=300)
    plt.close(fig)
    logger.info('Accuracy heatmap generated')

def plot_efficiency_metrics(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot efficiency metrics including token processing rates and memory usage."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Token processing efficiency
    token_rates = calculate_token_rates(data)
    
    # Plot 1: Token processing rates
    sns.barplot(
        data=token_rates,
        x='mode',
        y='total_tokens_per_sec',
        hue='experiment',
        ax=ax1
    )
    
    ax1.set_title('Token Processing Efficiency')
    ax1.set_xlabel('Mode')
    ax1.set_ylabel('Tokens per Second')
    
    # Add value annotations
    for i, container in enumerate(ax1.containers):
        ax1.bar_label(container, fmt='%.1f', padding=3)
    
    # Plot 2: Memory efficiency (if available)
    memory_data = []
    for mode, mode_data in data.items():
        for exp_type, df in mode_data.items():
            if df is not None and 'memory_usage' in df.columns:
                memory_data.append({
                    'mode': mode,
                    'experiment': exp_type,
                    'avg_memory': df['memory_usage'].mean()
                })
    
    if memory_data:
        memory_df = pd.DataFrame(memory_data)
        sns.barplot(
            data=memory_df,
            x='mode',
            y='avg_memory',
            hue='experiment',
            ax=ax2
        )
        
        ax2.set_title('Memory Usage by Mode')
        ax2.set_xlabel('Mode')
        ax2.set_ylabel('Average Memory Usage (MB)')
        
        # Add value annotations
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f', padding=3)
    else:
        ax2.text(0.5, 0.5, 'No memory usage data available',
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    fig.savefig(EFFICIENCY_CHARTS / 'efficiency_metrics.png', dpi=300)
    plt.close(fig)
    logger.info('Efficiency metrics plot generated')

def plot_error_distribution(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot distribution of errors by subject."""
    error_data = {}
    
    for mode, mode_data in data.items():
        if 'classification' in mode_data:
            df = mode_data['classification']
            
            # Group errors by subject
            error_by_subject = df[~df['correct']].groupby('subject').size().reset_index(name='error_count')
            total_by_subject = df.groupby('subject').size().reset_index(name='total_count')
            
            # Merge and calculate error rate
            merged = pd.merge(error_by_subject, total_by_subject, on='subject')
            merged['error_rate'] = merged['error_count'] / merged['total_count']
            
            # Store in dictionary
            for _, row in merged.iterrows():
                subject = row['subject']
                error_rate = row['error_rate']
                
                if subject not in error_data:
                    error_data[subject] = {}
                
                error_data[subject][mode] = error_rate
    
    # Convert to DataFrame
    error_df = pd.DataFrame(error_data).T
    
    # Handle empty data case
    if error_df.empty:
        logger.warning("No error data available to plot")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(error_df, annot=True, cmap='YlOrRd', fmt='.3f', linewidths=0.5, ax=ax)
    ax.set_title('Error Rate by Subject across Different Modes')
    
    plt.tight_layout()
    fig.savefig(ERROR_ANALYSIS_CHARTS / 'error_distribution.png', dpi=300)
    plt.close(fig)
    logger.info('Error distribution plot generated')

def plot_timing_flamegraph(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Create a flamegraph-style visualization of timing breakdowns."""
    timing_data = []
    
    for mode, mode_data in data.items():
        for exp_type, df in mode_data.items():
            if df is not None and all(col in df.columns for col in ['duration', 'prompt_duration']):
                prompt_time = df['prompt_duration'].mean()
                response_time = (df['duration'] - df['prompt_duration']).mean()
                
                timing_data.extend([
                    {'mode': mode, 'component': 'Response Generation', 'time': response_time},
                    {'mode': mode, 'component': 'Prompt Processing', 'time': prompt_time}
                ])
    
    if not timing_data:
        logger.warning("No timing data available")
        return
    
    timing_df = pd.DataFrame(timing_data)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    timing_pivot = timing_df.pivot(index='mode', columns='component', values='time')
    timing_pivot.plot(kind='bar', stacked=True, ax=ax)
    
    total_times = timing_pivot.sum(axis=1)
    for i, total in enumerate(total_times):
        ax.text(i, total, f'{total:.2f}s', ha='center', va='bottom')
    
    ax.set_title('Processing Time Breakdown by Mode')
    ax.set_xlabel('Mode')
    ax.set_ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    fig.savefig(FLAMEGRAPH_CHARTS / 'timing_flamegraph.png', dpi=300)
    plt.close(fig)
    logger.info('Timing flamegraph generated')

def create_summary_report(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Create a comprehensive summary report."""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {},
            'mode_performance': {},
            'toxicity_analysis': {},
            'timing_analysis': {},
            'error_analysis': {}
        }

        # Add mode performance metrics
        for mode, mode_data in data.items():
            mode_metrics = {}

            if 'classification' in mode_data:
                df = mode_data['classification']
                mode_metrics.update({
                    'accuracy': df['correct'].mean() if 'correct' in df.columns else None,
                    'sample_count': len(df),
                    'avg_duration': df['duration'].mean() if 'duration' in df.columns else None,
                    'avg_token_count': df['token_count'].mean() if 'token_count' in df.columns else None
                })

            if 'toxicity' in mode_data:
                df = mode_data['toxicity']
                mode_metrics.update({
                    'toxicity_score': df['score'].mean() if 'score' in df.columns else None,
                    'toxic_responses': len(df[df['classification'] == 'toxic']) if 'classification' in df.columns else None
                })

            report['mode_performance'][mode] = mode_metrics

        # Print summary to console
        print("\n===== Mode Comparison Summary Report =====\n")
        
        # Print mode performance summary
        print("\nMode Performance Summary:")
        for mode, metrics in report['mode_performance'].items():
            print(f"\nMode: {mode}")
            for metric, value in metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")

        print("\n===== End of Summary Report =====")
        
    except Exception as e:
        logger.error(f"Failed to create summary report: {e}")
        raise

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
    fig.savefig(TOKEN_CHARTS / "token_count_comparison.png", dpi=300, bbox_inches='tight')
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
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy.png", dpi=300)
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
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy_by_subject.png", dpi=300, bbox_inches='tight')
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
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy_valid_only.png", dpi=300)
    plt.close(fig)
    logger.info("Classification accuracy (valid only) plot generated")

def plot_summarization_response_length(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot response length distribution for summarization across modes.
    
    Args:
        data: Dictionary containing mode data by experiment type
    """
    # Create figure with subplots (one per mode)
    fig, axes = plt.subplots(1, len(EXPERIMENT_MODES), figsize=(18, 6))
    
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
                    color=sns.color_palette()[i],
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
    fig.savefig(SUMMARIZATION_CHARTS / "summarization_response_length.png", dpi=300)
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
    fig.savefig(TOXICITY_CHARTS / "toxicity_rate.png", dpi=300)
    plt.close(fig)
    logger.info("Toxicity rate plot generated")

if __name__ == "__main__":
    run_aws_analysis() 