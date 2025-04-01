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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_analysis.log'),
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
CHART_DIR = Path('analysis/charts')
CLASSIFICATION_CHARTS = CHART_DIR / 'classification'
SUMMARIZATION_CHARTS = CHART_DIR / 'summarization'
TOXICITY_CHARTS = CHART_DIR / 'toxicity'
TOKEN_CHARTS = CHART_DIR / 'token_analysis'

def create_chart_directories():
    """Create all necessary chart directories."""
    for directory in [CLASSIFICATION_CHARTS, SUMMARIZATION_CHARTS, TOXICITY_CHARTS, TOKEN_CHARTS]:
        directory.mkdir(parents=True, exist_ok=True)

# Model paths and names
MODEL_DIRS = [
    "Meta-Llama-3-8B-Instruct.Q2_K.gguf",
    "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
]

# Experiment types
EXPERIMENT_TYPES = {
    "classification": "llama_classification.parquet",
    "summarization": "llama_summaries.parquet", 
    "toxicity": "llama3_7b.parquet_analysis.parquet"
}

def load_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all datasets for all models and experiments.
    
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Nested dictionary with 
            model and experiment type as keys, and dataframes as values
    """
    data = {}
    base_path = Path("./quantization_ablation_model")
    
    for model_dir in MODEL_DIRS:
        model_name = re.sub(r'Meta-Llama-3-8B-Instruct\.(.+)\.gguf', r'\1', model_dir)
        data[model_name] = {}
        
        for exp_type, file_name in EXPERIMENT_TYPES.items():
            file_path = base_path / model_dir / file_name
            try:
                df = pd.read_parquet(file_path)
                if exp_type == "classification":
                    df = add_subject_if_missing(df)
                data[model_name][exp_type] = df
                logger.info(f"Loaded {exp_type} data for {model_name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
                data[model_name][exp_type] = None
    
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

def analyze_token_counts(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze and visualize token counts for each model and experiment.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    logger.info("Analyzing token counts...")
    
    # Prepare data for token count analysis
    token_data = []
    
    for model_name, model_data in data.items():
        for exp_type, df in model_data.items():
            if df is not None and 'token_count' in df.columns and 'prompt_tokens' in df.columns:
                token_data.append({
                    "model": model_name,
                    "experiment": exp_type,
                    "prompt_tokens": df["prompt_tokens"].mean(),
                    "response_tokens": df["token_count"].mean(),
                    "tokenize_time": df["tokenize_duration"].mean(),
                    "prompt_time": df["prompt_duration"].mean(),
                    "response_time": (df["duration"] - df["prompt_duration"]).mean(),
                    "total_time": df["duration"].mean()
                })
    
    if not token_data:
        logger.warning("No token count data available")
        return
        
    token_df = pd.DataFrame(token_data)
    
    # Create token count comparison plot
    plot_token_count_comparison(token_df)
    
    # Create token distribution plots
    plot_token_distribution(data)
    
    # Create runtime distribution plots
    plot_runtime_distribution(data)
    
    # Create timing comparison plot
    plot_timing_comparison(data)

def plot_token_count_comparison(token_df: pd.DataFrame) -> None:
    """Plot token count comparison across models and experiments.
    
    Args:
        token_df: DataFrame with token count data
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up the bar positions
    models = token_df['model'].unique()
    experiments = token_df['experiment'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    # Plot bars for each experiment type
    for i, exp in enumerate(experiments):
        exp_data = token_df[token_df['experiment'] == exp]
        offset = (i - len(experiments)/2 + 0.5) * width
        
        # Plot prompt tokens
        ax.bar(x + offset, exp_data['prompt_tokens'], 
               width/2, label=f'{exp} Prompt', 
               color=f'C{i}', alpha=0.8)
        
        # Plot response tokens
        ax.bar(x + offset + width/2, exp_data['response_tokens'], 
               width/2, label=f'{exp} Response',
               color=f'C{i}', alpha=0.4)
    
    # Customize plot
    ax.set_title("Token Count Comparison by Model and Experiment")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Token Count")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOKEN_CHARTS / "token_count_comparison.png", dpi=300)
    plt.close(fig)
    logger.info("Token count comparison plot generated")

def plot_token_distribution(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot token distribution for each model and experiment.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Create distributions for each model and experiment
    for row, model_name in enumerate(data.keys()):
        for col, exp_type in enumerate(EXPERIMENT_TYPES.keys()):
            df = data[model_name].get(exp_type)
            if df is not None and "token_count" in df.columns:
                # Create histogram
                sns.histplot(
                    data=df,
                    x="token_count",
                    kde=True,
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
                axes[row, col].set_ylabel(model_name)
    
    # Add global title
    fig.suptitle("Token Count Distribution by Model and Experiment", fontsize=20)
    
    # Save plot
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(TOKEN_CHARTS / "token_distribution.png", dpi=300)
    plt.close(fig)
    logger.info("Token distribution plot generated")

def plot_runtime_distribution(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot runtime distribution for each experiment using PDFs.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define x-axis limits for each experiment type
    x_limits = {
        "classification": 2,      # Adjust based on your data
        "summarization": 4,       # Adjust based on your data
        "toxicity": 20           # Adjust based on your data
    }
    
    # Create PDF plots for each experiment type
    for col, exp_type in enumerate(EXPERIMENT_TYPES.keys()):
        plot_data = []
        
        # First pass to collect statistics
        for model_name, model_data in data.items():
            df = model_data.get(exp_type)
            if df is not None and "duration" in df.columns:
                durations = df["duration"].values
                plot_data.append({
                    'model': model_name,
                    'durations': durations[durations <= x_limits[exp_type]],  # Filter outliers
                    'median': np.median(durations),
                    'mean': np.mean(durations)
                })
        
        if not plot_data:
            continue
            
        # Plot distributions
        for item in plot_data:
            # Plot kernel density estimation (PDF)
            sns.kdeplot(data=item['durations'], 
                       ax=axes[col],
                       label=item['model'],
                       alpha=0.7)
            
            # Add median line
            axes[col].axvline(item['median'], 
                            color='gray', 
                            linestyle='--', 
                            alpha=0.3)
            
            # Add annotation at the top of the plot
            axes[col].text(0.98, 0.98 - plot_data.index(item) * 0.15,
                         f"{item['model']}: median={item['median']:.2f}s, mean={item['mean']:.2f}s",
                         transform=axes[col].transAxes,
                         ha='right',
                         va='top',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Customize each subplot
        axes[col].set_title(f"{exp_type.capitalize()} Runtime Distribution")
        axes[col].set_xlabel("Time (seconds)")
        axes[col].set_ylabel("Density")
        
        # Set axis limits
        axes[col].set_xlim(0, x_limits[exp_type])
        
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
        data: Dictionary containing model data by experiment type
    """
    # Collect timing data
    timing_data = []
    
    for model_name, model_data in data.items():
        for exp_type, df in model_data.items():
            if df is not None and "prompt_duration" in df.columns and "duration" in df.columns:
                # Calculate average timings
                prompt_time = float(df["prompt_duration"].mean())
                total_time = float(df["duration"].mean())
                tokenize_time = float(df["tokenize_duration"].mean())
                # Response time is total duration minus prompt duration and tokenize duration
                response_time = total_time - prompt_time - tokenize_time
                
                timing_data.append({
                    "model": model_name,
                    "experiment": exp_type,
                    "prompt_time": prompt_time,
                    "response_time": response_time,
                    "total_time": total_time,
                    "tokenize_time": tokenize_time
                })
                
                # Log timing information for debugging
                logger.info(f"{model_name} - {exp_type}:")
                logger.info(f"  Total time: {total_time:.3f}s")
                logger.info(f"  Prompt time: {prompt_time:.3f}s")
                logger.info(f"  Response time: {response_time:.3f}s")
                logger.info(f"  Tokenize time: {tokenize_time:.3f}s")
    
    if not timing_data:
        logger.warning("No timing data available")
        return
        
    timing_df = pd.DataFrame(timing_data)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up the bar positions
    models = timing_df['model'].unique()
    experiments = timing_df['experiment'].unique()
    x = np.arange(len(models))
    width = 0.25  # Make bars thinner to fit better
    
    # Plot bars for each experiment type
    for i, exp in enumerate(experiments):
        exp_data = timing_df[timing_df['experiment'] == exp]
        offset = (i - len(experiments)/2 + 0.5) * width
        
        # Convert Series to numpy arrays for plotting
        tokenize_times = exp_data['tokenize_time'].to_numpy()
        prompt_times = exp_data['prompt_time'].to_numpy()
        response_times = exp_data['response_time'].to_numpy()
        total_times = exp_data['total_time'].to_numpy()
        
        # Plot tokenize time
        tokenize_bars = ax.bar(x + offset, tokenize_times,
                             width, label=f'{exp} Tokenize',
                             color=f'C{i}', alpha=0.3)
        
        # Plot prompt time
        prompt_bars = ax.bar(x + offset, prompt_times,
                           width, bottom=tokenize_times,
                           label=f'{exp} Prompt',
                           color=f'C{i}', alpha=0.6)
        
        # Plot response time
        response_bars = ax.bar(x + offset, response_times,
                             width, 
                             bottom=tokenize_times + prompt_times,
                             label=f'{exp} Response',
                             color=f'C{i}', alpha=0.9)
        
        # Add time annotations
        def autolabel(bars, times, bottom=None):
            for j, (bar, time) in enumerate(zip(bars, times)):
                if time > 0:  # Only show non-zero times
                    height = time  # Use the actual time value
                    y_pos = height/2
                    if bottom is not None:
                        y_pos = bottom[j] + height/2
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                           f'{time:.3f}s',
                           ha='center', va='center')
        
        autolabel(tokenize_bars, tokenize_times)
        autolabel(prompt_bars, prompt_times, tokenize_times)
        autolabel(response_bars, response_times, tokenize_times + prompt_times)
        
        # Add total time annotation
        for j, total in enumerate(total_times):
            ax.text(x[j] + offset, total + 0.1,
                   f'Total: {total:.3f}s',
                   ha='center', va='bottom')
    
    # Customize plot
    ax.set_title("Timing Comparison by Model and Experiment")
    ax.set_xlabel("Model")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOKEN_CHARTS / "timing_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Timing comparison plot generated")

def analyze_classification_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze classification performance for all models.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    logger.info("Analyzing classification performance...")
    
    # Check if classification data exists
    has_classification_data = any(
        data[model].get("classification") is not None 
        for model in data
    )
    
    if not has_classification_data:
        logger.warning("No classification data available")
        return
    
    # 1. Overall accuracy comparison
    plot_classification_accuracy(data)
    
    # 2. Accuracy by subject
    plot_classification_accuracy_by_subject(data)
    
    # 3. Accuracy for valid responses only
    plot_classification_accuracy_valid_only(data)

def plot_classification_accuracy(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot overall classification accuracy for each model.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    # Collect accuracy data
    accuracies = []
    
    for model_name, model_data in data.items():
        if model_data.get("classification") is not None:
            df = model_data["classification"]
            accuracy = df["correct"].mean() if "correct" in df.columns else np.nan
            accuracies.append({
                "model": model_name,
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
        x="model",
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
    ax.set_title("Classification Accuracy by Model")
    ax.set_xlabel("Model")
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
    """Plot classification accuracy by subject for each model.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    # Collect accuracy data by subject
    accuracy_by_subject = {}
    
    for model_name, model_data in data.items():
        if model_data.get("classification") is not None:
            df = model_data["classification"]
            if "subject" in df.columns and "correct" in df.columns:
                grouped = df.groupby("subject")["correct"].agg(["mean", "count"]).reset_index()
                
                for _, row in grouped.iterrows():
                    subject = row["subject"]
                    if subject not in accuracy_by_subject:
                        accuracy_by_subject[subject] = {}
                    
                    accuracy_by_subject[subject][model_name] = {
                        "accuracy": row["mean"],
                        "count": row["count"]
                    }
    
    if not accuracy_by_subject:
        logger.warning("No accuracy by subject data available")
        return
    
    # Convert to DataFrame for heatmap
    subjects = list(accuracy_by_subject.keys())
    models = list(data.keys())
    
    heatmap_data = np.zeros((len(subjects), len(models)))
    
    for i, subject in enumerate(subjects):
        for j, model in enumerate(models):
            if model in accuracy_by_subject.get(subject, {}):
                heatmap_data[i, j] = accuracy_by_subject[subject][model]["accuracy"]
            else:
                heatmap_data[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(subjects) * 0.4)))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Accuracy"},
        xticklabels=models,
        yticklabels=subjects,
        ax=ax
    )
    
    # Customize plot
    ax.set_title("Classification Accuracy by Subject and Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Subject")
    
    # Save plot
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy_by_subject.png", dpi=300)
    plt.close(fig)
    logger.info("Classification accuracy by subject plot generated")

def plot_classification_accuracy_valid_only(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot classification accuracy considering only valid responses (A, B, C, D).
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    # Collect accuracy data for valid responses only
    accuracies = []
    
    for model_name, model_data in data.items():
        if model_data.get("classification") is not None:
            df = model_data["classification"]
            # Consider only responses that are single letters A, B, C, or D
            valid_responses = df["response"].str.strip().str.match('^[ABCD]$')
            valid_df = df[valid_responses]
            
            if len(valid_df) > 0:
                accuracy = valid_df["correct"].mean()
                accuracies.append({
                    "model": model_name,
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
        x="model",
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
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3)
    
    # Plot 2: Valid response rate
    sns.barplot(
        data=accuracy_df,
        x="model",
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
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Rate of Valid Responses (A,B,C,D)")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(CLASSIFICATION_CHARTS / "classification_accuracy_valid_only.png", dpi=300)
    plt.close(fig)
    logger.info("Classification accuracy (valid only) plot generated")

def analyze_summarization_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze summarization performance for all models.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    logger.info("Analyzing summarization performance...")
    
    # Check if summarization data exists
    has_summarization_data = any(
        data[model].get("summarization") is not None 
        for model in data
    )
    
    if not has_summarization_data:
        logger.warning("No summarization data available")
        return
    
    # 1. BERT score comparison
    plot_summarization_bert_scores(data)
    
    # 2. Response length distribution
    plot_summarization_response_length(data)

def plot_summarization_bert_scores(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot BERT scores for summarization across models.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    # Collect BERT score data
    bert_scores = []
    
    for model_name, model_data in data.items():
        if model_data.get("summarization") is not None:
            df = model_data["summarization"]
            if "score" in df.columns:
                bert_scores.append({
                    "model": model_name,
                    "avg_score": df["score"].mean(),
                    "median_score": df["score"].median(),
                    "min_score": df["score"].min(),
                    "max_score": df["score"].max()
                })
    
    if not bert_scores:
        logger.warning("No BERT score data available")
        return
    
    score_df = pd.DataFrame(bert_scores)
    
    # Create violin plot with boxplot overlay
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect all scores for violin plot
    all_scores = []
    
    for model_name, model_data in data.items():
        if model_data.get("summarization") is not None:
            df = model_data["summarization"]
            if "score" in df.columns:
                scores = df["score"].to_frame()
                scores["model"] = model_name
                all_scores.append(scores)
    
    if all_scores:
        all_scores_df = pd.concat(all_scores)
        
        # Create violin plot
        sns.violinplot(
            data=all_scores_df,
            x="model",
            y="score",
            palette="viridis",
            inner=None,
            ax=ax
        )
        
        # Overlay boxplot
        sns.boxplot(
            data=all_scores_df,
            x="model",
            y="score",
            color="white",
            width=0.3,
            boxprops=dict(alpha=0.7),
            ax=ax
        )
    
    # Add text annotations
    for i, row in score_df.iterrows():
        ax.text(
            i, 
            row["max_score"] + 0.01,
            f"Avg: {row['avg_score']:.3f}",
            ha="center",
            va="bottom"
        )
    
    # Customize plot
    ax.set_title("BERT Scores for Summarization by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("BERT Score")
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(SUMMARIZATION_CHARTS / "summarization_bert_scores.png", dpi=300)
    plt.close(fig)
    logger.info("Summarization BERT scores plot generated")

def plot_summarization_response_length(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot response length distribution for summarization across models.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    # Create figure with 3 subplots (one per model)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (model_name, model_data) in enumerate(data.items()):
        if model_data.get("summarization") is not None:
            df = model_data["summarization"]
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
                axes[i].set_title(f"{model_name} Response Length")
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
    fig.suptitle("Summarization Response Length Distribution by Model", fontsize=16)
    
    # Save plot
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(SUMMARIZATION_CHARTS / "summarization_response_length.png", dpi=300)
    plt.close(fig)
    logger.info("Summarization response length plot generated")

def analyze_toxicity_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze toxicity detection performance for all models.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    logger.info("Analyzing toxicity performance...")
    
    # Check if toxicity data exists
    has_toxicity_data = any(
        data[model].get("toxicity") is not None 
        for model in data
    )
    
    if not has_toxicity_data:
        logger.warning("No toxicity data available")
        return
    
    # 1. Toxicity rate comparison
    plot_toxicity_rate(data)
    
    # 2. Response length analysis
    plot_toxicity_response_length(data)
    
    # 3. Expected vs. actual toxic responses
    plot_toxicity_expected_vs_actual(data)

def plot_toxicity_rate(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot toxicity rate for each model.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    # Collect toxicity rate data
    toxicity_rates = []
    
    for model_name, model_data in data.items():
        if model_data.get("toxicity") is not None:
            df = model_data["toxicity"]
            if "toxic" in df.columns:
                toxic_rate = df["toxic"].mean()
            elif "confidence" in df.columns:
                # Assuming confidence > 0.5 means toxic
                toxic_rate = (df["confidence"] > 0.5).mean()
            else:
                toxic_rate = np.nan
            
            toxicity_rates.append({
                "model": model_name,
                "toxic_rate": toxic_rate,
                "total_samples": len(df)
            })
    
    if not toxicity_rates:
        logger.warning("No toxicity rate data available")
        return
    
    rate_df = pd.DataFrame(toxicity_rates)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=rate_df,
        x="model",
        y="toxic_rate",
        palette="viridis",
        ax=ax
    )
    
    # Add text annotations
    for i, row in rate_df.iterrows():
        ax.text(
            i, 
            row["toxic_rate"] + 0.01,
            f"{row['toxic_rate']:.3f}\n({int(row['toxic_rate'] * row['total_samples'])}/{row['total_samples']})",
            ha="center",
            va="bottom"
        )
    
    # Customize plot
    ax.set_title("Toxicity Rate by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Toxicity Rate")
    ax.set_ylim(0, max(rate_df["toxic_rate"]) * 1.2)  # Set y-axis limits for better visibility
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOXICITY_CHARTS / "toxicity_rate.png", dpi=300)
    plt.close(fig)
    logger.info("Toxicity rate plot generated")

def plot_toxicity_response_length(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot response length analysis for toxicity detection.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    # Create scatter plot of toxicity score vs response length
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (model_name, model_data) in enumerate(data.items()):
        if model_data.get("toxicity") is not None:
            df = model_data["toxicity"]
            if all(col in df.columns for col in ["token_count", "confidence"]):
                # Create scatter plot
                sns.scatterplot(
                    data=df,
                    x="token_count",
                    y="confidence",
                    label=model_name,
                    alpha=0.6,
                    ax=ax
                )
                
                # Add trend line
                sns.regplot(
                    data=df,
                    x="token_count",
                    y="confidence",
                    scatter=False,
                    label=f"{model_name} trend",
                    line_kws={"linestyle": "--"},
                    ax=ax
                )
    
    # Add horizontal line at confidence = 0.5 (typical threshold)
    ax.axhline(
        0.5,
        color="black",
        linestyle=":",
        label="Toxicity Threshold (0.5)"
    )
    
    # Customize plot
    ax.set_title("Toxicity Score vs. Response Length")
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Toxicity Confidence Score")
    ax.legend()
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOXICITY_CHARTS / "toxicity_response_length.png", dpi=300)
    plt.close(fig)
    logger.info("Toxicity response length plot generated")

def plot_toxicity_expected_vs_actual(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot how often toxic prompts lead to toxic responses.
    
    Args:
        data: Dictionary containing model data by experiment type
    """
    # Collect prompt toxicity vs response toxicity data
    toxicity_comparison = []
    
    for model_name, model_data in data.items():
        if model_data.get("toxicity") is not None:
            df = model_data["toxicity"]
            
            if "prompt_toxic" in df.columns and "response_toxic" in df.columns:
                # Calculate rate of toxic responses given toxic prompts
                toxic_prompts = df[df["prompt_toxic"] == True]
                if len(toxic_prompts) > 0:
                    toxic_response_rate = toxic_prompts["response_toxic"].mean()
                    total_toxic_prompts = len(toxic_prompts)
                else:
                    toxic_response_rate = np.nan
                    total_toxic_prompts = 0
                
                # Calculate rate of toxic responses given non-toxic prompts
                non_toxic_prompts = df[df["prompt_toxic"] == False]
                if len(non_toxic_prompts) > 0:
                    unexpected_toxic_rate = non_toxic_prompts["response_toxic"].mean()
                    total_non_toxic_prompts = len(non_toxic_prompts)
                else:
                    unexpected_toxic_rate = np.nan
                    total_non_toxic_prompts = 0
                
                toxicity_comparison.append({
                    "model": model_name,
                    "toxic_prompt_response_rate": toxic_response_rate,
                    "non_toxic_prompt_response_rate": unexpected_toxic_rate,
                    "total_toxic_prompts": total_toxic_prompts,
                    "total_non_toxic_prompts": total_non_toxic_prompts
                })
    
    if not toxicity_comparison:
        logger.warning("No toxicity comparison data available")
        return
    
    comparison_df = pd.DataFrame(toxicity_comparison)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    # Plot bars
    rects1 = ax.bar(x - width/2, comparison_df["toxic_prompt_response_rate"], 
                    width, label="Toxic Prompts → Toxic Responses",
                    color="#cc8963")
    rects2 = ax.bar(x + width/2, comparison_df["non_toxic_prompt_response_rate"], 
                    width, label="Non-Toxic Prompts → Toxic Responses",
                    color="#5975a4")
    
    # Add annotations
    def autolabel(rects, counts, total_counts):
        for i, (rect, count, total) in enumerate(zip(rects, counts, total_counts)):
            height = rect.get_height()
            if not np.isnan(height):
                ax.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.3f}\n({count}/{total})',
                        ha='center', va='bottom')
    
    autolabel(rects1, 
              comparison_df["total_toxic_prompts"] * comparison_df["toxic_prompt_response_rate"],
              comparison_df["total_toxic_prompts"])
    autolabel(rects2, 
              comparison_df["total_non_toxic_prompts"] * comparison_df["non_toxic_prompt_response_rate"],
              comparison_df["total_non_toxic_prompts"])
    
    # Customize plot
    ax.set_title("Toxicity Propagation Analysis")
    ax.set_xlabel("Model")
    ax.set_ylabel("Rate of Toxic Responses")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model"])
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add grid
    ax.grid(axis="y", alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    fig.savefig(TOXICITY_CHARTS / "toxicity_propagation.png", dpi=300)
    plt.close(fig)
    logger.info("Toxicity propagation plot generated")

def run_analysis() -> None:
    """Execute the complete analysis pipeline."""
    logger.info("Starting analysis pipeline...")
    
    # Create chart directories
    create_chart_directories()
    
    # Load all experiment data
    data = load_data()
    
    # Token and timing analysis
    analyze_token_counts(data)
    
    # Classification performance analysis
    analyze_classification_performance(data)
    
    # Summarization performance analysis
    analyze_summarization_performance(data)
    
    analyze_toxicity_performance(data)
    
    logger.info("Analysis pipeline completed successfully")

if __name__ == "__main__":
    run_analysis()
