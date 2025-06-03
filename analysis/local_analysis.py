from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
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
OUTPUT_DIR = Path(f'local-analysis-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def setup_plot(ax: plt.Axes, xlabel: str, ylabel: str, ylim: Optional[Tuple[float, float]] = None) -> None:
    """Configure common plot settings."""
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(axis="y", alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)

def save_plot(fig: plt.Figure, filename: str) -> None:
    """Save plot with consistent settings."""
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf", dpi=300)
    plt.close(fig)
    logger.info(f"{filename} plot generated")

def plot_bar_with_annotations(
    data: pd.DataFrame,
    x: str,
    y: str,
    annotations: List[Tuple[str, str, str]],
    title: str,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """Create a bar plot with annotations.
    
    Args:
        data: DataFrame containing the data to plot
        x: Column name for x-axis
        y: Column name for y-axis
        annotations: List of tuples (value_col, count_col, total_col) for annotations
        title: Plot title
        ylim: Optional y-axis limits
    """
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
    
    setup_plot(ax, "Model", title, ylim)
    save_plot(fig, title.lower().replace(" ", "_"))

def load_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all datasets for all models and experiments."""
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
            question_to_subject = dict(zip(input_data['question'], input_data['subject']))
            df['subject'] = df['question'].map(question_to_subject)
            logger.info(f"Successfully added subject mapping to {len(df)} rows")
        except Exception as e:
            logger.warning(f"Could not add subject information: {e}")
            df['subject'] = 'unknown'
    return df

def analyze_classification_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze classification performance for all models."""
    logger.info("Analyzing classification performance...")
    
    if not any(data[model].get("classification") is not None for model in data):
        logger.warning("No classification data available")
        return
    
    # Overall accuracy
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
    
    if accuracies:
        plot_bar_with_annotations(
            pd.DataFrame(accuracies),
            "model",
            "accuracy",
            [("accuracy", "correct", "total")],
            "Classification Accuracy",
            (0, 1.1)
        )
    
    # Valid responses only
    valid_accuracies = []
    for model_name, model_data in data.items():
        if model_data.get("classification") is not None:
            df = model_data["classification"]
            valid_responses = df["response"].str.strip().str.match('^[ABCD]$')
            valid_df = df[valid_responses]
            
            if len(valid_df) > 0:
                valid_accuracies.append({
                    "model": model_name,
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
        sns.barplot(data=valid_df, x="model", y="accuracy", hue="model", palette="viridis", ax=ax1, legend=False)
        for i, row in valid_df.iterrows():
            ax1.text(i, row["accuracy"] + 0.01,
                    f"{row['accuracy']:.3f}\n({row['correct']}/{row['total_valid']})",
                    ha="center", va="bottom", fontsize=14)
        setup_plot(ax1, "Model", "Accuracy", (0, 1.1))
        
        # Valid rate plot
        sns.barplot(data=valid_df, x="model", y="valid_rate", hue="model", palette="viridis", ax=ax2, legend=False)
        for i, row in valid_df.iterrows():
            ax2.text(i, row["valid_rate"] + 0.01,
                    f"{row['valid_rate']:.3f}\n({row['total_valid']}/{row['total_all']})",
                    ha="center", va="bottom", fontsize=14)
        setup_plot(ax2, "Model", "Rate of Valid Responses", (0, 1.1))
        
        save_plot(fig, "classification_accuracy_valid_only")

def analyze_summarization_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze summarization performance for all models."""
    logger.info("Analyzing summarization performance...")
    
    if not any(data[model].get("summarization") is not None for model in data):
        logger.warning("No summarization data available")
        return
    
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
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.violinplot(data=all_scores_df, x="model", y="score", palette="viridis", inner=None, ax=ax)
        sns.boxplot(data=all_scores_df, x="model", y="score", color="white", width=0.3,
                   boxprops=dict(alpha=0.7), ax=ax)
        
        setup_plot(ax, "Model", "BERT Score")
        save_plot(fig, "summarization_bert_scores")

def analyze_toxicity_performance(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Analyze toxicity detection performance for all models."""
    logger.info("Analyzing toxicity performance...")
    
    if not any(data[model].get("toxicity") is not None for model in data):
        logger.warning("No toxicity data available")
        return
    
    # Toxicity rate
    toxicity_rates = []
    for model_name, model_data in data.items():
        if model_data.get("toxicity") is not None:
            df = model_data["toxicity"]
            if "toxic" in df.columns:
                toxic_rate = df["toxic"].mean()
            elif "confidence" in df.columns:
                toxic_rate = (df["confidence"] > 0.5).mean()
            else:
                toxic_rate = np.nan
            
            toxicity_rates.append({
                "model": model_name,
                "toxic_rate": toxic_rate,
                "toxic_count": int(toxic_rate * len(df)),
                "total_samples": len(df)
            })
    
    if toxicity_rates:
        plot_bar_with_annotations(
            pd.DataFrame(toxicity_rates),
            "model",
            "toxic_rate",
            [("toxic_rate", "toxic_count", "total_samples")],
            "Toxicity Rate",
            (0, max(pd.DataFrame(toxicity_rates)["toxic_rate"]) * 1.2)
        )
    
    # Toxicity propagation
    toxicity_comparison = []
    for model_name, model_data in data.items():
        if model_data.get("toxicity") is not None:
            df = model_data["toxicity"]
            if "prompt_toxic" in df.columns and "response_toxic" in df.columns:
                toxic_prompts = df[df["prompt_toxic"] == True]
                non_toxic_prompts = df[df["prompt_toxic"] == False]
                
                toxic_response_rate = toxic_prompts["response_toxic"].mean() if len(toxic_prompts) > 0 else np.nan
                unexpected_toxic_rate = non_toxic_prompts["response_toxic"].mean() if len(non_toxic_prompts) > 0 else np.nan
                
                toxicity_comparison.append({
                    "model": model_name,
                    "toxic_prompt_response_rate": toxic_response_rate,
                    "non_toxic_prompt_response_rate": unexpected_toxic_rate,
                    "toxic_prompt_count": int(toxic_response_rate * len(toxic_prompts)) if not np.isnan(toxic_response_rate) else 0,
                    "non_toxic_prompt_count": int(unexpected_toxic_rate * len(non_toxic_prompts)) if not np.isnan(unexpected_toxic_rate) else 0,
                    "total_toxic_prompts": len(toxic_prompts),
                    "total_non_toxic_prompts": len(non_toxic_prompts)
                })
    
    if toxicity_comparison:
        comparison_df = pd.DataFrame(toxicity_comparison)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, comparison_df["toxic_prompt_response_rate"], 
                        width, label="Toxic Prompts → Toxic Responses",
                        color="#cc8963")
        rects2 = ax.bar(x + width/2, comparison_df["non_toxic_prompt_response_rate"], 
                        width, label="Non-Toxic Prompts → Toxic Responses",
                        color="#5975a4")
        
        def autolabel(rects, counts, total_counts):
            for i, (rect, count, total) in enumerate(zip(rects, counts, total_counts)):
                height = rect.get_height()
                if not np.isnan(height):
                    ax.text(rect.get_x() + rect.get_width()/2., height,
                            f'{height:.3f}\n({count}/{total})',
                            ha='center', va='bottom', fontsize=14)
        
        autolabel(rects1, 
                  comparison_df["toxic_prompt_count"],
                  comparison_df["total_toxic_prompts"])
        autolabel(rects2, 
                  comparison_df["non_toxic_prompt_count"],
                  comparison_df["total_non_toxic_prompts"])
        
        setup_plot(ax, "Model", "Rate of Toxic Responses", (0, 1.1))
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["model"])
        ax.legend(fontsize=14)
        
        save_plot(fig, "toxicity_propagation")

def run_analysis() -> None:
    """Execute the complete analysis pipeline."""
    logger.info("Starting analysis pipeline...")
    
    data = load_data()
    analyze_classification_performance(data)
    analyze_summarization_performance(data)
    analyze_toxicity_performance(data)
    
    logger.info("Analysis pipeline completed successfully")

if __name__ == "__main__":
    run_analysis()