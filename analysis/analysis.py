import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    base_path: str
    experiments: List[str]
    chart_dirs: Dict[str, str]
    plot_style: str = 'fivethirtyeight'
    figure_size: Tuple[int, int] = (12, 8)
    title_size: int = 16
    label_size: int = 14
    dpi: int = 300
    max_workers: int = 4

class AnalysisError(Exception):
    """Base class for analysis exceptions."""
    pass

class DataLoadError(AnalysisError):
    """Exception raised for errors during data loading."""
    pass

class VisualizationError(AnalysisError):
    """Exception raised for errors during visualization."""
    pass

def load_config(config_path: str = "analysis_config.yaml") -> AnalysisConfig:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return AnalysisConfig(**config_data)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Provide default configuration
        return AnalysisConfig(
            base_path="quantization_ablation_model",
            experiments=[
                "Meta-Llama-3-8B-Instruct.Q2_K.gguf",
                "Meta-Llama-3-8B-Instruct.Q4_K.gguf",
                "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
                "Meta-Llama-3-8B-Instruct.Q6_K.gguf",
                "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
                "Meta-Llama-3-8B-Instruct.Q8_K.gguf",
                "Meta-Llama-3-8B-Instruct.Q8_K_M.gguf"
            ],
            chart_dirs={
                'accuracy': 'charts/accuracy',
                'timing': 'charts/timing',
                'efficiency': 'charts/efficiency',
                'error_analysis': 'charts/error_analysis',
                'summarization': 'charts/summarization',
                'toxicity': 'charts/toxicity',
                'flamegraph': 'charts/flamegraph'
            }
        )

class AnalysisManager:
    """Manager class for coordinating analysis tasks."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.setup_plotting_style()
        self.create_chart_directories()
        self.experiment_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    def setup_plotting_style(self) -> None:
        """Configure matplotlib plotting style."""
        plt.style.use(self.config.plot_style)
        sns.set_context("notebook")
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['axes.titlesize'] = self.config.title_size
        plt.rcParams['axes.labelsize'] = self.config.label_size
    
    def create_chart_directories(self) -> None:
        """Create directories for saving charts."""
        try:
            for dir_path in self.config.chart_dirs.values():
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create chart directories: {e}")
            raise AnalysisError("Failed to create chart directories")

    def load_experiment_data(self) -> None:
        """Load experiment data using parallel processing."""
        logger.info("Starting data loading...")
        
        def load_single_experiment(exp: str) -> Tuple[str, Dict[str, pd.DataFrame]]:
            """Load data for a single experiment."""
            try:
                exp_path = Path(self.config.base_path) / exp
                if not exp_path.exists():
                    logger.warning(f"Directory does not exist: {exp_path}")
                    return exp, {}
                
                data = {}
                parquet_files = list(exp_path.glob("*.parquet"))
                
                for file_path in parquet_files:
                    try:
                        df = pd.read_parquet(file_path)
                        if file_path.name.endswith('_analysis.parquet'):
                            data['llama3_7b'] = df
                        else:
                            file_type = file_path.stem
                            if file_type == 'llama_classification':
                                df = self.add_subject_if_missing(df)
                            data[file_type] = df
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
                
                return exp, data
            except Exception as e:
                logger.error(f"Error processing experiment {exp}: {e}")
                return exp, {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(load_single_experiment, self.config.experiments))
        
        self.experiment_data = dict(results)
        logger.info("Data loading completed")

    @staticmethod
    def add_subject_if_missing(df: pd.DataFrame, input_data_path: str = "./input_datasets/classification_pairs.parquet") -> pd.DataFrame:
        """Add subject column to the dataframe if missing."""
        if 'subject' not in df.columns:
            try:
                input_data = pd.read_parquet(input_data_path)
                question_to_subject = dict(zip(input_data['question'], input_data['subject']))
                df['subject'] = df['question'].map(question_to_subject)
            except Exception as e:
                logger.warning(f"Could not add subject information: {e}")
                df['subject'] = 'unknown'
        return df

    def save_plot(self, fig: plt.Figure, category: str, name: str) -> None:
        """Save plot with standardized naming and error handling."""
        try:
            output_path = Path(self.config.chart_dirs[category]) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(output_path, bbox_inches='tight', dpi=self.config.dpi)
            logger.info(f"Saved plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {name}: {e}")
            raise VisualizationError(f"Failed to save plot {name}")

    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        try:
            logger.info("Starting analysis pipeline")
            self.load_experiment_data()
            
            # Generate and save all plots
            self.generate_accuracy_plots()
            self.generate_timing_plots()
            self.generate_efficiency_plots()
            self.generate_error_analysis()
            self.generate_summarization_analysis()
            self.generate_toxicity_analysis()
            
            # Create summary report
            self.create_summary_report()
            
            logger.info("Analysis pipeline completed successfully")
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise AnalysisError("Analysis pipeline failed")

    def generate_accuracy_plots(self) -> None:
        """Generate all accuracy-related plots."""
        try:
            # Generate accuracy heatmap
            fig, ax = self.generate_accuracy_heatmap()
            self.save_plot(fig, 'accuracy', 'accuracy_heatmap')
            plt.close(fig)

            # Generate overall accuracy comparison
            fig, ax = self.plot_overall_accuracy_comparison()
            self.save_plot(fig, 'accuracy', 'overall_accuracy')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to generate accuracy plots: {e}")
            raise VisualizationError("Failed to generate accuracy plots")

    def generate_timing_plots(self) -> None:
        """Generate all timing-related plots."""
        try:
            # Generate timing distributions
            fig, axs = self.plot_timing_distributions()
            self.save_plot(fig, 'timing', 'timing_distributions')
            plt.close(fig)

            # Generate timing boxplots
            fig, ax = self.plot_timing_boxplots()
            self.save_plot(fig, 'timing', 'timing_boxplots')
            plt.close(fig)

            # Generate timing vs token count
            fig, axs = self.plot_timing_vs_token_count()
            self.save_plot(fig, 'timing', 'timing_vs_tokens')
            plt.close(fig)

            # Generate flamegraph
            fig, ax = self.plot_timing_flamegraph()
            self.save_plot(fig, 'flamegraph', 'timing_flamegraph')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to generate timing plots: {e}")
            raise VisualizationError("Failed to generate timing plots")

    def generate_efficiency_plots(self) -> None:
        """Generate all efficiency-related plots."""
        try:
            fig, ax = self.plot_token_efficiency()
            self.save_plot(fig, 'efficiency', 'token_efficiency')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to generate efficiency plots: {e}")
            raise VisualizationError("Failed to generate efficiency plots")

    def generate_error_analysis(self) -> None:
        """Generate error analysis plots."""
        try:
            fig, ax = self.plot_error_distribution()
            self.save_plot(fig, 'error_analysis', 'error_distribution')
            plt.close(fig)

            confusion_matrices = self.plot_confusion_matrix()
            for model_name, (fig, ax) in confusion_matrices.items():
                self.save_plot(fig, 'error_analysis', f'confusion_matrix_{model_name}')
                plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to generate error analysis plots: {e}")
            raise VisualizationError("Failed to generate error analysis plots")

    def generate_summarization_analysis(self) -> None:
        """Generate summarization analysis plots."""
        try:
            plots = self.analyze_summarization_data()
            for plot_name, (fig, _) in plots.items():
                self.save_plot(fig, 'summarization', plot_name)
                plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to generate summarization plots: {e}")
            raise VisualizationError("Failed to generate summarization plots")

    def generate_toxicity_analysis(self) -> None:
        """Generate toxicity analysis plots."""
        try:
            fig, axes = self.analyze_toxicity()
            self.save_plot(fig, 'toxicity', 'toxicity_analysis')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to generate toxicity plots: {e}")
            raise VisualizationError("Failed to generate toxicity plots")

    def create_summary_report(self) -> None:
        """Create a comprehensive summary report."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'analysis_summary': {},
                'model_performance': {},
                'toxicity_analysis': {},
                'timing_analysis': {},
                'error_analysis': {}
            }

            # Add model performance metrics
            for exp_name, exp_data in self.experiment_data.items():
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                model_metrics = {}

                if 'llama_classification' in exp_data:
                    df = exp_data['llama_classification']
                    model_metrics.update({
                        'accuracy': df['correct'].mean(),
                        'sample_count': len(df),
                        'avg_duration': df['duration'].mean(),
                        'avg_token_count': df['token_count'].mean()
                    })

                if 'llama3_7b' in exp_data:
                    df = exp_data['llama3_7b']
                    model_metrics.update({
                        'toxicity_score': df['score'].mean() if 'score' in df.columns else None,
                        'toxic_responses': len(df[df['classification'] == 'toxic']) if 'classification' in df.columns else None
                    })

                report['model_performance'][model_name] = model_metrics

            # Print summary to console
            self._print_summary_report(report)
        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
            raise AnalysisError("Failed to create summary report")

    def _print_summary_report(self, report: Dict[str, Any]) -> None:
        """Print a formatted summary report to console."""
        print("\n===== Model Comparison Summary Report =====\n")
        
        # Print model performance summary
        print("\nModel Performance Summary:")
        for model, metrics in report['model_performance'].items():
            print(f"\nModel: {model}")
            for metric, value in metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")

        print("\n===== End of Summary Report =====")

    def generate_accuracy_heatmap(self) -> Tuple[plt.Figure, plt.Axes]:
        """Generate accuracy heatmap by subject across different models."""
        has_classification_data = any('llama_classification' in exp_data for exp_data in self.experiment_data.values())
        
        if not has_classification_data:
            logger.warning("No classification data available")
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.text(0.5, 0.5, 'No classification data available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Accuracy by Subject across Different Models')
            return fig, ax
        
        accuracy_data = {}
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama_classification' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama_classification']
                
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
                        
                    accuracy_data[subject][model_name] = accuracy
        
        # Convert to DataFrame for heatmap
        heatmap_df = pd.DataFrame(accuracy_data).T
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.3f', 
                    linewidths=0.5, ax=ax, vmin=0, vmax=1)
        ax.set_title('Accuracy by Subject across Different Models')
        
        return fig, ax

    def plot_overall_accuracy_comparison(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot overall accuracy for each model."""
        model_names = []
        accuracies = []
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama_classification' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama_classification']
                
                accuracy = df['correct'].mean()
                model_names.append(model_name)
                accuracies.append(accuracy)
        
        if not model_names:
            logger.warning("No classification data available")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No classification data available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overall Accuracy by Model')
            return fig, ax
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=model_names, y=accuracies, ax=ax, palette='deep')
        
        ax.set_title('Overall Accuracy by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
        
        ax.set_ylim(0, 1.0)
        plt.xticks(rotation=45)
        
        return fig, ax

    def plot_timing_distributions(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot timing distributions for each model."""
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        timing_cols = ['duration', 'prompt_duration', 'tokenize_duration']
        timing_labels = ['Total Duration', 'Prompt Processing Duration', 'Tokenization Duration']
        
        # Define a color palette for different quantization levels
        quantization_palettes = {
            'Q2': 'Blues',
            'Q4': 'Greens',
            'Q6': 'Reds',
            'Q8': 'Purples'
        }
        
        for i, (col, label) in enumerate(zip(timing_cols, timing_labels)):
            # Create violin plot for distribution
            sns.violinplot(data=self.experiment_data, x='model', y=col, ax=axs[i], alpha=0.3)
            
            # Add box plot for quartiles
            sns.boxplot(data=self.experiment_data, x='model', y=col, ax=axs[i], width=0.2)
            
            # Add individual points with jitter
            sns.stripplot(data=self.experiment_data, x='model', y=col, ax=axs[i], 
                         alpha=0.2, jitter=0.2, size=4)
            
            # Calculate and add mean lines
            means = self.experiment_data.groupby('model')[col].mean()
            for j, (model, mean) in enumerate(means.items()):
                axs[i].axhline(y=mean, xmin=j/len(means)-0.1, xmax=j/len(means)+0.1, 
                             color='red', linestyle='--', alpha=0.5)
                axs[i].text(j, mean, f'μ={mean:.2f}s', ha='center', va='bottom')
            
            axs[i].set_title(f'{label} Distribution')
            axs[i].set_xlabel('Model')
            axs[i].set_ylabel('Time (seconds)')
            axs[i].tick_params(axis='x', rotation=45)
            
            # Add grid for better readability
            axs[i].grid(True, alpha=0.3)
            
            # Add legend
            handles = [
                plt.Line2D([0], [0], color='red', linestyle='--', label='Mean'),
                plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', 
                             label='Quartiles'),
                plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, 
                             label='Distribution')
            ]
            axs[i].legend(handles=handles, loc='upper right')
        
        plt.tight_layout()
        return fig, axs

    def plot_timing_boxplots(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot timing boxplots for each model."""
        timing_data = []
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama_classification' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama_classification']
                
                temp_df = df[['duration', 'prompt_duration', 'tokenize_duration']].copy()
                temp_df['model'] = model_name
                timing_data.append(temp_df)
        
        if not timing_data:
            logger.warning("No timing data available")
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, 'No timing data available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Timing Comparison across Models')
            return fig, ax
        
        all_timing = pd.concat(timing_data)
        melted = all_timing.melt(id_vars=['model'], 
                                value_vars=['duration', 'prompt_duration', 'tokenize_duration'],
                                var_name='metric', value_name='time')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(data=melted, x='metric', y='time', hue='model', ax=ax)
        
        ax.set_title('Timing Comparison across Models')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Time (seconds)')
        
        plt.xticks(
            [0, 1, 2],
            ['Total Duration', 'Prompt Processing', 'Tokenization']
        )
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        return fig, ax

    def plot_timing_vs_token_count(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot timing vs token count for each model."""
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        metrics = ['duration', 'prompt_duration', 'tokenize_duration']
        titles = ['Total Duration vs Token Count', 
                'Prompt Processing Time vs Token Count', 
                'Tokenization Time vs Token Count']
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama_classification' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama_classification']
                
                for i, (metric, title) in enumerate(zip(metrics, titles)):
                    axs[i].scatter(df['token_count'], df[metric], alpha=0.5, label=model_name)
                    
                    # Add trend line
                    z = np.polyfit(df['token_count'], df[metric], 1)
                    p = np.poly1d(z)
                    axs[i].plot(sorted(df['token_count']), p(sorted(df['token_count'])), 
                              linestyle='--', alpha=0.8)
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            axs[i].set_title(title)
            axs[i].set_xlabel('Token Count')
            axs[i].set_ylabel('Time (seconds)')
            axs[i].legend()
        
        plt.tight_layout()
        return fig, axs

    def plot_timing_flamegraph(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create a flamegraph-style visualization of timing breakdowns."""
        timing_data = []
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama_classification' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama_classification']
                
                prompt_time = df['prompt_duration'].mean()
                response_time = (df['duration'] - df['prompt_duration']).mean()
                
                timing_data.extend([
                    {'model': model_name, 'component': 'Response Generation', 'time': response_time},
                    {'model': model_name, 'component': 'Prompt Processing', 'time': prompt_time}
                ])
        
        if not timing_data:
            logger.warning("No timing data available")
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.text(0.5, 0.5, 'No timing data available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Processing Time Breakdown by Model')
            return fig, ax
        
        timing_df = pd.DataFrame(timing_data)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        timing_pivot = timing_df.pivot(index='model', columns='component', values='time')
        timing_pivot.plot(kind='bar', stacked=True, ax=ax)
        
        total_times = timing_pivot.sum(axis=1)
        for i, total in enumerate(total_times):
            ax.text(i, total, f'{total:.2f}s', ha='center', va='bottom')
        
        ax.set_title('Processing Time Breakdown by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        return fig, ax

    def plot_token_efficiency(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot token processing efficiency (tokens per second) for each model."""
        efficiency_data = []
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama_classification' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama_classification']
                
                # Calculate efficiency metrics
                df['prompt_tokens_per_sec'] = df['prompt_tokens'] / df['prompt_duration']
                df['response_duration'] = df['duration'] - df['prompt_duration'] - df['tokenize_duration']
                df['response_tokens'] = df['token_count'] - df['prompt_tokens']
                df['response_tokens_per_sec'] = df['response_tokens'] / df['response_duration']
                
                # Calculate statistics
                prompt_stats = {
                    'model': model_name,
                    'phase': 'Prompt',
                    'mean': df['prompt_tokens_per_sec'].mean(),
                    'std': df['prompt_tokens_per_sec'].std(),
                    'median': df['prompt_tokens_per_sec'].median(),
                    'q1': df['prompt_tokens_per_sec'].quantile(0.25),
                    'q3': df['prompt_tokens_per_sec'].quantile(0.75)
                }
                
                response_stats = {
                    'model': model_name,
                    'phase': 'Response',
                    'mean': df['response_tokens_per_sec'].mean(),
                    'std': df['response_tokens_per_sec'].std(),
                    'median': df['response_tokens_per_sec'].median(),
                    'q1': df['response_tokens_per_sec'].quantile(0.25),
                    'q3': df['response_tokens_per_sec'].quantile(0.75)
                }   
                
                efficiency_data.extend([prompt_stats, response_stats])
        
        if not efficiency_data:
            logger.warning("No efficiency data available")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No efficiency data available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Token Processing Efficiency by Model')
            return fig, ax
        
        efficiency_df = pd.DataFrame(efficiency_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot 1: Bar plot with error bars
        sns.barplot(data=efficiency_df, x='model', y='mean', hue='phase', ax=ax1)
        
        # Add error bars
        for i, row in efficiency_df.iterrows():
            ax1.errorbar(i, row['mean'], yerr=row['std'], 
                        fmt='none', color='black', capsize=5)
        
        ax1.set_title('Token Processing Efficiency by Model')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Tokens per Second')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Box plot for distribution
        sns.boxplot(data=efficiency_df, x='model', y='mean', hue='phase', ax=ax2)
        ax2.set_title('Token Processing Efficiency Distribution')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Tokens per Second')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add grid for better readability
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add sample size annotations
        for phase in ['Prompt', 'Response']:
            phase_data = efficiency_df[efficiency_df['phase'] == phase]
            for i, row in phase_data.iterrows():
                ax1.text(i, row['mean'], 
                        f'μ={row["mean"]:.1f}\nσ={row["std"]:.1f}', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_error_distribution(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot distribution of errors by subject."""
        error_data = {}
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama_classification' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama_classification']
                
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
                    
                    error_data[subject][model_name] = error_rate
        
        # Convert to DataFrame
        error_df = pd.DataFrame(error_data).T
        
        # Handle empty data case
        if error_df.empty:
            logger.warning("No error data available to plot")
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.text(0.5, 0.5, 'No error data available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Distribution by Subject')
            return fig, ax
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(error_df, annot=True, cmap='YlOrRd', fmt='.3f', linewidths=0.5, ax=ax)
        ax.set_title('Error Rate by Subject across Different Models')
        
        return fig, ax

    def plot_confusion_matrix(self) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """Plot confusion matrices for each model."""
        confusion_matrices = {}
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama_classification' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama_classification']
                
                if 'response' in df.columns and 'expected' in df.columns:
                    # Get unique responses
                    unique_responses = sorted(set(df['response'].unique()) | set(df['expected'].unique()))
                    
                    # Create confusion matrix
                    conf_matrix = pd.DataFrame(0, index=unique_responses, columns=unique_responses)
                    
                    # Fill confusion matrix
                    for _, row in df.iterrows():
                        conf_matrix.loc[row['expected'], row['response']] += 1
                    
                    # Normalize by row
                    row_sums = conf_matrix.sum(axis=1)
                    norm_conf_matrix = conf_matrix.div(row_sums, axis=0).fillna(0)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(norm_conf_matrix, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5, ax=ax)
                    
                    ax.set_title(f'Normalized Confusion Matrix - {model_name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    
                    confusion_matrices[model_name] = (fig, ax)
        
        return confusion_matrices

    def analyze_toxicity(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Analyze and visualize toxicity scores across models."""
        has_toxicity_data = any('llama3_7b' in exp_data for exp_data in self.experiment_data.values())
        
        if not has_toxicity_data:
            logger.warning("No toxicity data available")
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No toxicity data available', 
                        ha='center', va='center', transform=ax.transAxes)
            axes[0, 0].set_title('Toxicity Score Distribution')
            axes[0, 1].set_title('Toxicity Rate by Model')
            axes[1, 0].set_title('Toxicity Reduction Analysis')
            axes[1, 1].set_title('Toxicity Score vs Response Length')
            return fig, axes
        
        toxicity_data = []
        model_stats = []
        
        logger.info("Processing toxicity data...")
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'llama3_7b' in exp_data:
                model_name = exp_name.replace('.gguf', '').replace('Meta-Llama-3-8B-Instruct.', '')
                df = exp_data['llama3_7b']
                
                # Check required columns
                required_columns = ['score', 'classification', 'expected', 'duration']
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"Missing required columns in {model_name}. Found: {list(df.columns)}")
                    continue
                
                # Calculate basic statistics
                toxic_count = len(df[df['classification'] == 'toxic'])
                total = len(df)
                toxic_rate = toxic_count / total if total > 0 else 0
                
                # Calculate toxicity reduction metrics
                toxic_inputs = df[df['expected'] == 'toxic']
                toxic_outputs = df[df['classification'] == 'toxic']
                reduction_rate = len(toxic_inputs[toxic_inputs['classification'] == 'non-toxic']) / len(toxic_inputs) if len(toxic_inputs) > 0 else 0
                
                model_stats.append({
                    'model': model_name,
                    'toxic_rate': toxic_rate,
                    'reduction_rate': reduction_rate,
                    'avg_score': df['score'].mean(),
                    'total_samples': total
                })
                
                # Add individual scores for distribution plot
                df['model'] = model_name
                toxicity_data.append(df[['model', 'score', 'duration', 'expected', 'classification']])
        
        if not toxicity_data:
            logger.warning("No valid toxicity data found")
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No valid toxicity data available', 
                        ha='center', va='center', transform=ax.transAxes)
            return fig, axes
        
        # Combine all data
        all_toxicity = pd.concat(toxicity_data)
        stats_df = pd.DataFrame(model_stats)
        
        # Create multiple plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Toxicity Score Distribution
        sns.boxplot(data=all_toxicity, x='model', y='score', ax=axes[0, 0])
        axes[0, 0].set_title('Toxicity Score Distribution')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Toxicity Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add violin plot overlay for better distribution visualization
        sns.violinplot(data=all_toxicity, x='model', y='score', ax=axes[0, 0], alpha=0.3)
        
        # Plot 2: Toxicity Rate by Model
        sns.barplot(data=stats_df, x='model', y='toxic_rate', ax=axes[0, 1])
        axes[0, 1].set_title('Toxicity Rate by Model')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Toxicity Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add sample size annotations
        for i, row in stats_df.iterrows():
            axes[0, 1].text(i, row['toxic_rate'], 
                          f"n={row['total_samples']}", 
                          ha='center', va='bottom')
        
        # Plot 3: Toxicity Reduction Analysis
        reduction_data = pd.melt(stats_df, 
                               id_vars=['model'],
                               value_vars=['toxic_rate', 'reduction_rate'],
                               var_name='metric',
                               value_name='rate')
        
        sns.barplot(data=reduction_data, x='model', y='rate', hue='metric', ax=axes[1, 0])
        axes[1, 0].set_title('Toxicity Reduction Analysis')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Toxicity Score vs Response Length
        sns.scatterplot(data=all_toxicity, x='duration', y='score', hue='model', alpha=0.5, ax=axes[1, 1])
        axes[1, 1].set_title('Toxicity Score vs Response Length')
        axes[1, 1].set_xlabel('Response Duration (seconds)')
        axes[1, 1].set_ylabel('Toxicity Score')
        
        # Add trend lines for each model
        for model in all_toxicity['model'].unique():
            model_data = all_toxicity[all_toxicity['model'] == model]
            z = np.polyfit(model_data['duration'], model_data['score'], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(model_data['duration'], p(model_data['duration']), 
                           linestyle='--', alpha=0.8)
        
        plt.tight_layout()
        return fig, axes

def main():
    """Main entry point for analysis."""
    try:
        # Create default configuration
        default_config = AnalysisConfig(
            base_path="quantization_ablation_model",
            experiments=[
                "Meta-Llama-3-8B-Instruct.Q2_K.gguf",
                "Meta-Llama-3-8B-Instruct.Q4_K.gguf",
                "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
                "Meta-Llama-3-8B-Instruct.Q6_K.gguf",
                "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
                "Meta-Llama-3-8B-Instruct.Q8_K.gguf",
                "Meta-Llama-3-8B-Instruct.Q8_K_M.gguf"
            ],
            chart_dirs={
                'accuracy': 'charts/accuracy',
                'timing': 'charts/timing',
                'efficiency': 'charts/efficiency',
                'error_analysis': 'charts/error_analysis',
                'summarization': 'charts/summarization',
                'toxicity': 'charts/toxicity',
                'flamegraph': 'charts/flamegraph'
            }
        )
        
        # Initialize analyzer with default configuration
        analyzer = AnalysisManager(config=default_config)
        
        # Run analysis pipeline
        logger.info("Starting analysis pipeline")
        analyzer.load_experiment_data()
        
        # Generate and save all plots
        analyzer.generate_accuracy_plots()
        analyzer.generate_timing_plots()
        analyzer.generate_efficiency_plots()
        analyzer.generate_error_analysis()
        analyzer.generate_summarization_analysis()
        analyzer.generate_toxicity_analysis()
        
        # Create summary report
        analyzer.create_summary_report()
        
        logger.info("Analysis pipeline completed successfully")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
