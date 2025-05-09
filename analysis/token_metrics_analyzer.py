from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_parquet_file(file_path: Path) -> pd.DataFrame:
    """
    Load a parquet file into a pandas DataFrame.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        DataFrame containing the parquet data
    """
    return pd.read_parquet(file_path)


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate total metrics from the DataFrame.
    
    Args:
        df: DataFrame containing token metrics
        
    Returns:
        Dictionary containing summed metrics
    """
    total_duration = df["duration"].sum()
    total_token_count = df["token_count"].sum()
    total_prompt_duration = df["prompt_duration"].sum()
    total_prompt_tokens = df["prompt_tokens"].sum()
    
    metrics = {
        "total_duration": total_duration,
        "total_token_count": total_token_count,
        "total_tokenize_duration": df["tokenize_duration"].sum(),
        "total_prompt_duration": total_prompt_duration,
        "total_prompt_tokens": total_prompt_tokens,
        "tokens_per_second": total_token_count / total_duration if total_duration > 0 else 0,
        "prompt_tokens_per_second": total_prompt_tokens / total_prompt_duration if total_prompt_duration > 0 else 0
    }
    return metrics


def analyze_parquet_files(file_paths: List[Path]) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Analyze multiple parquet files and return combined metrics.
    
    Args:
        file_paths: List of paths to parquet files
        
    Returns:
        Tuple containing (combined metrics, combined DataFrame)
    """
    dfs = []
    for file_path in file_paths:
        df = load_parquet_file(file_path)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    metrics = calculate_metrics(combined_df)
    
    return metrics, combined_df


def main() -> None:
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze token metrics from parquet files")
    parser.add_argument("--files", nargs="+", help="Paths to parquet files to analyze")
    parser.add_argument("--dir", help="Directory containing parquet files to analyze")
    args = parser.parse_args()
    
    if args.dir:
        dir_path = Path(args.dir)
        file_paths = [f for f in dir_path.glob("*.parquet") 
                     if "without_similarities" not in f.name]
    elif args.files:
        file_paths = [Path(f) for f in args.files]
    else:
        parser.error("Either --files or --dir must be provided")
    
    print(f"\nAnalyzing {len(file_paths)} files:")
    for f in file_paths:
        print(f"- {f.name}")
    
    metrics, df = analyze_parquet_files(file_paths)
    
    print("\nMetrics Summary:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nSample of data:")
    print("-" * 50)
    print(df.head())


if __name__ == "__main__":
    main() 