import argparse
from pathlib import Path
from typing import Optional

from .classification import run_classification
from .config import TaskConfig
from .toxicity import run_toxicity

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument("--task", type=str, choices=["classification", "summarization", "toxicity"], 
                       default="classification", help="Task to run")
    parser.add_argument("--limit", type=int, help="Override the dataset limit")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory")
    
    args = parser.parse_args()
    
    if args.task == "classification":
        run_classification(args.limit)
    elif args.task == "summarization":
        run_summarization(args.limit)
    elif args.task == "toxicity":
        run_toxicity(args.limit)

if __name__ == "__main__":
    main() 