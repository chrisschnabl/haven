from dataclasses import dataclass
from typing import List, Optional
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from openai import OpenAI
import statistics
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import torch
import requests
from datasets import load_dataset

from .config import TaskConfig

@dataclass
class SummarizationContent:
    id: int
    dialogue: str
    summary: str

@dataclass
class DatasetEntry:
    id: int
    content: SummarizationContent

@dataclass
class BenchmarkMetrics:
    total_duration: float
    avg_duration: float
    median_duration: float
    total_tokens: int
    avg_tokens: float
    median_tokens: float
    total_prompt_tokens: int
    avg_prompt_tokens: float
    median_prompt_tokens: float
    avg_similarity: float
    total_entries: int
    combined_tokens_per_second: float
    total_combined_tokens: int

class SummarizationPromptBuilder:
    @staticmethod
    def build_messages(entry: DatasetEntry) -> list[dict]:
        system_msg = {
            "role": "system",
            "content": "You are a professional summarizer. Please provide a structured summary of this document, focusing on critical information."
        }
        document_msg = {
            "role": "user",
            "content": f"Document:\n{entry.content.dialogue}\n\nPlease summarize the document in 150 characters or less."
        }
        return [system_msg, document_msg]

class SummarizationProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def process_response(self, response: str, expected: str) -> float:
        # Calculate similarity using sentence embeddings
        response_embedding = self.model.encode(response, convert_to_tensor=True)
        expected_embedding = self.model.encode(expected, convert_to_tensor=True)
        similarity = 1 - cosine(response_embedding.cpu().numpy(), expected_embedding.cpu().numpy())
        return similarity

def load_summarization_data(dataset_path: str, dataset_url: str, limit: Optional[int] = None, start_from: int = 0) -> List[DatasetEntry]:
    """Load summarization data from the dataset."""
    from datasets import load_dataset
    import os
    from pathlib import Path
    
    # Ensure the data directory exists
    data_dir = Path(dataset_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the XSUM dataset using Hugging Face datasets
        dataset = load_dataset("xsum", split="test")
        print(f"Loaded XSUM dataset with {len(dataset)} examples")
        
        entries = []
        for idx, example in enumerate(dataset):
            if idx >= start_from and (limit is None or len(entries) < limit):
                try:
                    # Get the document and summary
                    dialogue = str(example['document'])
                    summary = str(example['summary'])
                    
                    # Skip empty entries
                    if not dialogue or not summary:
                        continue
                    
                    # Skip entries that are too long
                    if len(dialogue) > 1750:  # Skip if longer than 1750 characters
                        continue
                    
                    entries.append(DatasetEntry(
                        id=idx,
                        content=SummarizationContent(
                            id=idx,
                            dialogue=dialogue,
                            summary=summary
                        )
                    ))
                except Exception as e:
                    print(f"Error processing example {idx}: {str(e)}")
                    continue
        
        print(f"Successfully loaded {len(entries)} entries")
        return entries
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def run_benchmark(limit_override: Optional[int] = None) -> BenchmarkMetrics:
    """Run a benchmark test on the summarization task and return detailed metrics."""
    config = TaskConfig.summarization()
    
    entries = load_summarization_data(
        config.data.dataset_path,
        config.data.dataset_url,
        limit_override or 500,  # Default to 500 entries
        config.data.start_from
    )
    
    print(f"\nLoaded {len(entries)} entries for testing")
    
    prompt_builder = SummarizationPromptBuilder()
    response_processor = SummarizationProcessor()
    
    try:
        client = OpenAI(
            api_key="sk-no-key-needed",
            base_url="https://7e4zolmwbqe2j8-8000.proxy.runpod.net/v1",
        )
        print("Successfully initialized OpenAI client")
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        raise
    
    durations = []
    token_counts = []
    prompt_token_counts = []
    similarities = []
    
    pbar = tqdm(entries, desc="Running benchmark")
    for entry in pbar:
        try:
            start_time = time.time()
            try:
                response = client.chat.completions.create(
                    model="nreHieW/Llama-3.1-8B-Instruct",
                    messages=prompt_builder.build_messages(entry),
                    temperature=0.25,
                    top_p=0.7,
                    max_tokens=150,  # Limit output to 150 tokens
                )
                duration = time.time() - start_time
                durations.append(duration)
                
                if not hasattr(response, 'usage'):
                    print(f"\nWarning: Unexpected response structure for entry {entry.content.id}")
                    continue
                    
                token_count = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
                token_counts.append(token_count)
                prompt_token_counts.append(prompt_tokens)
                
                if not response.choices:
                    print(f"\nWarning: No choices in response for entry {entry.content.id}")
                    continue
                
                response_text = response.choices[0].message.content.strip()
                similarity = response_processor.process_response(response_text, entry.content.summary)
                similarities.append(similarity)
                
                avg_similarity = sum(similarities) / len(similarities)
                tokens_per_second = (token_count + prompt_tokens) / duration
                pbar.set_postfix({
                    'sim': f'{avg_similarity:.3f}',
                    'tps': f'{tokens_per_second:.1f}'
                }, refresh=True)
                
            except Exception as e:
                print(f"\nError in API call for entry {entry.content.id}: {str(e)}")
                continue
            
        except Exception as e:
            print(f"\nError processing entry {entry.content.id}: {str(e)}")
            continue
    
    if not durations:
        print("\nWarning: No successful API calls were made")
        return BenchmarkMetrics(
            total_duration=0,
            avg_duration=0,
            median_duration=0,
            total_tokens=0,
            avg_tokens=0,
            median_tokens=0,
            total_prompt_tokens=0,
            avg_prompt_tokens=0,
            median_prompt_tokens=0,
            avg_similarity=0,
            total_entries=len(entries),
            combined_tokens_per_second=0,
            total_combined_tokens=0
        )
    
    total_duration = sum(durations)
    total_tokens = sum(token_counts)
    total_prompt_tokens = sum(prompt_token_counts)
    total_combined_tokens = total_tokens + total_prompt_tokens
    
    combined_tokens_per_second = total_combined_tokens / total_duration if total_duration > 0 else 0
    
    print("\n=== Similarity Analysis ===")
    print(f"Total Entries: {len(entries)}")
    print(f"Processed Entries: {len(similarities)}")
    print(f"Average Similarity: {sum(similarities) / len(similarities):.3f}")
    print(f"Tokens per Second: {combined_tokens_per_second:.1f}")
    print("======================\n")
    
    return BenchmarkMetrics(
        total_duration=total_duration,
        avg_duration=statistics.mean(durations) if durations else 0,
        median_duration=statistics.median(durations) if durations else 0,
        total_tokens=total_tokens,
        avg_tokens=statistics.mean(token_counts) if token_counts else 0,
        median_tokens=statistics.median(token_counts) if token_counts else 0,
        total_prompt_tokens=total_prompt_tokens,
        avg_prompt_tokens=statistics.mean(prompt_token_counts) if prompt_token_counts else 0,
        median_prompt_tokens=statistics.median(prompt_token_counts) if prompt_token_counts else 0,
        avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
        total_entries=len(entries),
        combined_tokens_per_second=combined_tokens_per_second,
        total_combined_tokens=total_combined_tokens
    )

if __name__ == "__main__":
    metrics = run_benchmark()
    print("\nBenchmark Results:")
    print(f"Total Duration: {metrics.total_duration:.2f}s")
    print(f"Average Duration: {metrics.avg_duration:.2f}s")
    print(f"Median Duration: {metrics.median_duration:.2f}s")
    print(f"Total Tokens: {metrics.total_tokens}")
    print(f"Average Tokens: {metrics.avg_tokens:.1f}")
    print(f"Median Tokens: {metrics.median_tokens:.1f}")
    print(f"Total Prompt Tokens: {metrics.total_prompt_tokens}")
    print(f"Average Prompt Tokens: {metrics.avg_prompt_tokens:.1f}")
    print(f"Median Prompt Tokens: {metrics.median_prompt_tokens:.1f}")
    print(f"Combined Tokens per Second: {metrics.combined_tokens_per_second:.1f}")
    print(f"Total Combined Tokens: {metrics.total_combined_tokens}")
    print(f"Average Similarity: {metrics.avg_similarity:.3f}") 