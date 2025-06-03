from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import time
import random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from openai import OpenAI
import statistics

from .config import TaskConfig

@dataclass
class ClassificationContent:
    id: int
    question: str
    choices: List[str]
    answer_index: int
    answer: str

@dataclass
class DatasetEntry:
    id: int
    content: ClassificationContent

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
    accuracy: float
    valid_choices: int
    correct_responses: int
    total_entries: int
    combined_tokens_per_second: float
    total_combined_tokens: int

class ClassificationPromptBuilder:
    @staticmethod
    def build_messages(entry: DatasetEntry) -> list[dict]:
        system_msg = {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant. Answer the multiple-choice "
                "question by returning only a single capital letter (A, B, C or D)."
            )
        }

        # Put question + choices in ONE user message
        choices = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(entry.content.choices))
        user_msg = {
            "role": "user",
            "content": f"{entry.content.question}\n\n{choices}"
        }
        return [system_msg, user_msg]

class ClassificationProcessor:
    @staticmethod
    def process_response(response: str) -> str:
        prefix = "<|start_header_id|>assistant<|end_header_id|>"
        if response.startswith(prefix):
            response = response[len(prefix):]
        return response.strip()

def load_classification_data(dataset_path: str, dataset_url: str, limit: Optional[int] = None, start_from: int = 0) -> List[DatasetEntry]:
    """Load classification data from the MMLU dataset."""
    import pandas as pd
    import os
    from pathlib import Path
    
    # Ensure the dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}, downloading...")
        import requests
        response = requests.get(dataset_url)
        Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_path, 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully.")
    
    # Load the parquet file
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} entries from parquet dataset")
    
    entries = []
    for idx, row in df.iterrows():
        if idx >= start_from and (limit is None or len(entries) < limit):
            # Split choices into list
            choices = row['choices'].split(',')
            
            # Use answer_index directly if available, otherwise try to find it
            if 'answer_index' in row and pd.notna(row['answer_index']):
                answer_index = int(row['answer_index'])
            else:
                try:
                    answer_index = choices.index(row['answer'])
                except ValueError:
                    print(f"Warning: Could not find answer '{row['answer']}' in choices for entry {idx}")
                    continue
            
            # Convert index to letter (0->A, 1->B, etc.)
            answer_letter = chr(65 + answer_index)
            
            entries.append(DatasetEntry(
                id=idx,
                content=ClassificationContent(
                    id=idx,
                    question=row['question'],
                    choices=choices,
                    answer_index=answer_index,
                    answer=answer_letter
                )
            ))
    
    return entries

def run_benchmark(limit_override: Optional[int] = None) -> BenchmarkMetrics:
    """Run a benchmark test on the classification task and return detailed metrics."""
    config = TaskConfig.classification()
    
    # Override limit to 50 for testing
    #limit_override = 50
    
    entries = load_classification_data(
        "/Users/chris/haven/analysis/benchmarks/data/classification_entries.parquet",
        config.data.dataset_url,
        limit_override,
        config.data.start_from
    )
    
    print(f"\nLoaded {len(entries)} entries for testing")
    
    prompt_builder = ClassificationPromptBuilder()
    
    # Set up OpenAI client with proper error handling
    try:
        client = OpenAI(
            api_key="sk-no-key-needed",  # This is a placeholder, the actual key is not needed for the proxy
            base_url="https://7e4zolmwbqe2j8-8000.proxy.runpod.net/v1",
        )
        print("Successfully initialized OpenAI client")
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        raise
    
    durations = []
    token_counts = []
    prompt_token_counts = []
    valid_choices = []
    correct_responses = []
    incorrect_responses = []
    invalid_responses = []
    
    pbar = tqdm(entries, desc="Running benchmark")
    for entry in pbar:
        try:
            start_time = time.time()
            try:
                response = client.chat.completions.create(
                    model="nreHieW/Llama-3.1-8B-Instruct",
                    messages=ClassificationPromptBuilder.build_messages(entry),
                    max_tokens=1,          # 1 token is enough for 'A'‒'D'
                    temperature=0.25,
                    top_p=0.7,
                )
                duration = time.time() - start_time
                durations.append(duration)
                
                # Check if response has the expected structure
                if not hasattr(response, 'usage') or not hasattr(response.usage, 'completion_tokens'):
                    print(f"\nWarning: Unexpected response structure for entry {entry.content.id}")
                    print(f"Response: {response}")
                    continue
                    
                token_count = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
                token_counts.append(token_count)
                prompt_token_counts.append(prompt_tokens)
                
                # Check if response has choices
                if not response.choices or len(response.choices) == 0:
                    print(f"\nWarning: No choices in response for entry {entry.content.id}")
                    continue
                
                answer_raw = response.choices[0].message.content.strip()
                answer_letter = answer_raw[0] if answer_raw else "E"
                
                valid_answer_chars = ['A', 'B', 'C', 'D']
                is_valid_choice = answer_letter in valid_answer_chars
                
                expected_answer = chr(65 + entry.content.answer_index)
                is_correct = is_valid_choice and answer_letter == expected_answer
                
                if is_valid_choice:
                    valid_choices.append(entry.content.id)
                    if is_correct:
                        correct_responses.append(entry.content.id)
                    else:
                        incorrect_responses.append(entry.content.id)
                else:
                    # In dubio pro rerum natura, try to parse the response as text
                    # This matches the Rust logic where it tries to match the answer text
                    if answer_raw == entry.content.answer:
                        valid_choices.append(entry.content.id)
                        correct_responses.append(entry.content.id)
                    else:
                        incorrect_responses.append(entry.content.id)
                        print(f"\nInvalid response: '{answer_raw}' (expected: {expected_answer})")
                
                # Update progress bar with accuracy and counts
                accuracy = len(correct_responses) / (len(entries[:len(durations)]))
                invalid_count = len(invalid_responses)
                correct_count = len(correct_responses)
                incorrect_count = len(incorrect_responses)
                pbar.set_postfix({
                    'acc': f'{accuracy:.2%}',
                    'cor': f'{correct_count}',
                    'inc': f'{incorrect_count}',
                    'inv': f'{invalid_count}'
                }, refresh=True)
                
            except Exception as e:
                print(f"\nError in API call for entry {entry.content.id}: {str(e)}")
                continue
            
        except Exception as e:
            print(f"\nError processing entry {entry.content.id}: {str(e)}")
            print(f"Question: {entry.content.question}")
            print(f"Choices: {entry.content.choices}")
            print(f"Expected Answer: {chr(65 + entry.content.answer_index)}")
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
            accuracy=0,
            valid_choices=0,
            correct_responses=0,
            total_entries=len(entries),
            combined_tokens_per_second=0,
            total_combined_tokens=0
        )
    
    total_duration = sum(durations)
    total_tokens = sum(token_counts)
    total_prompt_tokens = sum(prompt_token_counts)
    total_combined_tokens = total_tokens + total_prompt_tokens
    
    # Calculate combined tokens per second
    combined_tokens_per_second = total_combined_tokens / total_duration if total_duration > 0 else 0
    
    # Print detailed accuracy metrics
    print("\n=== Accuracy Analysis ===")
    print(f"Total Entries: {len(entries)}")
    print(f"Valid Choices: {len(valid_choices)}")
    print(f"Correct Responses: {len(correct_responses)}")
    print(f"Incorrect Responses: {len(incorrect_responses)}")
    print(f"Invalid Responses: {len(invalid_responses)}")
    print(f"Overall Accuracy: {len(correct_responses) / len(entries):.2%}")
    print(f"Accuracy (Valid Only): {len(correct_responses) / len(valid_choices):.2%} if valid_choices > 0 else 0%")
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
        accuracy=len(correct_responses) / len(entries) if entries else 0,
        valid_choices=len(valid_choices),
        correct_responses=len(correct_responses),
        total_entries=len(entries),
        combined_tokens_per_second=combined_tokens_per_second,
        total_combined_tokens=total_combined_tokens
    )

def run_classification(limit_override: Optional[int] = None) -> None:
    """Run the classification task and save results."""
    config = TaskConfig.classification()
    
    if limit_override is not None:
        config.data.limit = limit_override
    
    entries = load_classification_data(
        config.data.dataset_path,
        config.data.dataset_url,
        config.data.limit,
        config.data.start_from
    )
    
    if config.data.limit is not None:
        random.shuffle(entries)
        entries = entries[:config.data.limit]
    
    prompt_builder = ClassificationPromptBuilder()
    response_processor = ClassificationProcessor()
    
    os.environ["OPENAI_API_KEY"] = "sk-no-key-needed"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://7e4zolmwbqe2j8-8000.proxy.runpod.net/v1",
    )
    
    results = []
    valid_choices = []
    correct_responses = []
    incorrect_responses = []
    
    for entry in tqdm(entries, desc="Processing entries"):
        prompt = prompt_builder.build_prompt(entry)
        
        start_time = time.time()
        response = client.completions.create(
            model="nreHieW/Llama-3.1-8B-Instruct",
            prompt=prompt,
            max_tokens=4, # Do not overwrite this, force to output one token
            temperature=0.25,  # From Rust config temp
            top_p=0.7,  # From Rust config top_p
        )
        duration = time.time() - start_time
        
        processed_response = response_processor.process_response(response.choices[0].text)
        
        valid_answer_chars = ['A', 'B', 'C', 'D']
        response_char = processed_response[0] if processed_response else 'E'
        is_valid_choice = response_char in valid_answer_chars
        
        expected_answer = chr(65 + entry.content.answer_index)
        is_correct = is_valid_choice and response_char == expected_answer
        
        if is_valid_choice:
            valid_choices.append(entry.content.id)
            if is_correct:
                correct_responses.append(entry.content.id)
            else:
                incorrect_responses.append(entry.content.id)
        else:
            # In dubio pro rerum natura, try to parse the response as text
            # This matches the Rust logic where it tries to match the answer text
            if processed_response == entry.content.answer:
                valid_choices.append(entry.content.id)
                correct_responses.append(entry.content.id)
            else:
                incorrect_responses.append(entry.content.id)
        
        results.append({
            'id': entry.content.id,
            'question': entry.content.question,
            'response': processed_response,
            'expected': expected_answer,
            'correct': is_correct,
            'duration': duration,
            'token_count': response.usage.completion_tokens,
            'tokenize_duration': 0.0,  # Not available from OpenAI API
            'prompt_duration': 0.0,  # Not available from OpenAI API
            'prompt_tokens': response.usage.prompt_tokens,
        })
    
    # Save results to parquet
    df = pd.DataFrame(results)
    output_path = config.output.output_dir / f"{config.output.file_prefix}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    
    print(f"Valid choices: {len(valid_choices)}")
    print(f"Correct responses: {len(correct_responses)}")
    print(f"Incorrect responses: {len(incorrect_responses)}")
    print(f"Total: {len(entries)}")
    print(f"Accuracy: {100.0 * len(correct_responses) / len(entries):.2f}%")

if __name__ == "__main__":
    # Run benchmark and print detailed metrics
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
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"Valid Choices: {metrics.valid_choices}/{metrics.total_entries}")
    print(f"Correct Responses: {metrics.correct_responses}/{metrics.total_entries}") 