from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ModelConfig:
    context_size: int
    threads: int
    n_len: int
    seed: int
    temp: float
    top_p: float
    skip_non_utf8: bool
    truncate_if_context_full: bool
    max_tokens: int

    @classmethod
    def classification(cls) -> 'ModelConfig':
        return cls(
            context_size=8 * 1024,
            threads=4,
            n_len=256,
            seed=1337,
            temp=0.25,
            top_p=0.7,
            skip_non_utf8=True,
            truncate_if_context_full=True,
            max_tokens=10,
        )

    @classmethod
    def summarization(cls) -> 'ModelConfig':
        return cls(
            context_size=4 * 1024,
            threads=4,
            n_len=512,
            seed=1337,
            temp=0.1,
            top_p=0.7,
            skip_non_utf8=True,
            truncate_if_context_full=True,
            max_tokens=200,
        )

    @classmethod
    def toxicity(cls) -> 'ModelConfig':
        return cls(
            context_size=2048,
            threads=4,
            n_len=256,
            seed=42,
            temp=0.3,
            top_p=0.75,
            skip_non_utf8=True,
            truncate_if_context_full=True,
            max_tokens=100,
        )

@dataclass
class DataConfig:
    dataset_path: str
    dataset_url: str
    limit: Optional[int]
    start_from: int
    skip_if_longer_than: Optional[int]

@dataclass
class OutputConfig:
    output_dir: Path
    file_prefix: str

@dataclass
class TaskConfig:
    model: ModelConfig
    data: DataConfig
    output: OutputConfig

    @classmethod
    def classification(cls) -> 'TaskConfig':
        return cls(
            model=ModelConfig.classification(),
            data=DataConfig(
                dataset_path="classification_pairs.parquet",
                dataset_url="https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet",
                limit=1000,
                start_from=0,
                skip_if_longer_than=None,
            ),
            output=OutputConfig(
                output_dir=Path("quantization_ablation_model"),
                file_prefix="llama_classification",
            ),
        )

    @classmethod
    def summarization(cls) -> 'TaskConfig':
        return cls(
            model=ModelConfig.summarization(),
            data=DataConfig(
                dataset_path="analysis/benchmarks/data/xsum_test.csv",
                dataset_url="https://huggingface.co/datasets/knkarthick/xsum/resolve/main/test.csv?download=true",
                limit=500,
                start_from=0,
                skip_if_longer_than=1750,
            ),
            output=OutputConfig(
                output_dir=Path("quantization_ablation_model"),
                file_prefix="llama_summaries",
            ),
        )

    @classmethod
    def toxicity(cls) -> 'TaskConfig':
        return cls(
            model=ModelConfig.toxicity(),
            data=DataConfig(
                dataset_path="analysis/benchmarks/data/toxicity_data.csv",
                dataset_url="https://huggingface.co/datasets/lmsys/toxic-chat/resolve/main/data/0124/toxic-chat_annotation_test.csv",
                limit=500,
                start_from=0,
                skip_if_longer_than=None,
            ),
            output=OutputConfig(
                output_dir=Path("."),
                file_prefix="llama3_7b",
            ),
        ) 