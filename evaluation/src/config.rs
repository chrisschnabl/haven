use std::path::PathBuf;
use std::num::NonZero;
use llama_runner::LlamaConfig;

#[derive(Clone)]
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub context_size: u32,
    pub threads: u32,
    pub n_len: u32,
    pub seed: u64,
    pub temp: f32,
    pub top_p: f32,
    pub skip_non_utf8: bool,
    pub truncate_if_context_full: bool,
}

const Q8_MODEL_PATH: &str = "model/Meta-Llama-3-8B-Instruct.Q8_0.gguf";
const Q4_MODEL_PATH: &str = "model/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf";

impl ModelConfig {
    pub fn to_llama_config(&self) -> LlamaConfig {
        LlamaConfig {
            model_path: Some(self.model_path.to_str().unwrap().to_string()),
            context_size: NonZero::new(self.context_size).unwrap(),
            threads: self.threads as i32,
            n_len: self.n_len as i32,
            seed: self.seed as i64,
            temp: self.temp,
            top_p: self.top_p,
            skip_non_utf8: self.skip_non_utf8,
            truncate_if_context_full: self.truncate_if_context_full,
        }
    }

    pub fn classification() -> Self {
        Self {
            model_path: PathBuf::from(Q8_MODEL_PATH),
            context_size: 8 * 1024, //
            // lTODO: CSlama_init_from_model: n_ctx_per_seq (4096) < n_ctx_train (8192) -- the full capacity of the model will not be utilized
            threads: 4,
            n_len: 256,
            seed: 1337,
            temp: 0.25,
            top_p: 0.7,
            skip_non_utf8: true,
            truncate_if_context_full: true,
        }
    }

    pub fn summarization() -> Self {
        Self {
            model_path: PathBuf::from(Q8_MODEL_PATH),
            context_size: 4 * 1024,
            threads: 4,
            n_len: 512,
            seed: 1337,
            temp: 0.1,
            top_p: 0.7,
            skip_non_utf8: true,
            truncate_if_context_full: true,
        }
    }

    pub fn toxicity() -> Self {
        Self {
            model_path: PathBuf::from(Q8_MODEL_PATH),
            context_size: 2048,
            threads: 4,
            n_len: 256,
            seed: 42,
            temp: 0.3,
            top_p: 0.75,
            skip_non_utf8: true,
            truncate_if_context_full: true,
        }
    }
}

#[derive(Clone)]
pub struct DataConfig {
    pub dataset_path: String,
    pub dataset_url: String,
    pub limit: Option<usize>,
    pub start_from: usize,
    pub skip_if_longer_than: Option<usize>,
}

#[derive(Clone)]
pub struct OutputConfig {
    pub output_dir: PathBuf,
    pub file_prefix: String,
}

#[derive(Clone)]
pub struct TaskConfig {
    pub model: ModelConfig,
    pub data: DataConfig,
    pub output: OutputConfig,
}

impl TaskConfig {
    pub fn summarization() -> Self {
        Self {
            model: ModelConfig::summarization(),
            data: DataConfig {
                dataset_path: "similarity_pairs.csv".to_string(),
                dataset_url: "https://huggingface.co/datasets/knkarthick/xsum/resolve/main/test.csv".to_string(),
                limit: Some(500),
                start_from: 0,
                skip_if_longer_than: Some(1750),
            },
            output: OutputConfig {
                output_dir: PathBuf::from("quantization_ablation_model"),
                file_prefix: "llama_summaries".to_string(),
            },
        }
    }

    pub fn classification() -> Self {
        Self {
            model: ModelConfig::classification(),
            data: DataConfig {
                dataset_path: "classification_pairs.parquet".to_string(),
                dataset_url: "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet".to_string(),
                limit: Some(1000),
                start_from: 0,
                skip_if_longer_than: None,
            },
            output: OutputConfig {
                output_dir: PathBuf::from("quantization_ablation_model"),
                file_prefix: "llama_classification".to_string(),
            },
        }
    }

    pub fn toxicity() -> Self {
        Self {
            model: ModelConfig::toxicity(),
            data: DataConfig {
                dataset_path: "toxic-chat_annotation_test.csv".to_string(),
                dataset_url: "https://huggingface.co/datasets/lmsys/toxic-chat/resolve/main/data/0124/toxic-chat_annotation_test.csv".to_string(),
                limit: Some(500),
                start_from: 0,
                skip_if_longer_than: None,
            },
            output: OutputConfig {
                output_dir: PathBuf::from("."),
                file_prefix: "llama3_7b".to_string(),
            },
        }
    }
} 