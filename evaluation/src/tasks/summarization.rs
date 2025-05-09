use std::sync::Arc;
use std::io::{stdout, Write};
use anyhow::Result;
use llama_runner::LlamaRunner;
use bert_runner::score::SimilarityModel;
use arrow::array::{Int64Array, StringArray, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use std::path::PathBuf;
use tracing::debug;
use std::convert::TryInto;

use crate::dataset::{DatasetEntry, SimilarityContent, DatasetLoader};
use crate::config::TaskConfig;
use crate::progress::ProgressTracker;
use crate::writer::ParquetWriter;
use crate::util::format_header;

struct SummarizationPromptBuilder;
impl SummarizationPromptBuilder {
    fn build_prompt(&self, entry: &DatasetEntry<SimilarityContent>) -> String {
        // Prompt taken from TODO
        let system_prompt = "You are a professional summarizer. Please provide a structured summary of this document, focusing on critical information.";
        let formatted_system_prompt = format_header("system", system_prompt);
        let formatted_document = format_header("document", &entry.content.dialogue);
        let formatted_user_prompt = format_header("user", "Summarize the document in 150 characters or less.");
        format!("{}{}{}", formatted_system_prompt, formatted_document, formatted_user_prompt)
    }
}

struct SummarizationProcessor;
impl SummarizationProcessor {
    fn process_response(&self, response: &str) -> String {
        let mut result = response.to_string();
        let prefix = "<|start_header_id|>assistant<|end_header_id|>";
        if result.starts_with(prefix) {
            result = result[prefix.len()..].to_string();
        }
        result.trim().to_string()
    }
}

pub fn run_summarization(limit_override: Option<usize>, model_override: Option<String>, config_override: Option<TaskConfig>, run_similarity: bool) -> Result<()> {
    debug!("Loading summarization dataset...");
    
    let mut config = TaskConfig::summarization();
    if let Some(config_override) = config_override {
        config = config_override;
    }
    
    if let Some(limit) = limit_override {
        config.data.limit = Some(limit);
    }
            
    if let Some(model_path) = model_override {
        config.model.model_path = PathBuf::from(&model_path);
        config.output.output_dir = PathBuf::from(format!("quantization_ablation_{}", model_path));
    } else {
        config.output.output_dir = PathBuf::from("quantization_ablation_");
    }
    
    let loader = DatasetLoader::<SimilarityContent>::new(
        &config.data.dataset_path,
        &config.data.dataset_url
    );
    let entries = loader.load_or_download(config.data.limit, config.data.start_from)?;

    let mut llama = LlamaRunner::new(config.model.to_llama_config());
    llama.load_model()?;
    
    let prompt_builder = SummarizationPromptBuilder;
    let response_processor = SummarizationProcessor;
    let mut progress = ProgressTracker::new(entries.len());
    let mut model = SimilarityModel::new()?;

    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("dialogue", DataType::Utf8, false),
        Field::new("response", DataType::Utf8, false),
        Field::new("expected", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
        Field::new("duration", DataType::Float64, false),
        Field::new("token_count", DataType::Float64, false),
        Field::new("tokenize_duration", DataType::Float64, false),
        Field::new("prompt_duration", DataType::Float64, false),
        Field::new("prompt_tokens", DataType::Float64, false),
    ]);
    let mut writer = ParquetWriter::new(schema, config.output)?;

    let mut total_score = 0.0;
    let mut count = 0;
    
    for (idx, entry) in entries.iter().enumerate() {
        if let Some(max_len) = config.data.skip_if_longer_than {
            if entry.content.dialogue.len() > max_len {
                continue;
            }
        }
        
        stdout().flush()?;
        progress.update(format!("Processing entry {}", entry.content.id));
        
        let prompt = prompt_builder.build_prompt(entry);
        let mut response = String::new();
        
        let (token_count, duration, tokenize_duration, prompt_duration, prompt_tokens) = llama.generate_blocking(&prompt, |token| {
            if let Ok(token_str) = String::from_utf8(token.as_bytes().to_vec()) {
                response.push_str(&token_str);
            }
        })?;

        progress.add_tokens(token_count.try_into().unwrap(), duration.as_secs_f64());

        let processed_response = response_processor.process_response(&response);

        let mut score = 0.0;
        if run_similarity {
            score = model.similarity(&entry.content.summary, &processed_response)?;
            //score = 0.0;
        }
        
        total_score += score;
        count += 1;
        
        writer.add_row(idx, vec![
            Arc::new(Int64Array::from(vec![entry.content.id])),
            Arc::new(StringArray::from(vec![entry.content.dialogue.clone()])),
            Arc::new(StringArray::from(vec![processed_response.clone()])),
            Arc::new(StringArray::from(vec![entry.content.summary.clone()])),
            Arc::new(Float64Array::from(vec![score])),
            Arc::new(Float64Array::from(vec![duration.as_secs_f64()])),
            Arc::new(Float64Array::from(vec![token_count as f64])),
            Arc::new(Float64Array::from(vec![tokenize_duration.as_secs_f64()])),
            Arc::new(Float64Array::from(vec![prompt_duration.as_secs_f64()])),
            Arc::new(Float64Array::from(vec![prompt_tokens as f64])),
        ])?;
    }

    writer.close()?;
    progress.finish("Summarization complete!");
    
    if count > 0 {
        println!("Average similarity score: {:.4}", total_score / count as f64);
    }
    println!("Processed {} entries", count);
    
    Ok(())
} 