use std::sync::Arc;
use std::io::{stdout, Write};
use std::fs::File;
use anyhow::Result;
use llama_runner::LlamaRunner;
use arrow::array::{Int64Array, StringArray, BooleanArray, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use std::path::PathBuf;
use tracing::debug;
use std::convert::TryInto;
use serde_json;

use crate::dataset::{DatasetEntry, ClassificationContent, ParquetDatasetLoader};
use crate::config::TaskConfig;
use crate::progress::ProgressTracker;
use crate::writer::ParquetWriter;
use crate::util::format_header;

use rand::seq::SliceRandom;
use rand::thread_rng;

struct ClassificationPromptBuilder;
impl ClassificationPromptBuilder {
    fn build_prompt(&self, entry: &DatasetEntry<ClassificationContent>) -> String {
        let system_prompt = "You are a knowledgeable assistant. Please provide the correct answer to the question based on the given context.";
        let formatted_system_prompt = format_header("system", system_prompt);
        let formatted_question = format_header("question", &entry.content.question);
        let choices_formatted = format_header(
            "choices",
            &entry.content.choices.iter()
                .enumerate()
                .map(|(i, choice)| format!("{}) {}", (b'A' + i as u8) as char, choice))
                .collect::<Vec<_>>()
                .join("\n")
        );
        let user_prompt = format_header("user", "GIVE YOUR ANSWER AS A, B, C, or D ONLY. DO NOT PROVIDE ANY OTHER TEXT.");
        format!("{}{}{}{}", formatted_system_prompt, formatted_question, choices_formatted, user_prompt)
    }
}

struct ClassificationProcessor;
impl ClassificationProcessor {
    fn process_response(&self, response: &str) -> String {
        let mut result = response.to_string();
        let prefix = "<|start_header_id|>assistant<|end_header_id|>";
        if result.starts_with(prefix) {
            result = result[prefix.len()..].to_string();
        }
        result.trim().to_string()
    }
}

pub fn get_classificaiton_data(dataset_path: &str, dataset_url: &str, limit_override: Option<usize>) -> Result<Vec<DatasetEntry<ClassificationContent>>> {
    let loader = ParquetDatasetLoader::new(
        dataset_path,
        dataset_url
    );
    let mut entries: Vec<DatasetEntry<ClassificationContent>> = loader.load_classification_data(None, 0)?;

    let mut rng = thread_rng();
    let n = limit_override.unwrap_or(entries.len());
    entries.shuffle(&mut rng);
    entries = entries.into_iter().take(n).collect();

    // Create schema for the entries
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("question", DataType::Utf8, false),
        Field::new("subject", DataType::Utf8, false),
        Field::new("choices", DataType::Utf8, false),
        Field::new("answer", DataType::Utf8, false),
        Field::new("answer_index", DataType::Int64, false),
    ]);

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;
    
    // Create Parquet writer
    let mut writer = ParquetWriter::new(schema, crate::config::OutputConfig {
        output_dir: PathBuf::from("output"),
        file_prefix: "classification_entries".to_string(),
    })?;

    // Write entries to Parquet
    for entry in &entries {
        writer.add_row(entry.id, vec![
            Arc::new(Int64Array::from(vec![entry.content.id])),
            Arc::new(StringArray::from(vec![entry.content.question.clone()])),
            Arc::new(StringArray::from(vec![entry.content.subject.clone()])),
            Arc::new(StringArray::from(vec![entry.content.choices.join(",")])),
            Arc::new(StringArray::from(vec![entry.content.answer.clone()])),
            Arc::new(Int64Array::from(vec![entry.content.answer_index])),
        ])?;
    }

    writer.close()?;
    println!("Saved {} entries to output/classification_entries.parquet", entries.len());
    Ok(entries)
}

pub fn run_classification(limit_override: Option<usize>, model_override: Option<String>) -> Result<()> {
    debug!("Loading classification dataset...");
    
    let mut config = TaskConfig::classification();

    if let Some(limit) = limit_override {
        config.data.limit = Some(limit);
    }
    
    if let Some(model_path) = model_override {
        config.model.model_path = PathBuf::from(&model_path);
        config.output.output_dir = PathBuf::from(format!("quantization_ablation_{}", model_path));
    } else {
        config.output.output_dir = PathBuf::from("quantization_ablation_");
    }
    
    let loader = ParquetDatasetLoader::new(
        &config.data.dataset_path,
        &config.data.dataset_url
    );
    let entries: Vec<DatasetEntry<ClassificationContent>> = loader.load_classification_data(None, 0)?;
    
    // Filter out 1000 rows randomly
    let mut rng = rand::rng();
    let n = limit_override.unwrap_or(config.data.limit.unwrap_or(entries.len()));
    let mut entries = entries.clone();
    entries.shuffle(&mut rng);
    entries = entries.into_iter().take(n).collect();

    let mut llama = LlamaRunner::new(config.model.to_llama_config());
    llama.load_model()?;
    
    let prompt_builder = ClassificationPromptBuilder;
    let response_processor = ClassificationProcessor;
    let mut progress = ProgressTracker::new(n);

    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("question", DataType::Utf8, false),
        Field::new("response", DataType::Utf8, false),
        Field::new("expected", DataType::Utf8, false),
        Field::new("correct", DataType::Boolean, false),
        Field::new("duration", DataType::Float64, false),
        Field::new("token_count", DataType::Float64, false),
        Field::new("tokenize_duration", DataType::Float64, false),
        Field::new("prompt_duration", DataType::Float64, false),
        Field::new("prompt_tokens", DataType::Float64, false),
    ]);

    let mut writer = ParquetWriter::new(schema, config.output)?;

    let mut valid_choices = Vec::new();
    let mut incorrect_responses = Vec::new();
    let mut correct_responses = Vec::new();
    
    for (idx, entry) in entries.iter().enumerate().take(n) {
        stdout().flush()?;
        let prompt = prompt_builder.build_prompt(entry);
        let mut response = String::new();
        
        let (token_count, duration, tokenize_duration, prompt_duration, prompt_tokens) = llama.generate_blocking(&prompt, |token| {
            if let Ok(token_str) = String::from_utf8(token.as_bytes().to_vec()) {
                response.push_str(&token_str);
            }
        })?;
        
        progress.add_tokens(token_count.try_into().unwrap(), duration.as_secs_f64());
        let processed_response = response_processor.process_response(&response);
        
        let valid_answer_chars = ['A', 'B', 'C', 'D'];
        let response_char = processed_response.chars().next().unwrap_or('E');
        let is_valid_choice = valid_answer_chars.contains(&response_char);
        
        // Convert answer index (0-3) to corresponding letter (A-D)
        let expected_answer = (b'A' + entry.content.answer_index as u8) as char;
        let is_correct = is_valid_choice && response_char == expected_answer;

        if is_valid_choice {
            valid_choices.push(entry.content.id);
            if is_correct {
                correct_responses.push(entry.content.id);
            } else {
                incorrect_responses.push(entry.content.id);
            }
        } else {
            // In dubio pro rerum natura, try to parse the response as text 
            if processed_response == entry.content.answer {
                valid_choices.push(entry.content.id);
                correct_responses.push(entry.content.id);
            } else {
                incorrect_responses.push(entry.content.id);
            }
        }
        
        writer.add_row(idx, vec![
            Arc::new(Int64Array::from(vec![entry.content.id])),
            Arc::new(StringArray::from(vec![entry.content.question.clone()])),
            Arc::new(StringArray::from(vec![processed_response])),
            Arc::new(StringArray::from(vec![expected_answer.to_string()])),
            Arc::new(BooleanArray::from(vec![is_correct])),
            Arc::new(Float64Array::from(vec![duration.as_secs_f64()])),
            Arc::new(Float64Array::from(vec![token_count as f64])),
            Arc::new(Float64Array::from(vec![tokenize_duration.as_secs_f64()])),
            Arc::new(Float64Array::from(vec![prompt_duration.as_secs_f64()])),
            Arc::new(Float64Array::from(vec![prompt_tokens as f64])),
        ])?;

        progress.add_tokens(token_count.try_into().unwrap(), duration.as_secs_f64());
        progress.update(format!("Processing entry {}", entry.id));
    }

    writer.close()?;
    progress.finish("Classification complete!");
    
    println!("Valid choices: {}", valid_choices.len());
    println!("Correct responses: {}", correct_responses.len());
    println!("Incorrect responses: {}", incorrect_responses.len());
    println!("Total: {}", entries.len());
    println!("Accuracy: {:.2}%", 100.0 * correct_responses.len() as f64 / entries.len() as f64);
    
    Ok(())
} 