use std::sync::Arc;
use std::io::{stdout, Write};
use anyhow::Result;
use llama_runner::LlamaRunner;
use arrow::array::{Int64Array, StringArray, BooleanArray};
use arrow::datatypes::{Schema, Field, DataType};
use std::path::PathBuf;
use tracing::debug;

use crate::dataset::{DatasetEntry, ClassificationContent, ParquetDatasetLoader};
use crate::config::TaskConfig;
use crate::progress::ProgressTracker;
use crate::writer::ParquetWriter;
use crate::util::format_header;

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

pub fn run_classification(limit_override: Option<usize>, model_override: Option<String>) -> Result<()> {
    debug!("Loading classification dataset...");
    
    let mut config = TaskConfig::classification();
    
    if let Some(limit) = limit_override {
        config.data.limit = Some(limit);
    }
    
    if let Some(model_path) = model_override {
        config.model.model_path = PathBuf::from(model_path);
    }
    
    let loader = ParquetDatasetLoader::new(
        &config.data.dataset_path,
        &config.data.dataset_url
    );
    let entries: Vec<DatasetEntry<ClassificationContent>> = loader.load_classification_data(config.data.limit, config.data.start_from)?;
    
    let mut llama = LlamaRunner::new(config.model.to_llama_config());
    llama.load_model()?;
    
    let prompt_builder = ClassificationPromptBuilder;
    let response_processor = ClassificationProcessor;
    let total = limit_override.unwrap_or(entries.len());
    let mut progress = ProgressTracker::new(total);

    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("question", DataType::Utf8, false),
        Field::new("response", DataType::Utf8, false),
        Field::new("expected", DataType::Utf8, false),
        Field::new("correct", DataType::Boolean, false),
    ]);
    let mut writer = ParquetWriter::new(schema, config.output)?;

    let mut valid_choices = Vec::new();
    let mut incorrect_responses = Vec::new();
    let mut correct_responses = Vec::new();
    
    for (idx, entry) in entries.iter().enumerate().take(total) {
        stdout().flush()?;
        let prompt = prompt_builder.build_prompt(entry);
        let mut response = String::new();
        
        let (token_count, duration) = llama.generate_blocking(&prompt, |token| {
            if let Ok(token_str) = String::from_utf8(token.as_bytes().to_vec()) {
                response.push_str(&token_str);
            }
        })?;
        
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
        ])?;

        progress.add_tokens(token_count as usize);
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