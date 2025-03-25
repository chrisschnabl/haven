mod dataset;
use llama_runner::{LlamaRunner, LlamaConfig};
use bert_runner::{BertRunner, BertRunnerTrait};
use bert_runner::score::SimilarityModel;
use std::path::PathBuf;
use std::fs::File;
use std::io::{BufReader, Write};
use reqwest;
use csv;
use anyhow::{Result, Context};
use std::io::stdout;
use std::num::NonZero;
use arrow::array::{Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::sync::Arc;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow::array::Array;
use arrow::array::BooleanArray;
use arrow::array::Float64Array;
use indicatif::{ProgressBar, ProgressStyle};
use dataset::{DatasetLoader, ToxicityContent, SimilarityContent, DatasetEntry};
use std::cmp::min;
   
fn main() -> Result<()> {
    let test_size = 500;
    let start_from = 200;
    
    if std::env::args().any(|arg| arg == "--analyze") {
        analyze_responses()?;
    } else if std::env::args().any(|arg| arg == "--score") {
        score_responses()?;
    } else {
        generate_responses(test_size, start_from)?;
    }
    Ok(())
}

fn score_responses() -> Result<()> {
    let model = SimilarityModel::new()?;
    
    let loader = DatasetLoader::<SimilarityContent>::new(
        "similarity_pairs.csv",
        "https://huggingface.co/datasets/knkarthick/xsum/resolve/main/test.csv"
    );
    
    let entries: Vec<DatasetEntry<SimilarityContent>> = loader.load_or_download(Some(20), 0)?;
    
    let model_path = PathBuf::from("model/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf");
    
    let llama_config = LlamaConfig {
        model_path: Some(model_path.to_str().unwrap().to_string()),
        context_size: NonZero::new(4 * 1024).unwrap(), // If this is too small, the model cannot generate a summary.
        threads: 12,
        n_len: 512,
        seed: 12,
    };
    let mut llama = LlamaRunner::new(llama_config);
    llama.load_model()?;

    let mut ids = Vec::new();
    let mut dialogues = Vec::new();
    let mut summaries = Vec::new();
    let mut expected_summaries = Vec::new();

    let mut scores = Vec::new();

    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("=>-"));

    for (idx, entry) in entries.iter().enumerate() {
        stdout().flush()?;
        
        pb.set_message(format!("Processing entry {}", entry.content.id));
        
        let mut response = String::new();

        let oneshot = ".document {Part of the Broad Road was closed to traffic on Sunday at about 18:00 GMT.
        The three adults and three children have been taken to Altnagelvin Hospital
        with non life-threatening injuries. The Fire Service, Northern Ireland Ambulance Service
        and police attended the crash. The Broad Road has since been reopened.}
        .summary {Three adults and three children have been taken to hospital following a crash involving
        a tractor and a campervan in Limavady, County Londonderry}";

        // Ok adding a oneshot does not increase summarization quality by a lot. 
        //let example = "<|start_header_id|>example<|end_header_id|>{oneshot}<|start_header_id|>end_example<|end_header_id|>";
        let example = "";
        let truncated_length = min(1750, entry.content.dialogue.len());
        // TODO CS: Alternatively skip those entries that are too long.
        let truncated_dialogue = entry.content.dialogue[..truncated_length].to_string();
        
        // Skipping entries does not influence
        if truncated_dialogue.len() > 1750 {
            println!("Skipping entry {}", entry.content.id);
            continue;
        }
        // Zero-shot summarization
        // BOS already present
        // Make sure the prompt is roughly 150 chracters.
        let prompt = format!("<|start_header_id|>system<|end_header_id|>You are a professional summarizer. Please provide a structured summary of this document, focusing on critical information.
        <|eot_id|><|start_header_id|>document<|end_header_id|>{}<|eot_id|>{example}<|start_header_id|>user<|end_header_id|>Summarize the document in 150 characters or less.<|eot_id|>", truncated_dialogue);

        llama.generate_blocking(&prompt, |token| {
            if let Ok(token_str) = String::from_utf8(token.as_bytes().to_vec()) {
                response.push_str(&token_str);
            }
        })?;

        // Remove the BOS and assistant tokens
        let prefix = "<|start_header_id|>assistant<|end_header_id|>";
        if response.starts_with(prefix) {
            response = response[prefix.len()..].to_string();
        }
        response = response.trim_start_matches('\n').to_string();


        println!("response: {}, len {}", response, response.len());
        println!("summary: {}, len {}", entry.content.summary, entry.content.summary.len());
        println!("--------------------------------");

        let score = model.similarity(&entry.content.summary, &response)?;
        scores.push(score);
        println!("Score: {}", score);

        ids.push(entry.content.id);
        dialogues.push(entry.content.dialogue.clone());
        summaries.push(response);
        expected_summaries.push(entry.content.summary.clone());
        
        if (idx + 1) % 20 == 0 || idx == entries.len() - 1 {
            let id_array = Int64Array::from(ids.clone());
            let dialogue_array = StringArray::from(dialogues.clone());
            let summary_array = StringArray::from(summaries.clone());
            let expected_summary_array = StringArray::from(expected_summaries.clone());
            
            let schema = Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("dialogue", DataType::Utf8, false),
                Field::new("summary", DataType::Utf8, false),
                Field::new("expected_summary", DataType::Utf8, false),
            ]);

            let batch = RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(id_array),
                    Arc::new(dialogue_array),
                    Arc::new(summary_array),
                    Arc::new(expected_summary_array),
                ],
            )?;

            let file = File::create(format!("llama_summaries_{}.parquet", idx + 1))?;
            let props = WriterProperties::builder().build();
            let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;

            writer.write(&batch)?;
            writer.close()?;

            println!("Saved {} summaries to llama_summaries.parquet", idx + 1);
        }
        
        pb.inc(1);
    }

    pb.finish_with_message("Generation complete!");
    println!("All summaries generated and saved to llama_summaries.parquet");

    let average_score = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("Average score: {}", average_score);

    Ok(())
}

fn generate_responses(limit: usize, start_from: usize) -> Result<()> {
    let model_path = PathBuf::from("model/Meta-Llama-3-8B-Instruct.Q8_0.gguf");
    let url = "https://huggingface.co/datasets/lmsys/toxic-chat/resolve/main/data/0124/toxic-chat_annotation_test.csv";
    
    let loader = DatasetLoader::<ToxicityContent>::new(
        "toxic-chat_annotation_test.csv",
        url
    );
    
    let entries: Vec<DatasetEntry<ToxicityContent>> = loader.load_or_download(Some(limit), start_from)?;
    
    let llama_config = LlamaConfig {
        model_path: Some(model_path.to_str().unwrap().to_string()),
        context_size: NonZero::new(2048).unwrap(),
        threads: 12,
        n_len: 256,
        seed: 42,
    };
    let mut llama = LlamaRunner::new(llama_config);
    llama.load_model()?;

    println!("Processing {} test entries", entries.len());

    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("=>-"));

    let mut ids = Vec::new();
    let mut inputs = Vec::new();
    let mut responses = Vec::new();
    let mut expected_toxics = Vec::new();

    for (idx, entry) in entries.iter().enumerate() {
        stdout().flush()?;
        
        pb.set_message(format!("Processing entry {}", entry.content.id));
        
        let mut response = String::new();
        llama.generate_blocking(&entry.content.input, |token| {
            if let Ok(token_str) = String::from_utf8(token.as_bytes().to_vec()) {
                response.push_str(&token_str);
            }
        })?;

        ids.push(entry.content.id as i64);
        inputs.push(entry.content.input.clone());
        responses.push(response);
        expected_toxics.push(entry.content.toxic);
        
        if (idx + 1) % 20 == 0 || idx == entries.len() - 1 {
            let id_array = Int64Array::from(ids.clone());
            let input_array = StringArray::from(inputs.clone());
            let response_array = StringArray::from(responses.clone());
            let expected_toxic_array = Float64Array::from(expected_toxics.clone());

            let schema = Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("input", DataType::Utf8, false),
                Field::new("response", DataType::Utf8, false),
                Field::new("expected_toxic", DataType::Float64, false),
            ]);

            let batch = RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(id_array),
                    Arc::new(input_array),
                    Arc::new(response_array),
                    Arc::new(expected_toxic_array),
                ],
            )?;

            let file = File::create(format!("llama_responses_{}.parquet", idx + 1))?;
            let props = WriterProperties::builder().build();
            let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;

            writer.write(&batch)?;
            writer.close()?;

            println!("Saved {} responses to llama_responses.parquet", idx + 1);
        }
        
        pb.inc(1);
    }

    pb.finish_with_message("Generation complete!");
    println!("All responses generated and saved to llama_responses.parquet");
    Ok(())
}

fn analyze_responses() -> Result<()> {
    let model_path = PathBuf::from("model/rust_model.ot");
    let config_path = PathBuf::from("model/config.json");
    let vocab_path = PathBuf::from("model/vocab.txt");

    let mut bert = BertRunner::new(model_path, config_path, vocab_path);
    bert.load_model()?;

    let file = File::open("./results/llama_responses.parquet")
        .context("Failed to open llama responses file")?;
    
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create parquet reader")?;
    let mut reader = builder.build()
        .context("Failed to build parquet reader")?;
    
    let batch = reader.next()
        .context("No record batch found")?
        .context("Failed to read record batch")?;

    let id_array = batch.column(0).as_any().downcast_ref::<Int64Array>()
        .context("Failed to get id column")?;
    let input_array = batch.column(1).as_any().downcast_ref::<StringArray>()
        .context("Failed to get input column")?;
    let response_array = batch.column(2).as_any().downcast_ref::<StringArray>()
        .context("Failed to get response column")?;
    let expected_toxic_array = batch.column(3).as_any().downcast_ref::<Float64Array>()
        .context("Failed to get expected toxicity column")?;

    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("input", DataType::Utf8, false),
        Field::new("response", DataType::Utf8, false),
        Field::new("toxic", DataType::Boolean, false),
        Field::new("confidence", DataType::Float64, false),
        Field::new("expected_toxic", DataType::Float64, false),
    ]);

    let pb = ProgressBar::new(input_array.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) Analyzing responses")
        .unwrap()
        .progress_chars("=>-"));

    let batch_size = 50;
    let mut toxicity_scores = Vec::new();
    let mut toxics = Vec::new();


    for chunk_start in (0..response_array.len()).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(response_array.len());
        
        let batch_responses: Vec<String> = (chunk_start..chunk_end)
            .map(|i| {
                pb.inc(1);
                response_array.value(i).to_string()
            })
            .collect();

        let batch_predictions = bert.predict(batch_responses)?;

        for prediction in batch_predictions {
            toxicity_scores.push(prediction.score);
            toxics.push(prediction.text == "toxic");
        }
    }

    pb.finish_with_message("Analysis complete!");

    let toxicity_array = arrow::array::Float64Array::from(toxicity_scores);
    let toxic_array = arrow::array::BooleanArray::from(toxics);
    let output_batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            batch.column(0).clone(),
            batch.column(1).clone(),
            batch.column(2).clone(),
            Arc::new(toxic_array),
            Arc::new(toxicity_array),
            batch.column(3).clone(),
        ],
    )?;

    let output_file = File::create("analysis_results.parquet")?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(output_file, output_batch.schema(), Some(props))?;
    
    writer.write(&output_batch)?;
    writer.close()?;

    let toxic_count = output_batch.column(3).as_any().downcast_ref::<BooleanArray>()
        .context("Failed to get toxic column")?
        .iter()
        .filter(|&b| b.unwrap_or(false))
        .count();

    println!("Toxic count: {}, rate: {}", toxic_count, toxic_count as f64 / output_batch.num_rows() as f64);
    
    Ok(())
}