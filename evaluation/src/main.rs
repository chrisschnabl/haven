mod dataset;
mod config;
mod progress;
mod writer;
mod tasks;
mod util;

use clap::{Parser, Subcommand};
use anyhow::Result;
use tasks::classification::{run_classification, get_classificaiton_data};
use tasks::summarization::run_summarization;
use tasks::toxicity::run_toxicity;
use tasks::toxicity::analyze_toxicity;
use tracing_subscriber;
use tracing::debug;
use std::path::PathBuf;
use std::sync::Arc;
use arrow::array::{Int64Array, StringArray, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use crate::progress::ProgressTracker;
use crate::writer::ParquetWriter;
use rand;

/// Benchmark tool for evaluating LLM performance on different tasks
/// We would expect the model to be around 60 - 65% correct for GGUF-I-Quant,
/// we test the Instruct finetuned ones. So expect to be slightly below that.
#[derive(Parser, Debug)]
#[command(name = "Haven Evaluation")]
#[command(author = "Christoph Schnabl <cs228@cam.ac.uk>")]
#[command(version = "1.0")]
#[command(about = "TODO", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run classification benchmark (MMLU dataset)
    Class {
        /// Number of examples to test (default: use value from config)
        #[arg(short, long)]
        limit: Option<usize>,
        
        /// Model to use (default: use value from config)
        #[arg(short, long)]
        model: Option<String>,
    },
    
    /// Run summarization benchmark (XSUM dataset)
    Summarize {
        /// Number of examples to test (default: use value from config)
        #[arg(short, long)]
        limit: Option<usize>,
        
        /// Model to use (default: use value from config)
        #[arg(short, long)]
        model: Option<String>,
    },
    
    /// Run toxicity evaluation benchmark
    Toxicity {
        /// Number of examples to test (default: use value from config)
        #[arg(short, long)]
        limit: Option<usize>,
        
        /// Model to use (default: use value from config)
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Classify data from the MMLU dataset
    ClassifyData {
        /// Number of examples to process (default: use value from config)
        #[arg(short, long)]
        limit: Option<usize>,
    },

    ClassifyToxicity {
        #[arg(short, long)]
        file: String,
    },
    
    Similarity {
        #[arg(short, long)]
        file: String, 
    }
}

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "llama_runner=info");
    
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .without_time()
        .init();
    
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Class { limit, model } => {
            debug!("Running classification task...");
            for model in ["model/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"] { //, "model/Meta-Llama-3-8B-Instruct.Q8_0.gguf", "model/Meta-Llama-3-8B-Instruct.Q2_K.gguf"] {
                run_classification(limit.clone(), Some(model.to_string()))?;
            }
        },
        Commands::Summarize { limit, model } => {
            debug!("Running summarization task...");
            for model in ["model/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", "model/Meta-Llama-3-8B-Instruct.Q8_0.gguf", "model/Meta-Llama-3-8B-Instruct.Q2_K.gguf"] {
                run_summarization(limit.clone(), Some(model.to_string()))?;
            }
        },
        Commands::Toxicity { limit, model } => {
            debug!("Running toxicity task...");
            for model in ["model/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", "model/Meta-Llama-3-8B-Instruct.Q8_0.gguf", "model/Meta-Llama-3-8B-Instruct.Q2_K.gguf"] {
                run_toxicity(limit.clone(), Some(model.to_string()))?;
            }
        },
        Commands::ClassifyData { limit } => {
            debug!("Loading classification dataset...");
            let entries = get_classificaiton_data(
                "classification_pairs.parquet",
                "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet",
                limit.clone()
            )?;
            println!("Successfully processed {} classification entries", entries.len());
        },
        Commands::ClassifyToxicity { file } => {
            // Load the toxicity dataset
            // And run the bert classifier on it
            let file = PathBuf::from(file);
            analyze_toxicity(&file)?;
           
        }

        Commands::Similarity { file } => {
            // This is a datset of parquet with
            // {"id":,"dialogue":"","response":"","expected":"","score":0,"duration":10.334778,"token_count":37,"tokenize_duration":0.00052,"prompt_duration":51.713853,"prompt_tokens":280}
            // We want to run the similiarty task on it
            // And save the results to a new parquet file
            // Score the summarization in score column 
            // Keep all the other columsn the same
            let file = PathBuf::from(file);
            run_similarity(file)?;
        }

    }
    
    Ok(())
}
pub fn run_similarity(file: PathBuf) -> Result<()> {
    debug!("Loading summarization dataset from file: {:?}", file);

    // Load the parquet file
    let file_reader = std::fs::File::open(&file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file_reader)?;
    let record_batch_reader = builder.build()?;
    
    let similarity_model = SimilarityModel::new()?;
    let mut progress = ProgressTracker::new(0); // Will update count after loading
    
    // Create output file path by adding "_analyzed" before the extension
    let file_stem = file.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let file_ext = file.extension().unwrap_or_default().to_string_lossy().to_string();
    let output_path = file.with_file_name(format!("{}_{}.{}", file_stem, "analyzed", file_ext));
    
    // Use the same schema as the input file
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
    
    // Create output configuration
    let output_config = crate::config::OutputConfig {
        output_dir: output_path.parent().unwrap_or(&PathBuf::from(".")).to_path_buf(),
        file_prefix: output_path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
    };
    
    let mut writer = ParquetWriter::new(schema, output_config)?;
    
    let mut total_score = 0.0;
    let mut count = 0;
    
    // Process each batch from the parquet file
    for batch_result in record_batch_reader {
        let batch = batch_result?;
        // Update progress with the number of rows in this batch
        progress.update(format!("Processing batch of {} rows", batch.num_rows()));
        
        // Extract columns from the batch
        let id_array = batch.column_by_name("id").unwrap().as_any().downcast_ref::<Int64Array>().unwrap();
        let dialogue_array = batch.column_by_name("dialogue").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let response_array = batch.column_by_name("response").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let expected_array = batch.column_by_name("expected").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let duration_array = batch.column_by_name("duration").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        let token_count_array = batch.column_by_name("token_count").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        let tokenize_duration_array = batch.column_by_name("tokenize_duration").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        let prompt_duration_array = batch.column_by_name("prompt_duration").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        let prompt_tokens_array = batch.column_by_name("prompt_tokens").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Process each row
        for i in 0..batch.num_rows() {
            progress.update(format!("Processing entry {}", i));
            
            let id = id_array.value(i);
            let dialogue = dialogue_array.value(i);
            let response = response_array.value(i);
            let expected = expected_array.value(i);
            let duration = duration_array.value(i);
            let token_count = token_count_array.value(i);
            let tokenize_duration = tokenize_duration_array.value(i);
            let prompt_duration = prompt_duration_array.value(i);
            let prompt_tokens = prompt_tokens_array.value(i);
            
            // Calculate similarity score
            let score = similarity_model.similarity(expected, response)?;
            total_score += score;
            count += 1;
            
            // Add row to output
            writer.add_row(i, vec![
                Arc::new(Int64Array::from(vec![id])),
                Arc::new(StringArray::from(vec![dialogue.to_string()])),
                Arc::new(StringArray::from(vec![response.to_string()])),
                Arc::new(StringArray::from(vec![expected.to_string()])),
                Arc::new(Float64Array::from(vec![score])),
                Arc::new(Float64Array::from(vec![duration])),
                Arc::new(Float64Array::from(vec![token_count])),
                Arc::new(Float64Array::from(vec![tokenize_duration])),
                Arc::new(Float64Array::from(vec![prompt_duration])),
                Arc::new(Float64Array::from(vec![prompt_tokens])),
            ])?;
        }
    }
    
    writer.close()?;
    progress.finish("Summarization analysis complete!");
    
    if count > 0 {
        println!("Average similarity score: {:.4}", total_score / count as f64);
    }
    println!("Processed {} entries", count);
    
    Ok(())
}

// Add the SimilarityModel implementation
struct SimilarityModel {
    // Add any fields needed for the model
}

impl SimilarityModel {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }
    
    pub fn similarity(&self, expected: &str, response: &str) -> Result<f64> {
        // Simple implementation - can be replaced with a more sophisticated one
        // This is just a placeholder that returns a random score between 0 and 1
        Ok(rand::random::<f64>())
    }
}