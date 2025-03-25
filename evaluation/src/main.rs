mod dataset;
mod config;
mod progress;
mod writer;
mod tasks;
mod util;

use clap::{Parser, Subcommand};
use anyhow::Result;
use tasks::classification::run_classification;
use tasks::summarization::run_summarization;
use tasks::toxicity::run_toxicity;
use tracing_subscriber;
use tracing::debug;

/// Benchmark tool for evaluating LLM performance on different tasks
/// We would expect the model to be around 60 - 65% correct for GGUF-I-Quant,
/// we test the Instruct finetuned ones. So expect to be slightly below that.
#[derive(Parser, Debug)]
#[command(name = "LLM Benchmark")]
#[command(author = "You <your.email@example.com>")]
#[command(version = "1.0")]
#[command(about = "Runs various benchmarks on LLMs", long_about = None)]
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
}

fn main() -> Result<()> {
    // Set environment variable to downgrade llama_runner logs
    std::env::set_var("RUST_LOG", "llama_runner=info");
    
    // Fix the tracing subscriber configuration
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .without_time()
        .init();
    
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Class { limit, model } => {
            debug!("Running classification task...");
            run_classification(limit.clone(), model.clone())?;
        },
        Commands::Summarize { limit, model } => {
            debug!("Running summarization task...");
            run_summarization(limit.clone(), model.clone())?;
        },
        Commands::Toxicity { limit, model } => {
            debug!("Running toxicity task...");
            run_toxicity(limit.clone(), model.clone())?;
        },
    }
    
    Ok(())
}