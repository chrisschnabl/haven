use indicatif::{ProgressBar, ProgressStyle};
use tracing::debug;
use std::time::Instant;

pub struct ProgressTracker {
    bar: ProgressBar,
    start_time: Instant,
    tokens_generated: usize,
    completed_examples: usize,
}

impl ProgressTracker {
    pub fn new(total: usize) -> Self {
        let bar = ProgressBar::new(total as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        
        Self {
            bar,
            start_time: Instant::now(),
            tokens_generated: 0,
            completed_examples: 0,
        }
    }
    
    pub fn update(&mut self, message: impl Into<String>) {
        self.completed_examples += 1;
        self.bar.inc(1);
        
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let current_tokens_per_sec = self.tokens_generated as f64 / elapsed;
        
        let msg = format!("{} - {:.2} tokens/sec", message.into(), current_tokens_per_sec);
        self.bar.set_message(msg);
    }
    
    pub fn add_tokens(&mut self, tokens: usize) {
        self.tokens_generated += tokens;
    }
    
    pub fn finish(&self, _message: impl Into<String>) {
        self.bar.finish();
        
        let total_elapsed = self.start_time.elapsed().as_secs_f64();
        let overall_tokens_per_sec = self.tokens_generated as f64 / total_elapsed;
        
        debug!("Generated {} tokens across {} examples in {:.2}s ({:.2} tokens/sec)",
               self.tokens_generated, 
               self.completed_examples,
               total_elapsed, 
               overall_tokens_per_sec);
    }
} 