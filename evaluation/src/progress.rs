use indicatif::{ProgressBar, ProgressStyle};
use tracing::debug;

pub struct ProgressTracker {
    bar: ProgressBar,
    tokens_generated: usize,
    completed_examples: usize,
    total_duration: f64,
    last_tokens: usize,
    last_duration: f64,
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
            tokens_generated: 0,
            completed_examples: 0,
            total_duration: 0.0,
            last_tokens: 0,
            last_duration: 0.0,
        }
    }
    
    pub fn update(&mut self, message: impl Into<String>) {
        self.completed_examples += 1;
        self.bar.inc(1);
        
        let current_rate = self.last_tokens as f64 / self.last_duration;
        let overall_rate = self.tokens_generated as f64 / self.total_duration;
        
        let msg = format!("{} - Current: {:.2} tokens/sec, Total: {} tokens in {:.2}s ({:.2} tokens/sec)", 
            message.into(), 
            current_rate,
            self.tokens_generated,
            self.total_duration,
            overall_rate);
        self.bar.set_message(msg);
    }
    
    pub fn add_tokens(&mut self, tokens: usize, duration: f64) {
        self.last_tokens = tokens;
        self.last_duration = duration;
        self.tokens_generated += tokens;
        self.total_duration += duration;
    }
    
    pub fn finish(&self, _message: impl Into<String>) {
        self.bar.finish();
        
        debug!("Generated {} tokens across {} examples in {:.2}s ({:.2} tokens/sec)",
               self.tokens_generated, 
               self.completed_examples,
               self.total_duration, 
               self.tokens_generated as f64 / self.total_duration);
    }
} 