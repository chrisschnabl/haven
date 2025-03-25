use std::num::NonZeroU32;

pub struct LlamaConfig {
    pub model_path: Option<String>,
    pub seed: i64,
    pub threads: i32,
    pub context_size: NonZeroU32,
    pub n_len: i32,
    pub temp: f32,
    pub top_p: f32,
    pub skip_non_utf8: bool,
    pub truncate_if_context_full: bool,
}

impl LlamaConfig {
    pub fn new() -> Self {
        Self {
            model_path: None,
            seed: 1337,
            threads: 2,
            context_size: NonZeroU32::new(2048).unwrap(),
            n_len: 64,
            temp: 0.3,
            top_p: 0.75,
            skip_non_utf8: true,
            truncate_if_context_full: true,
        }
    }
}
