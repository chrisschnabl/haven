use std::num::NonZeroU32;

pub struct LlamaConfig {
    pub model_path: Option<String>,
    pub seed: i64,
    pub threads: i32,
    pub context_size: NonZeroU32,
    pub n_len: i32,
}

impl LlamaConfig {
    // TODO CS: optimize config parameters
    pub fn new() -> Self {
        Self {
            model_path: None,
            seed: 1337,
            threads: 2,
            context_size: NonZeroU32::new(2048).unwrap(),
            n_len: 64,
        }
    }
}
