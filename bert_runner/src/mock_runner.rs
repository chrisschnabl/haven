use crate::{BertRunnerTrait, label::Label};
use std::path::PathBuf;

pub struct BertRunner;

impl BertRunnerTrait for BertRunner {
    fn new(model_path: PathBuf, config_path: PathBuf, vocab_path: PathBuf) -> Self {
        Self
    }
    
    fn load_model(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    
    fn predict(&self, _input: Vec<String>) -> anyhow::Result<Vec<Label>> {
        Ok(vec![Label {
            text: "toxic".into(),
            score: 0.99,
            id: 0,
            sentence: 0,
        }])
    }
}