use crate::{BertRunnerTrait, label::Label};
use std::path::PathBuf;
use tracing::{instrument, info};

pub struct BertRunner;

impl BertRunnerTrait for BertRunner {
    #[instrument(skip(_model_path, _config_path, _vocab_path))]
    fn new(_model_path: PathBuf, _config_path: PathBuf, _vocab_path: PathBuf) -> Self {
        info!("Creating mock BERT runner");
        Self
    }
    
    #[instrument(skip(self))]
    fn load_model(&mut self) -> anyhow::Result<()> {
        info!("Mock model loaded");
        Ok(())
    }
    
    #[instrument(skip(self, _input), fields(input_len = _input.len()))]
    fn predict(&self, _input: Vec<String>) -> anyhow::Result<Vec<Label>> {
        info!("Making mock predictions for {} inputs", _input.len());
        Ok(vec![Label {
            text: "toxic".into(),
            score: 0.99,
            id: 0,
            sentence: 0,
        }])
    }
}