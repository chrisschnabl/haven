use rust_bert::resources::LocalResource;
use rust_bert::pipelines::sequence_classification::{SequenceClassificationConfig, SequenceClassificationModel};
use rust_bert::pipelines::common::ModelResource;
use crate::{BertRunnerTrait, label::Label};
use std::path::PathBuf;
use tracing::info;

pub struct BertRunner {
    model: Option<SequenceClassificationModel>,
    model_path: PathBuf,
    config_path: PathBuf,
    vocab_path: PathBuf,
}

impl BertRunner {
    pub fn with_paths(model_path: PathBuf, config_path: PathBuf, vocab_path: PathBuf) -> Self {
        Self { 
            model: None, 
            model_path,
            config_path,
            vocab_path,
        }
    }
}

impl BertRunnerTrait for BertRunner {
    fn new(model_path: PathBuf, config_path: PathBuf, vocab_path: PathBuf) -> Self {
        Self { 
            model: None, 
            model_path,
            config_path,
            vocab_path,
        }
    }

    fn load_model(&mut self) -> anyhow::Result<()> {
        let model_resource = ModelResource::Torch(Box::new(LocalResource {
            local_path: self.model_path.clone(),
        }));
        let config_resource = Box::new(LocalResource {
            local_path: self.config_path.clone(),
        });
        let vocab_resource = Box::new(LocalResource {
            local_path: self.vocab_path.clone(),
        });

        let custom_config = SequenceClassificationConfig {
            model_resource,
            vocab_resource,
            config_resource,
            ..Default::default()
        };

        self.model = Some(SequenceClassificationModel::new(custom_config)?);
        info!("Model loaded");
        Ok(())
    }

    fn predict(&self, input: Vec<String>) -> anyhow::Result<Vec<Label>> {
        match &self.model {
            Some(model) => {
                let input_refs: Vec<&str> = input.iter().map(String::as_str).collect();
                let predictions = model.predict(input_refs);
                Ok(predictions.into_iter().map(|real_label| {
                    Label {
                        text: real_label.text.clone(),
                        score: real_label.score,
                        id: real_label.id,
                        sentence: real_label.sentence,
                    }
                }).collect::<Vec<_>>())
            }
            None => Err(anyhow::anyhow!("Model not loaded")),
        }
    }
}
