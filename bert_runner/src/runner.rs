use rust_bert::resources::LocalResource;
use rust_bert::pipelines::sequence_classification::{SequenceClassificationConfig, SequenceClassificationModel};
use rust_bert::pipelines::common::ModelResource;
use crate::{BertRunnerTrait, label::Label};

pub struct BertRunner {
    model: Option<SequenceClassificationModel>,
    model_path: String = ".".to_string()
}

impl BertRunnerTrait for BertRunner {
    fn new(model_path: &str = ".".to_string()) -> Self {
        Self { model: None, model_path: model_path.to_string() }
    }

    fn load_model(&mut self) -> anyhow::Result<()> {
        let model_resource = ModelResource::Torch(Box::new(LocalResource {
            local_path: self.model_path.into() + "/rust_model.ot",  // TODO CS: paramterize this 
        }));
        let config_resource = Box::new(LocalResource {
            local_path: self.model_path.into() + "/config.json",
        });
        let vocab_resource = Box::new(LocalResource {
            local_path: self.model_path.into() + "/vocab.txt",
        });

        let custom_config = SequenceClassificationConfig {
            model_resource,
            vocab_resource,
            config_resource,
            ..Default::default()
        };

        self.model = Some(SequenceClassificationModel::new(custom_config)?);
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
