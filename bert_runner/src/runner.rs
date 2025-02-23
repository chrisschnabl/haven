use rust_bert::resources::LocalResource;
use rust_bert::pipelines::sequence_classification::{SequenceClassificationConfig, SequenceClassificationModel};
use rust_bert::pipelines::common::ModelResource;
use crate::{BertRunnerTrait, label::Label};

pub struct BertRunner {
    model: Option<SequenceClassificationModel>,
}

impl BertRunnerTrait for BertRunner {
    fn new() -> Self {
        Self { model: None }
    }

    fn load_model(&mut self) -> anyhow::Result<()> {
        let model_resource = ModelResource::Torch(Box::new(LocalResource {
            local_path: "model/rust_model.ot".into(),  // TODO CS: paramterize this 
        }));
        let config_resource = Box::new(LocalResource {
            local_path: "model/config.json".into(),
        });
        let vocab_resource = Box::new(LocalResource {
            local_path: "model/vocab.txt".into(),
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
