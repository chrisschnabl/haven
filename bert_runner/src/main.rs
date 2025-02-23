use rust_bert::resources::{LocalResource};
use rust_bert::pipelines::sequence_classification::{SequenceClassificationConfig, SequenceClassificationModel};
use rust_bert::pipelines::common::{ModelResource};

fn main() -> anyhow::Result<()> {
    // Define your custom resources (local or remote)
    let model_resource = ModelResource::Torch(Box::new(LocalResource {
        local_path: "model/rust_model.ot".into(),
    }));
    let config_resource = Box::new(LocalResource {
        local_path: "model/config.json".into(),
    });
    let vocab_resource = Box::new(LocalResource {
        local_path: "model/vocab.txt".into(),
    });
    // Create a custom configuration. (Set model_type if needed.)
    let custom_config = SequenceClassificationConfig {
        model_resource,
        vocab_resource,
        config_resource,
        ..Default::default()
    };

    // Instantiate the model using your custom configuration.
    let sequence_classification_model = SequenceClassificationModel::new(custom_config)?;

    // Define your input text.
    let input = [
        "I love programming in Rust.",
    ];

    // Run model inference.
    let output = sequence_classification_model.predict(input);
    for label in output {
        println!("{label:?}");
    }

    Ok(())
}