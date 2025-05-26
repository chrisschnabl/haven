use bert_runner::actor::{start_bert_actor};
use bert_runner::score::SimilarityModel;

// Example on how to use the actor handle
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (bert_actor, _join_handle) = start_bert_actor();

    bert_actor.load_model().await?;

    let input_texts = vec!["I hate freaking programming in Rust.".to_string()];

    let predictions = bert_actor.predict(input_texts).await?;
    for prediction in predictions {
        println!("{:?}", prediction);
    }

    let similarity_model = SimilarityModel::new().unwrap();
    let similarity = similarity_model.similarity("I hate freaking programming in Rust.", "I love programming in Rust.").unwrap();
    println!("Similarity: {}", similarity);

    Ok(())
}