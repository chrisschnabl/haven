use bert_runner::actor::{start_bert_actor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (bert_actor, _join_handle) = start_bert_actor();

    bert_actor.load_model().await?;

    let input_texts = vec!["I hate freaking programming in Rust.".to_string()];

    let predictions = bert_actor.predict(input_texts).await?;
    for prediction in predictions {
        println!("{:?}", prediction);
    }

    Ok(())
}