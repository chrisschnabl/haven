use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use anyhow::Result;
use semanticsimilarity_rs::cosine_similarity;

pub struct SimilarityModel {
    model: rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel,
}

impl SimilarityModel {
    pub fn new() -> Result<Self> {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::DistiluseBaseMultilingualCased)
            .create_model()?;
        
        Ok(Self { model })
    }

    pub fn similarity(&self, sentence1: &str, sentence2: &str) -> Result<f64> {
        let sentences = [sentence1, sentence2];
        let embeddings_f32 = self.model.encode(&sentences)?;
        
        let embeddings: Vec<Vec<f64>> = embeddings_f32
            .iter()
            .map(|embedding| embedding.iter().map(|&val| val as f64).collect())
            .collect();
            
        let similarity = cosine_similarity(&embeddings[0], &embeddings[1], false);
        Ok(similarity)
    }

    pub fn calculate_similarities<'a, I>(&self, sentence_pairs: I) -> Result<Vec<f64>>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>
    {
        sentence_pairs.into_iter().try_fold(Vec::new(), |mut acc, (sentence1, sentence2)| -> Result<Vec<f64>> {
            let similarity = self.similarity(sentence1, sentence2)?;
            acc.push(similarity);
            Ok(acc)
        })
    }
}