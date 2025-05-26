use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use anyhow::Result;
use tracing::{instrument, info, debug};

pub struct SimilarityModel {
    model: rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel,
}

// BERT Cosine Similarity Score Implementation
impl SimilarityModel {
    #[instrument]
    pub fn new() -> Result<Self> {
        info!("Initializing similarity model");
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::DistiluseBaseMultilingualCased)
            .create_model()?;
        
        info!("Similarity model initialized successfully");
        Ok(Self { model })
    }

    #[instrument(skip(v1, v2), fields(dim = v1.len()))]
    fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
        assert_eq!(v1.len(), v2.len(), "Vectors must have the same dimension");
        
        let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        
        let magnitude1: f64 = v1.iter().map(|a| a * a).sum::<f64>().sqrt();
        let magnitude2: f64 = v2.iter().map(|a| a * a).sum::<f64>().sqrt();
        
        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (magnitude1 * magnitude2)
    }

    #[instrument(skip(self, sentence1, sentence2))]
    pub fn similarity(&self, sentence1: &str, sentence2: &str) -> Result<f64> {
        info!("Calculating similarity between two sentences");
        let sentences = [sentence1, sentence2];
        let embeddings_f32 = self.model.encode(&sentences)?;
        
        let embeddings: Vec<Vec<f64>> = embeddings_f32
            .iter()
            .map(|embedding| embedding.iter().map(|&val| val as f64).collect())
            .collect();

        let similarity = Self::cosine_similarity(&embeddings[0], &embeddings[1]);
        debug!("Similarity score: {}", similarity);
        Ok(similarity)
    }

    #[instrument(skip(self, sentence_pairs))]
    pub fn calculate_similarities<'a, I>(&self, sentence_pairs: I) -> Result<Vec<f64>>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>
    {
        info!("Calculating similarities for multiple sentence pairs");
        sentence_pairs.into_iter().try_fold(Vec::new(), |mut acc, (sentence1, sentence2)| -> Result<Vec<f64>> {
            let similarity = self.similarity(sentence1, sentence2)?;
            acc.push(similarity);
            Ok(acc)
        })
    }
}