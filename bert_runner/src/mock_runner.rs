use crate::{BertRunnerTrait, label::Label};
pub struct BertRunner;

impl BertRunnerTrait for BertRunner {
    fn new() -> Self {
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