use std::path::PathBuf;

pub mod actor;
pub mod score;
pub mod label {
    #[derive(Debug)]
    pub struct Label {
        pub text: String,
        pub score: f64,
        pub id: i64,
        pub sentence: usize,
    }
}

pub trait BertRunnerTrait {
    fn new(model_path: PathBuf, config_path: PathBuf, vocab_path: PathBuf) -> Self where Self: Sized;
    fn load_model(&mut self) -> anyhow::Result<()>;
    fn predict(&self, input: Vec<String>) -> anyhow::Result<Vec<label::Label>>;
}

#[cfg(feature = "use_rust_bert")]
pub mod runner;
#[cfg(feature = "use_rust_bert")]
pub use runner::BertRunner;

#[cfg(not(feature = "use_rust_bert"))]
pub mod mock_runner;
#[cfg(not(feature = "use_rust_bert"))]
pub use mock_runner::BertRunner;

pub use actor::{
    BertCommand,
    BertActorHandle,
    start_bert_actor,
};
