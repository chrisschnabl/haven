 // Start of Selection
use bert_cpp_2::context::params::BertContextParams;
use bert_cpp_2::bert_backend::BertBackend;
use bert_cpp_2::bert_batch::BertBatch;
use bert_cpp_2::model::params::BertModelParams;
use bert_cpp_2::model::BertModel;
use bert_cpp_2::model::{AddSpecial, SpecialTokens};
use bert_cpp_2::sampling::BertSampler;
use std::io::Write;

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn main() {
    let model_path = std::env::args().nth(1).expect("Please specify model path");
    let backend = BertBackend::init().unwrap();
    let params = BertModelParams::default();

    let prompt =
        "[CLS] User: Hello! How are you? [SEP] Assistant:".to_string();
    BertContextParams::default();
    let model =
        BertModel::load_from_file(&backend, model_path, &params).expect("unable to load model");
    let ctx_params = BertContextParams::default();
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .expect("unable to create the bert_context");
    let tokens_list = model
        .str_to_token(&prompt, AddSpecial::Always)
        .unwrap_or_else(|_| panic!("failed to tokenize {prompt}"));
    let max_len = 64;

    // create a bert_batch with size 512
    // we use this object to submit token data for encoding
    let mut batch = BertBatch::new(512, 1);

    let last_index = tokens_list.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // bert_encode will output embeddings only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last).unwrap();
    }
    ctx.encode(&mut batch).expect("bert_encode() failed");

    let mut current_len = batch.n_tokens();

    // The `Decoder`
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = BertSampler::greedy();

    while current_len <= max_len {
        // sample the next token
        {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            sampler.accept(token);

            // is it an end of sequence?
            if token == model.token_eos() {
                eprintln!();
                break;
            }

            let output_bytes = model.token_to_bytes(token, SpecialTokens::Tokenize).unwrap();
            // use `Decoder.decode_to_string()` to avoid the intermediate buffer
            let mut output_string = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            print!("{output_string}");
            std::io::stdout().flush().unwrap();

            batch.clear();
            batch.add(token, current_len, &[0], true).unwrap();
        }

        current_len += 1;

        ctx.encode(&mut batch).expect("failed to encode");
    }
}