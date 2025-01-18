use tch::{Tensor, Kind};
use crate::tokenizer::tokenizer::Tokenizer;

pub struct Transformer {
    embeddings: Tensor,
    projection: Tensor,
}

impl Transformer {
    pub fn new(vocab_size: i64, embed_dim: i64, max_len: i64, num_heads: i64, num_layers: usize) -> Self {
        let embeddings = Tensor::randn(&[*vocab_size, *embed_dim], (Kind::Float, tch::Device::Cpu)) * 0.01;
        let projection = Tensor::randn(&[embed_dim, vocab_size], (Kind::Float, tch::Device::Cpu)) * 0.01;

        Self {
            embeddings,
            projection,
        }
    }

    pub fn generate(
        &self,
        prompt_ids: &Tensor,
        max_len: i64,
        tokenizer: &Tokenizer,
        temperature: f64,
        repetition_penalty: f64,
    ) -> Vec<i64> {
        let mut generated = prompt_ids.shallow_clone();
        for _ in 0..max_len {
            let logits = generated.matmul(&self.projection);
            let probs = logits.softmax(-1, Kind::Float);
            let next_token = probs.multinomial(1, true).squeeze();
            generated = Tensor::cat(&[generated, next_token.unsqueeze(0)], 0);
            if let Some(eos_id) = tokenizer.get("<EOS>") {
                if next_token.int64_value(&[]) == eos_id {
                    break;
                }
            }
        }
        generated.iter::<i64>().unwrap().collect()
    }
}
