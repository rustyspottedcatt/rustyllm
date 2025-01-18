use regex::Regex;
use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    unk_token: String,
    pad_token: String,
    eos_token: String,
}

impl Tokenizer {
    pub fn new(vocab: HashMap<String, i64>, unk_token: &str, pad_token: &str, eos_token: &str) -> Self {
        let id_to_token = vocab.iter().map(|(k, v)| (*&v, k.clone())).collect();
        Self {
            vocab,
            id_to_token,
            unk_token: unk_token.to_string(),
            pad_token: pad_token.to_string(),
            eos_token: eos_token.to_string(),
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        Regex::new(r"\w+|[^\w\s]")
            .unwrap()
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    pub fn encode(&self, tokens: &[String]) -> Vec<i64> {
        tokens
            .iter()
            .map(|t| *self.vocab.get(t).unwrap_or(&self.vocab[&self.unk_token]))
            .collect()
    }

    pub fn decode(&self, token_ids: &[i64]) -> String {
        token_ids
            .iter()
            .map(|id| self.id_to_token.get(id).unwrap_or(&self.unk_token).clone())
            .collect::<Vec<_>>()
            .join(" ")
    }
}
