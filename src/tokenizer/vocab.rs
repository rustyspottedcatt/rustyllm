use regex::Regex;
use std::collections::HashMap;

pub fn build_vocab(
    corpus: &str,
    vocab_size: usize,
    special_tokens: &[&str],
    include_punctuation: bool,
) -> HashMap<String, i64> {
    let re = Regex::new(r"\w+|[^\w\s]").unwrap();
    let mut token_counts = HashMap::new();
    for token in re.find_iter(corpus).map(|mat| mat.as_str()) {
        *token_counts.entry(token.to_string()).or_insert(0) += 1;
    }
    let mut sorted_tokens: Vec<(String, i64)> = token_counts.into_iter().collect();
    sorted_tokens.sort_by(|a, b| b.1.cmp(&a.1));
    let most_common_tokens: Vec<String> = sorted_tokens
        .iter()
        .take(vocab_size)
        .map(|(token, _)| token.clone())
        .collect();
    let mut vocab = HashMap::new();
    for (i, &st) in special_tokens.iter().enumerate() {
        vocab.insert(st.to_string(), i as i64);
    }
    for token in most_common_tokens {
        if !vocab.contains_key(&token) {
            let ix = vocab.len() as i64;
            vocab.insert(token, ix);
        }
    }
    if include_punctuation {
        let punctuation = [".", ",", "!", "?", ":", ";"];
        for &p in punctuation.iter() {
            if !vocab.contains_key(p) {
                let ix = vocab.len() as i64;
                vocab.insert(p.to_string(), ix);
            }
        }
    }
    vocab
}
