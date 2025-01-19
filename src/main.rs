use std::fs;
use tch::{Tensor, Kind};

mod tokenizer {
    pub mod vocab;
    pub mod tokenizer;
}

mod model {
    pub mod transformer;
    pub mod training;
}

fn main() {
    println!("Step 1: Loading Dataset...");
    let dataset_path = "path_to_dataset.txt";
    let train_texts = fs::read_to_string(dataset_path).expect("Failed to load dataset");
    println!("Dataset loaded. Number of samples: {}", train_texts.lines().count());

    println!("Step 2: Preprocessing Text...");
    let subset_size = 2000;
    let train_lines: Vec<_> = train_texts.lines().take(subset_size).collect();
    let corpus: String = train_lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join(" ");
    println!("Corpus length (characters): {}", corpus.len());

    println!("Step 3: Building Vocabulary and Initializing Tokenizer...");
    let vocab = tokenizer::vocab::build_vocab(&corpus, 3000, &["<PAD>", "<UNK>", "<EOS>"], true);
    println!("Vocabulary size: {}", vocab.len());
    let tokenizer = tokenizer::tokenizer::Tokenizer::new(vocab.clone(), "<UNK>", "<PAD>", "<EOS>");

    println!("Step 4: Preparing Training Data...");
    let sentences: Vec<_> = corpus.split(". ").collect();
    println!("Number of sentences: {}", sentences.len());

    let mut data = Vec::new();
    for sentence in sentences {
        let tokens = tokenizer.tokenize(sentence);
        let token_ids = tokenizer.encode(&tokens);
        if token_ids.len() > 1 {
            let input_ids = Tensor::of_slice(&token_ids[..token_ids.len() - 1]).to_kind(Kind::Int);
            let target_ids = Tensor::of_slice(&token_ids[1..]).to_kind(Kind::Int);
            data.push((input_ids, target_ids));
        }
    }
    println!("Number of training samples: {}", data.len());

    println!("Step 5: Defining Model Parameters...");
    let vocab_size = vocab.len() as i64;
    let max_len = data.iter().map(|(input, _)| input.size()[0]).max().unwrap_or(0) + 1;
    println!("Maximum sequence length: {}", max_len);

    let embed_dim = 128;
    let num_heads = 4;
    let num_layers = 2;
    let hidden_dim = 256;
    println!("Model parameters set. Vocab: {}, Embed: {}, Heads: {}, Layers: {}, Hidden: {}", vocab_size, embed_dim, num_heads, num_layers, hidden_dim);

    println!("Step 6: Initializing Transformer...");
    let mut model = model::transformer::Transformer::new(vocab_size, embed_dim, max_len, num_heads, num_layers);
    println!("Transformer initialized.");

    println!("Step 7: Skipping detailed training example.");

    println!("Step 8: Performing Inference...");
    let prompt = "This is";
    let prompt_tokens = tokenizer.tokenize(prompt);
    let prompt_ids = tokenizer.encode(&prompt_tokens);
    let prompt_ids_tensor = Tensor::of_slice(&prompt_ids).to_kind(Kind::Int);
    println!("Prompt token IDs (on CPU): {:?}", prompt_ids);

    let generated_ids = model.generate(&prompt_ids_tensor, 20, &tokenizer, 1.0, 1.2);
    let generated_text = tokenizer.decode(&generated_ids);
    println!("Generated Text: {}", generated_text);
}
