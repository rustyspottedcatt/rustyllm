use tch::{Device, IndexOp, Kind, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::Instant;

pub fn rowwise_softmax(v: &mut Tensor) {
    if v.dim() == 3 {
        let max_vals = v.max_dim(2, true).0;
        v.sub_(max_vals);
        let _ = v.exp_();
        let sums = v.sum_dim_intlist(&[2], true, Kind::Float);
        v.div_(sums);
    } else {
        let max_vals = v.max_dim(1, true).0;
        v.sub_(max_vals);
        let _ = v.exp_();
        let sums = v.sum_dim_intlist(&[1], true, Kind::Float);
        v.div_(sums);
    }
}

pub fn rowwise_cross_entropy(mut prob: &Tensor, targets: &Tensor) -> Tensor {
    let epsilon = 1e-9;
    if prob.dim() == 3 {
        let prob_clone = prob.clone();
        let (batch_size, seq_len, vocab_size) = (*prob_clone.size()[0], *prob_clone.size()[1], *prob_clone.size()[2]);
        let reshaped_prob = prob.view([batch_size * seq_len, vocab_size]);
        let reshaped_targets = targets.reshape(&[batch_size * seq_len]);
        let clipped = reshaped_prob.gather(1, &reshaped_targets.unsqueeze(-1), false).clamp(epsilon, 1.0);
        -clipped.log().mean(Kind::Float)
    } else {
        let clipped = prob.gather(1, &targets.unsqueeze(-1), false).clamp(epsilon, 1.0);
        -clipped.log().mean(Kind::Float)
    }
}

pub fn collate_batch(data: &[(Tensor, Tensor)], pad: i64) -> (Tensor, Tensor) {
    let max_len = data.iter().map(|(x, _)| x.size()[0]).max().unwrap_or(0);
    let batch_size = data.len();
    let x_tensor = Tensor::full(&[batch_size as i64, max_len], &pad, (Kind::Int, Device::Cpu));
    let y_tensor = Tensor::full(&[batch_size as i64, max_len], &pad, (Kind::Int, Device::Cpu));

    for (i, (x, y)) in data.iter().enumerate() {
        let length = x.size()[0];
        x_tensor.i((i as i64, 0..length)).copy_(x);
        y_tensor.i((i as i64, 0..length)).copy_(y);
    }
    (x_tensor, y_tensor)
}

pub fn train_single_epoch(
    embeddings: &mut Tensor,
    projection: &mut Tensor,
    data: &[(Tensor, Tensor)],
    _lr: f64,
    batch_size: usize,
) -> f64 {
    let mut rng = thread_rng();
    let mut shuffled_indices: Vec<_> = (0..data.len()).collect();
    shuffled_indices.shuffle(&mut rng);

    let mut total_loss = 0.0;
    let mut steps = 0;

    for batch_start in (0..data.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(data.len());
        let batch_indices = &shuffled_indices[batch_start..batch_end];
        let batch: Vec<_> = batch_indices.iter().map(|&i| data[i].clone()).collect();

        let (x, y) = collate_batch(&batch, 0);

        let flat_x = x.view([-1]);
        let embedded_x = embeddings
            .index_select(0, &flat_x)
            .reshape(&[&x.size()[0], &x.size()[1], &embeddings.size()[1]]);
        let mut logits = embedded_x.matmul(projection);
        rowwise_softmax(&mut logits);

        let loss = rowwise_cross_entropy(&logits, &y);
        total_loss += f64::try_from(&loss).unwrap_or(0.0);
        steps += 1;

        let batch_size = &logits.size()[0];
        let seq_len = &logits.size()[1];
        let vocab_size = &logits.size()[2];

        let mut logits_reshaped = logits.view([batch_size * seq_len, vocab_size]);
        let y_flat = y.view([&batch_size * &seq_len]);

        let _ = logits_reshaped.index_add_(
            1,
            &y_flat.unsqueeze(-1),
            &(-Tensor::ones(&[&batch_size * &seq_len, 1], (Kind::Float, logits.device()))),
        );
        logits_reshaped /= (&batch_size * &seq_len) as f64;

        let grad_embeddings = logits.matmul(&projection.transpose(0, 1));
        for b in 0..batch_size {
            for s in 0..seq_len {
                let embedding_idx = x.int64_value(&[&b, &s]);
                if embedding_idx == 0 {
                    continue;
                }
                let _ = embeddings.index_add_(
                    0,
                    &Tensor::from(&embedding_idx).to_device(Device::Cpu),
                    &(-grad_embeddings.i((&b, &s))),
                );
            }
        }
    }

    if steps > 0 {
        total_loss / steps as f64
    } else {
        0.0
    }
}

pub fn train_loop(
    embeddings: &mut Tensor,
    projection: &mut Tensor,
    data: &[(Tensor, Tensor)],
    epochs: usize,
    lr: f64,
    batch_size: usize,
    debug: bool,
) -> Vec<f64> {
    let mut losses = Vec::new();
    for epoch in 0..epochs {
        let epoch_loss = train_single_epoch(embeddings, projection, data, lr, batch_size);
        losses.push(epoch_loss);
        if debug {
            println!("Epoch {}: Loss = {:.6}", epoch + 1, epoch_loss);
        }
    }
    losses
}

pub fn train_model(
    embeddings: &mut Tensor,
    projection: &mut Tensor,
    data: &[(Tensor, Tensor)],
    epochs: usize,
    lr: f64,
    batch_size: usize,
    debug: bool,
) {
    if data.is_empty() || epochs == 0 {
        return;
    }

    let sample_size = (data.len() as f64 * 0.3) as usize;
    let sample_data = &data[..sample_size];

    let mut embeddings_sample = embeddings.shallow_clone();
    let mut projection_sample = projection.shallow_clone();

    let start = Instant::now();
    train_single_epoch(
        &mut embeddings_sample,
        &mut projection_sample,
        sample_data,
        lr,
        batch_size,
    );
    let elapsed = start.elapsed();
    let estimated_total_time = elapsed.as_secs_f64() / 0.3 * epochs as f64;

    println!(
        "Estimated total training time: {:.2} seconds ({:.2} minutes) for {} epochs.",
        estimated_total_time,
        estimated_total_time / 60.0,
        epochs
    );

    let training_start = Instant::now();
    let losses = train_loop(embeddings, projection, data, epochs, lr, batch_size, debug);
    let training_end = Instant::now();

    println!(
        "Training completed in {:.2} seconds for {} epochs.",
        training_end.duration_since(training_start).as_secs_f64(),
        epochs
    );

    if debug {
        for (epoch, loss) in losses.iter().enumerate() {
            println!("Epoch {}: Loss = {:.6}", epoch + 1, loss);
        }
    }
}
