use super::{
    model::{Model},
    lin_alg::*,
    data::{Image, Label}
};
use rand::seq::SliceRandom;



pub fn train(model: &mut Model, train_data: Vec<Image>, train_labels: Vec<Label>, test_data: Vec<Image>, test_labels: Vec<Label>) {
    let learning_rate = 0.05;
    let batch_size = 100;
    let epochs = 100;
    let mut rng = rand::rng();

    let mut accuracy: f32;
    let mut best_acc: f32 = 0.0;

    // Convert to matrices
    let train_data: Vec<Matrix> = train_data
        .into_iter()
        .map(|i| i.pixels.flatten())
        .collect();
    let train_labels: Vec<Matrix> = train_labels
        .into_iter()
        .map(|l| l.to_onehot()).collect();
    let mut pairs: Vec<(Matrix, Matrix)> = train_data
        .into_iter()
        .zip(train_labels)
        .collect();

    for epoch in 0..epochs {
        println!("Starting epoch {}", epoch);
        pairs.shuffle(&mut rng);
        for batch in pairs.chunks(batch_size) {
            let (images, labels): (Vec<Matrix>, Vec<Matrix>) = batch.iter().cloned().unzip();
            let image_matrix = Matrix::from_columns(&images).unwrap();
            let label_matrix = Matrix::from_columns(&labels).unwrap();
            model.update_batch(learning_rate, &image_matrix, &label_matrix);
        }
        accuracy = model.evaluate(&test_data, &test_labels);
        best_acc = best_acc.max(accuracy);

        println!("Current accuracy: {}%", accuracy * 100.0);
        println!("Best accuracy: {}%", best_acc * 100.0);
    }
}