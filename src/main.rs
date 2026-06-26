pub mod lin_alg;
pub mod data;
pub mod model;
pub mod train;

#[cfg(test)]
pub mod tests;

use std::path::Path;
use data::{image_reader, label_reader, Image, Label};
use model::Model;
use train::train;

const TRAIN_IMAGES: &str = "./mnist-dataset/train-images.idx3-ubyte";
const TRAIN_LABELS: &str = "./mnist-dataset/train-labels.idx1-ubyte";
const TEST_IMAGES: &str = "./mnist-dataset/t10k-images.idx3-ubyte";
const TEST_LABELS: &str = "./mnist-dataset/t10k-labels.idx1-ubyte";

fn main() {
    let test_images: Vec<Image> = image_reader(Path::new(TEST_IMAGES)).unwrap();
    let test_labels: Vec<Label> = label_reader(Path::new(TEST_LABELS)).unwrap();
    let train_images: Vec<Image> = image_reader(Path::new(TRAIN_IMAGES)).unwrap();
    let train_labels: Vec<Label> = label_reader(Path::new(TRAIN_LABELS)).unwrap();

    let mut model = Model::new(&[28*28, 16, 16, 10]);
    train(&mut model, train_images, train_labels, test_images, test_labels);
}
