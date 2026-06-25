pub mod lin_alg;
pub mod data;
pub mod model;
pub mod train;

#[cfg(test)]
pub mod tests;

use std::path::Path;
use data::{image_reader, label_reader, Image, Label};

const TRAIN_IMAGES: &str = "./mnist-dataset/train-images.idx3-ubyte";
const TRAIN_LABELS: &str = "./mnist-dataset/train-labels.idx1-ubyte";
const TEST_IMAGES: &str = "./mnist-dataset/t10k-images.idx3-ubyte";
const TEST_LABELS: &str = "./mnist-dataset/t10k-labels.idx1-ubyte";

fn main() {
    let _test_images: Vec<Image> = image_reader(Path::new(TEST_IMAGES)).unwrap();
    let _test_labels: Vec<Label> = label_reader(Path::new(TEST_LABELS)).unwrap();
    let _train_images: Vec<Image> = image_reader(Path::new(TRAIN_IMAGES)).unwrap();
    let _train_labels: Vec<Label> = label_reader(Path::new(TRAIN_LABELS)).unwrap();
}
