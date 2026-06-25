pub mod lin_alg;
pub mod data_loader;
pub mod model;

#[cfg(test)]
pub mod tests;

use std::path::Path;

use data_loader::{image_reader, label_reader, Image, Label};

use crate::lin_alg::Matrix;

const TRAIN_IMAGES: &str = "./mnist-dataset/train-images.idx3-ubyte";
const TRAIN_LABELS: &str = "./mnist-dataset/train-labels.idx1-ubyte";
const TEST_IMAGES: &str = "./mnist-dataset/t10k-images.idx3-ubyte";
const TEST_LABELS: &str = "./mnist-dataset/t10k-labels.idx1-ubyte";

fn main() {
    let test_images: Vec<Image> = image_reader(Path::new(TEST_IMAGES)).unwrap();
    let test_labels: Vec<Label> = label_reader(Path::new(TEST_LABELS)).unwrap();
    let train_images: Vec<Image> = image_reader(Path::new(TRAIN_IMAGES)).unwrap();
    let train_labels: Vec<Label> = label_reader(Path::new(TRAIN_LABELS)).unwrap();
}
