# MNIST MLP in Rust

A multilayer perceptron trained on the MNIST dataset, implemented in Rust without ML libraries.

## Architecture
 - 784 -> 128 -> 64 -> 10 fully connceted layers
 - ReLU activation function
 - Mini-batch SGD, quadratic cost function
 - Achieves ~97.5% test accuracy

## Usage
cargo run --release
