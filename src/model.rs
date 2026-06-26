use super::{
    lin_alg::Matrix,
    data::{Image, Label}
};
use rand_distr::{Distribution, Normal};

struct Layer {
    weights: Matrix,
    biases: Matrix,
}

pub trait ActivationFunction {
    fn f(x: &Matrix) -> Matrix;
    fn prime(x: &Matrix) -> Matrix;
}

pub struct ReLU;
impl ActivationFunction for ReLU {
    fn f(x: &Matrix) -> Matrix { 
        x.component_operation(|a| a.max(0.0)) 
    }
    fn prime(x: &Matrix) -> Matrix { 
        x.component_operation(|a| f32::from(a > 0.0)) 
    }
}

pub struct Model {
    layers: Vec<Layer>,
}

pub struct ForwardPass {
    pub output: Matrix,
    pub activations: Vec<Matrix>,
    pub preactivations: Vec<Matrix>
}

impl Model {

    // He initialization: sample from normal w/ mean 0, sqrt(2/weight inputs)
    // layer_sizes includes the input layer
    pub fn new(layer_sizes: &[usize]) -> Model {
        let mut rng = rand::rng();

        let layers: Vec<Layer> = layer_sizes
            .windows(2)
            .map(|w| {
                let normal = Normal::new(0.0, (2.0 / w[0] as f32).sqrt()).unwrap();
                let weights: Vec<f32> = normal.sample_iter(&mut rng).take(w[0] * w[1]).collect();
                Layer {
                    weights: Matrix::from_array(&weights, w[1], w[0]).unwrap(),
                    biases: Matrix::zeros(w[1], 1),
                }
            })
            .collect();
        
        Model {
            layers,
        }
    }

    pub fn feed_forward<T: ActivationFunction>(&self, input: &Matrix) -> ForwardPass {
        // either N x 1 or N x batch_size
        let mut current = input.clone();

        let mut activations = vec![current.clone()];
        let mut preactivations = vec![];

        for layer in &self.layers {
            current = layer.weights
                .multiply(&current).unwrap()
                .add(&layer.biases).unwrap();
            preactivations.push(current.clone());
            current = T::f(&current);
            activations.push(current.clone());
        }

        ForwardPass {
            output: current,
            activations,
            preactivations,
        }
    }

    // Returns accuracy on test data
    pub fn evaluate(&self, images: &[Image], labels: &[Label]) -> f32 {
        let image_matrix = Matrix::from_columns(&{
            images
                .iter()
                .map(|i| i.pixels.flatten())
                .collect::<Vec<_>>()
        }).unwrap();
        let ForwardPass {output, ..} = self.feed_forward::<ReLU>(&image_matrix);
        
        let max = output.argmax_cols();
        max.iter()
            .zip(labels)
            .fold(0, |a, (b,  c)| a + (*b == c.num as usize) as usize)
            as f32 / labels.len() as f32
    }

    pub fn update_batch(&mut self, learning_rate: f32, batch_images: &Matrix, batch_expected: &Matrix) {
        let forward_pass: ForwardPass = self.feed_forward::<ReLU>(batch_images);
        // calculate gradients
        let (nabla_w, nabla_b) = self.backprop::<ReLU>(&forward_pass, batch_expected);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.weights = layer.weights.subtract(&nabla_w[i].scale(learning_rate)).unwrap();
            layer.biases = layer.biases.subtract(&nabla_b[i].scale(learning_rate)).unwrap();
        }   
    }

    // Using the quadratic cost function 0.5(a - y)^2
    fn backprop<T: ActivationFunction>(&self, forward_pass: &ForwardPass,  expected_output: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut nabla_w: Vec<Matrix> = Vec::with_capacity(self.layers.len());
        let mut nabla_b: Vec<Matrix> = Vec::with_capacity(self.layers.len());
        let n = expected_output.cols as f32;

        let ForwardPass {output, activations, preactivations} = forward_pass;
        let num_layers = self.layers.len(); // num hidden layers + output layer

        // error = dC/dz
        let mut error = output.subtract(expected_output).unwrap()
            .hadamard(&T::prime(&preactivations[preactivations.len() - 1])).unwrap();
        nabla_w.push(error.outer(&activations[activations.len() - 2]).unwrap().scale(1.0 / n));
        nabla_b.push(error.sum_along_rows().scale(1.0 / n));

        // the rest of the model
        // l is the offset from the end of the network
        for l in 1..num_layers {
            error = self.layers[num_layers - l].weights.transpose()
                .multiply(&error).unwrap()
                .hadamard(&T::prime(&preactivations[num_layers - 1 - l])).unwrap();
            nabla_w.push(error.outer(&activations[activations.len() - l - 2]).unwrap().scale(1.0 / n));
            nabla_b.push(error.sum_along_rows().scale(1.0 / n));

        }
        
        (nabla_w.into_iter().rev().collect(), nabla_b.into_iter().rev().collect())
    }
}