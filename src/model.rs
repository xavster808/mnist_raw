use super::lin_alg::Matrix;

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
        x.component_operation(|a| f64::from(a > 0.0)) 
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
    pub fn feed_forward<T: ActivationFunction>(&self, input: Matrix) -> ForwardPass {
        // either N x 1 or N x batch_size
        let mut current = input;

        let mut activations = vec![current.clone()];
        let mut preactivations = vec![]; // first entry is to align indices with 

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
}

