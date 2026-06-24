pub mod lin_alg;
use lin_alg::Matrix;

fn main() {
    let m1: Matrix = Matrix::from_array(vec![vec![0.0, 1.0],
                                             vec![2.0, 3.0]]);

    let m2: Matrix = Matrix::from_array(vec![vec![1.0, 2.0],
                                             vec![3.0, 4.0]]);

    let mut p: Matrix = Matrix::multiply(&m1, &m2).expect("Impossible.");
    p.print();
    println!();
    p = Matrix::hadamard(&m1, &m2).expect("impossible?");
    p.print();
}
