use super::lin_alg::*;

fn assert_matrix_eq(matrix: &Matrix, expected: &[f64], rows: usize, cols: usize) {
    assert_eq!(matrix.rows, rows);
    assert_eq!(matrix.cols, cols);
    assert_eq!(matrix.array, expected);
}

#[test]
fn test_zeros() {
    let matrix = Matrix::zeros(2, 3);
    assert_matrix_eq(&matrix, &[0.0; 6], 2, 3);
}

#[test]
fn test_from_array() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let matrix = Matrix::from_array(&data, 2, 2).unwrap();
    assert_matrix_eq(&matrix, &data, 2, 2);
    assert!(Matrix::from_array(&[1.0], 2, 2).is_err());
}

#[test]
fn test_at() {
    let matrix = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    assert_eq!(matrix.at(0, 0), 1.0);
    assert_eq!(matrix.at(1, 1), 4.0);
    assert_eq!(matrix.at(0, 1), 2.0);
    assert_eq!(matrix.at(1, 0), 3.0);
}

#[test]
fn test_at_mut() {
    let mut matrix = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    *matrix.at_mut(1, 0) = 9.0;
    assert_eq!(matrix.at(1, 0), 9.0);
}

#[test]
fn test_print() {
    let matrix = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    matrix.print();
}

#[test]
fn test_transpose() {
    let matrix = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let transposed = matrix.transpose();
    assert_matrix_eq(&transposed, &[1.0, 3.0, 2.0, 4.0], 2, 2);
}

#[test]
fn test_multiply() {
    let left = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let right = Matrix::from_array(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    let product = left.multiply(&right).unwrap();
    assert_matrix_eq(&product, &[19.0, 22.0, 43.0, 50.0], 2, 2);
}

#[test]
fn test_scale() {
    let matrix = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let scaled = matrix.scale(2.0);
    assert_matrix_eq(&scaled, &[2.0, 4.0, 6.0, 8.0], 2, 2);
}

#[test]
fn test_add() {
    let left = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let right_matrix = Matrix::from_array(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    let right_vector = Matrix::from_array(&[9.0, 10.0], 2, 1).unwrap();
    let sum_1 = left.add(&right_matrix).unwrap();
    let sum_2 = left.add(&right_vector).unwrap();
    assert_matrix_eq(&sum_1, &[6.0, 8.0, 10.0, 12.0], 2, 2);
    assert_matrix_eq(&sum_2, &[10.0, 11.0, 13.0, 14.0], 2, 2);
}

#[test]
fn test_subtract() {
    let left = Matrix::from_array(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    let right = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let difference = left.subtract(&right).unwrap();
    assert_matrix_eq(&difference, &[4.0, 4.0, 4.0, 4.0], 2, 2);
}

#[test]
fn test_hadamard() {
    let left = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let right = Matrix::from_array(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
    let product = left.hadamard(&right).unwrap();
    assert_matrix_eq(&product, &[2.0, 6.0, 12.0, 20.0], 2, 2);
}

#[test]
fn test_outer() {
    let left = Matrix::from_array(&[1.0, 2.0], 2, 1).unwrap();
    let right = Matrix::from_array(&[3.0, 4.0], 2, 1).unwrap();
    let product = left.outer(&right).unwrap();
    assert_matrix_eq(&product, &[3.0, 4.0, 6.0, 8.0], 2, 2);
}

#[test]
fn test_component_operation() {
    let matrix = Matrix::from_array(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let transformed = matrix.component_operation(|value| value + 1.0);
    assert_matrix_eq(&transformed, &[2.0, 3.0, 4.0, 5.0], 2, 2);
}