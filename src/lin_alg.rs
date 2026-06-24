pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub array: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            array: vec![0.0; rows * cols],
        }
    }

    pub fn from_array(data: &[f64], rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            array: data.to_owned(),
        }
    }

    fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.array[self.index(row, col)]
    }

    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        let index = self.index(row, col);
        &mut self.array[index]
    }

    pub fn print(&self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                print!("{}\t", self.at(r, c));
            }
            println!();
        }
    }

    pub fn multiply(m1: &Matrix, m2: &Matrix) -> Result<Matrix, String> {
        if m1.cols != m2.rows {
            return Err(format!(
                "Bad dimensions: ({}, {}) * ({}, {})", 
                m1.rows, m1.cols, m2.rows, m2.cols
            ));
        }
        let mut product = Self::zeros(m1.rows, m2.cols);
        
        for row in 0..m1.rows {
            for col in 0..m2.cols {
                for i in 0..m1.cols {
                    *product.at_mut(row, col) += m1.at(row, i) * m2.at(i, col);
                }
            }
        }

        Ok(product)
    }

    pub fn hadamard(m1: &Matrix, m2: &Matrix) -> Result<Matrix, String> {
        if (m1.rows, m1.cols) != (m2.rows, m2.cols) {
            return Err(format!(
                "Bad dimensions: ({}, {}) * ({}, {})", 
                m1.rows, m1.cols, m2.rows, m2.cols
            ));
        }
        let mut product = Self::zeros(m1.rows, m1.cols);
        for row in 0..m1.rows {
            for col in 0..m1.cols {
                *product.at_mut(row, col) = m1.at(row, col) * m2.at(row, col);
            }
        }
        Ok(product)
    }
}