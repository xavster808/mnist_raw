#[derive(Clone)]
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

    pub fn from_array(data: &[f64], rows: usize, cols: usize) -> Result<Matrix, Box<dyn std::error::Error>> {
        if data.len() != rows * cols {
            return Err("Bad data size".into())
        };
        Ok(Matrix {
            rows,
            cols,
            array: data.to_owned(),
        })
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

    pub fn transpose(&self) -> Matrix {
        let mut new_array: Vec<f64> = vec![0.0; self.array.len()];
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_array[col * self.rows + row] = self.at(row, col);
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            array: new_array,
        }
    }

    pub fn multiply(&self, m2: &Matrix) -> Result<Matrix, String> {
        if self.cols != m2.rows {
            return Err(format!(
                "Bad dimensions: ({}, {}) x ({}, {})", 
                self.rows, self.cols, m2.rows, m2.cols
            ));
        }
        let mut product = Self::zeros(self.rows, m2.cols);
        
        for row in 0..self.rows {
            for col in 0..m2.cols {
                for i in 0..self.cols {
                    *product.at_mut(row, col) += self.at(row, i) * m2.at(i, col);
                }
            }
        }

        Ok(product)
    }

    pub fn scale(&self, scalar: f64) -> Matrix {
        Matrix {
            array: self.array
                .iter()
                .map(|a| a * scalar)
                .collect(),
            ..*self
        }
    }

    pub fn add(&self, m2: &Matrix) -> Result<Matrix, String> {
        if (self.rows != m2.rows) || (m2.cols != 1) && (m2.cols != self.cols) {
            return Err(format!(
                "Bad dimensions: ({}, {}) + ({}, {})", 
                self.rows, self.cols, m2.rows, m2.cols
            ));
        }
        let array = if m2.cols == self.cols {
            self.array
                .iter()
                .zip(&m2.array)
                .map(|(&a, &b)| a + b)
                .collect()
        } else {  // Broadcast
            self.array
                .iter()
                .enumerate()
                .map(|(i, a)| a + m2.array[i / self.cols])
                .collect()
        };
        Ok(Matrix {
            array,
            ..*self
        })
    }
    
    pub fn subtract(&self, m2: &Matrix) -> Result<Matrix, String> {
        if (self.rows, self.cols) != (m2.rows, m2.cols) {
            return Err(format!(
                "Bad dimensions: ({}, {}) - ({}, {})", 
                self.rows, self.cols, m2.rows, m2.cols
            ));
        }
        let array = self.array
            .iter()
            .zip(&m2.array)
            .map(|(&a, &b)| a - b)
            .collect();
        Ok(Matrix {
            array,
            ..*self
        })
    }
    
    pub fn hadamard(&self, m2: &Matrix) -> Result<Matrix, String> {
        if (self.rows, self.cols) != (m2.rows, m2.cols) {
            return Err(format!(
                "Bad dimensions: ({}, {}) * ({}, {})", 
                self.rows, self.cols, m2.rows, m2.cols
            ));
        }
        let product: Vec<f64> = self.array
            .iter()
            .zip(&m2.array)
            .map(|(a, &b)| a * b)
            .collect();
        Ok(Matrix {
            array: product,
            ..*self
        })
    }

    pub fn outer(&self, m2: &Matrix) -> Result<Matrix, String> {
        self.multiply(&m2.transpose())    
    }

    pub fn component_operation(&self, operation: impl Fn(f64) -> f64) -> Matrix {
        Matrix {
            array: self.array
                .iter()
                .map(|&f| operation(f))
                .collect(),
            ..*self
        }
    }

    pub fn from_columns(columns: &[Matrix]) -> Result<Matrix, String> {
        if columns.is_empty() {
            return Err("No columns".into())
        }
        let rows = columns[0].rows;
        if columns.iter().any(|c| c.rows != rows) {
            return Err("Columns aren't the same length".into())
        }
        let cols = columns.len();

        let mut array = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in columns {
                array.push(col.array[row]);
            }
        }
        Ok(Matrix {
            rows,
            cols,
            array,
        })
    }

    pub fn sum_along_rows(&self) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: 1,
            array: self.array.chunks(self.cols).map(|c| c.iter().sum()).collect(),
        }
    }
}
