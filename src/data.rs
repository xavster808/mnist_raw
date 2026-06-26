use super::lin_alg::Matrix;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MNIST_IMAGES_MAGIC_NUMBER: u32 = 0x00000803; // unsigned u8, 3 fields = 2051
const MNIST_LABELS_MAGIC_NUMBER: u32 = 0x00000801; // unsigned u8, 1 field = 2049

#[derive(Clone)]
pub struct Image {
    pub pixels: Matrix,
}

impl Image {
    pub fn display_image(&self) {
        const PIXEL_WIDTH: usize = 2;
        let data: &Matrix = &self.pixels;
        let mut display: String = String::with_capacity(data.rows * data.cols * PIXEL_WIDTH);

        for row in 0..data.rows {
            for col in 0..data.cols {
                let pixel = data.at(row, col);
                let symbol = if pixel > 0.8 {
                    '█'
                } else if pixel > 0.6 {
                    '▓'
                } else if pixel > 0.4 {
                    '▒'
                } else if pixel > 0.2 {
                    '░'
                } else {
                    ' '
                };
                display.push(symbol);
                display.push(symbol);
            }
            display.push('\n');
        }
        display.pop();
        print!("{}", display);
    }
}

pub struct Label {
    pub num: u8,
}

impl Label {
    pub fn display_label(&self) {
        print!("{}", self.num);
    }

    pub fn to_onehot(&self) -> Matrix {
        let mut column = Matrix::zeros(10, 1);
        *column.at_mut(self.num as usize, 0) = 1.0;
        column
    }
}

pub fn image_reader(data_path: &Path) -> Result<Vec<Image>, Box<dyn std::error::Error>> {
    let mut images: Vec<Image> = vec![];
    let mut data_file = File::open(data_path).expect("Bad data file path!");

    let mut buffer = [0u8; 4];

    data_file.read_exact(&mut buffer)?;
    let magic_number = u32::from_be_bytes(buffer);
    if magic_number != MNIST_IMAGES_MAGIC_NUMBER {
        return Err("Bad magic number!".into());
    }

    data_file.read_exact(&mut buffer)?;
    let num_images = u32::from_be_bytes(buffer);

    data_file.read_exact(&mut buffer)?;
    let num_rows = u32::from_be_bytes(buffer) as usize;

    data_file.read_exact(&mut buffer)?;
    let num_cols = u32::from_be_bytes(buffer) as usize;

    for _ in 0..num_images {
        let mut pixels = vec![0u8; num_rows * num_cols];
        data_file.read_exact(&mut pixels)?;
        let normalized: Vec<f32> = pixels.iter().map(|&p| (p as f32) / 255.0).collect();

        images.push(Image {
            pixels: Matrix::from_array(&normalized, num_rows, num_cols)?,
        });
    }

    Ok(images)
}

pub fn label_reader(label_path: &Path) -> Result<Vec<Label>, Box<dyn std::error::Error>> {
    let mut label_file = File::open(label_path).expect("Bad label file path!");

    let mut buffer = [0u8; 4];

    label_file.read_exact(&mut buffer).expect("Bad file data!");
    let magic_number = u32::from_be_bytes(buffer);
    if magic_number != MNIST_LABELS_MAGIC_NUMBER {
        return Err("Bad magic number!".into());
    }

    label_file.read_exact(&mut buffer)?;
    let num_labels = u32::from_be_bytes(buffer);

    let mut labels: Vec<u8> = vec![0; num_labels as usize];
    label_file.read_exact(&mut labels)?;
    let labels: Vec<Label> = labels.iter().map(|&b| Label { num: b }).collect();

    Ok(labels)
}
