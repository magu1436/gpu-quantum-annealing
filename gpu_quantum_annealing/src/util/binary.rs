use std::{fs::read, path::Path};
use std::io::{Error, Result, ErrorKind};

pub fn read_f64_le_bin<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    const F64_SIZE: usize = std::mem::size_of::<f64>();
    
    let bytes = read(path)?;

    if (bytes.len() % F64_SIZE) != 0 {
        return Err(Error::new(ErrorKind::InvalidData, "invalid data"));
    };

    let values: Vec<f64> = bytes
        .chunks_exact(F64_SIZE)
        .map(|chunk| {
            let array: [u8; F64_SIZE] = chunk
                .try_into()
                .expect("failed to convert chunk to array");
            f64::from_le_bytes(array)
        })
        .collect();

    Ok(values)
}