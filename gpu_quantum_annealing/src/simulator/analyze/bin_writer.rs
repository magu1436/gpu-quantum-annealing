use std::fs::File;
use std::io::{
    Result,
    Write,
};
use std::path::Path;

use crate::simulator::complex::Complex64;

pub fn write_f64_bin(path: &str, data: &[f64]) -> Result<()> {
    create_parent_dir(path)?;
    let mut file = File::create(path)?;
    file.write_all(bytemuck::cast_slice(data))?;
    Ok(())
}

pub fn write_complex64_bin(path: &str, data: &[Complex64]) -> Result<()> {
    create_parent_dir(path)?;
    let mut file = File::create(path)?;
    file.write_all(bytemuck::cast_slice(data))?;
    Ok(())
}

fn create_parent_dir(path: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}