use cudarc::driver::{DeviceRepr, ValidAsZeroBits};


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn abs(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

unsafe impl DeviceRepr for Complex64 {}
unsafe impl ValidAsZeroBits for Complex64 {}
