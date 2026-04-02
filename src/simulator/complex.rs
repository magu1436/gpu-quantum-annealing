use cudarc::driver::{DeviceRepr, ValidAsZeroBits};


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

unsafe impl DeviceRepr for Complex64 {}
unsafe impl ValidAsZeroBits for Complex64 {}
