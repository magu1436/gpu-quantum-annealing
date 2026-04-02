use std::fs;

use cudarc::nvrtc::{self, Ptx};

pub fn compile_ptx(kernel_src: &str) -> Ptx {
    let kernel_src = fs::read_to_string(kernel_src).unwrap();
    let ptx = nvrtc::compile_ptx(&kernel_src).unwrap();
    return ptx;
}