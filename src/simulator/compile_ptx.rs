use std::fs;

use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

pub fn compile_ptx(path: &str) -> Ptx {
    let kernel_src = fs::read_to_string(path).unwrap();

    let mut opts = CompileOptions::default();
    opts.include_paths = vec!["kernels".into()];

    compile_ptx_with_opts(&kernel_src, opts).unwrap()
}