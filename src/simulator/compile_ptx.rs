use std::fs;

use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

pub fn compile_ptx(path: &str) -> Ptx {
    let kernel_src = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(e) => {
            panic!("Failed to read file: {}", e);
        }
    };

    let options = vec![
        "--include-path=kernels".to_string(),
        "--gpu-architecture=compute_86".to_string(),
    ];

    let mut opts = CompileOptions {
        options,
        ..Default::default()
    };
    opts.include_paths = vec!["kernels".into()];

    match compile_ptx_with_opts(&kernel_src, opts) {
        Ok(v) => v,
        Err(e) => {
            panic!("Failed to compile ptx: {}", e);
        }
    }
}