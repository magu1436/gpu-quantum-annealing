use std::{fs, path::PathBuf};

use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::{QASimError, SimResult};

pub fn compile_ptx(file_name: &str) -> SimResult<Ptx> {
    let kernel_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let kernel_file = kernel_dir.join(file_name);
    let kernel_src = match fs::read_to_string(&kernel_file) {
        Ok(v) => v,
        Err(e) => {
            return Err(QASimError::KernelFileNotFound { file_path: kernel_file.to_string_lossy().into_owned(), err: e })
        }
    };

    let options = vec![
        format!("--include-path={}", kernel_dir.to_str().unwrap()),
        "--gpu-architecture=compute_86".to_string(),
    ];

    let mut opts = CompileOptions {
        options,
        ..Default::default()
    };
    opts.include_paths = vec!["kernels".into()];

    Ok(compile_ptx_with_opts(&kernel_src, opts)?)
}