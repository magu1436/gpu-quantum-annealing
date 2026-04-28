use std::{path::PathBuf, sync::Arc};

use cudarc::{driver::{CudaContext, CudaModule, CudaStream}, nvrtc::{CompileOptions, compile_ptx_with_opts}};

use crate::SimResult;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Worker {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
}

impl Worker {
    pub fn new(
        kernel_dir: &str,
        kernel_file: &str,
    ) -> SimResult<Self> {

        let kernel_dir_path = PathBuf::from(&kernel_dir);
        let kernel_file_path = kernel_dir_path.join(kernel_file);

        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let source = std::fs::read_to_string(kernel_file_path).unwrap();
        let module = ctx.load_module(
            compile_ptx_with_opts(
                &source,
                CompileOptions {
                    options: vec![
                        format!("--include-path={}", kernel_dir_path.to_str().unwrap()),
                        "--gpu-architecture=compute_86".to_string(),
                    ],
                    ..Default::default()
                },
            )?,
        )?;
        Ok(Worker {
            ctx,
            stream,
            module,
        })
    }
}