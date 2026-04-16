use cudarc::{driver::DriverError, nvrtc::CompileError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QASimError {

    #[error("kernel file not found: {file_path} \n {err} ")]
    KernelFileNotFound {
        file_path: String,
        #[source]
        err: std::io::Error
    },

    #[error("compile ptx failed. \n {0}")]
    CompilePtxFailed(#[from] CompileError),

    #[error("Kernel not found: {kernel} \n{e} ")]
    KernelNotFound {
        kernel: &'static str,
        #[source]
        e: DriverError
    },

    #[error("Kernel launch failed: {kernel} \n{e}")]
    KernelLaunchFailed {
        kernel: &'static str,
        #[source]
        e: DriverError
    },

    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
}

pub type SimResult<T> = Result<T, QASimError>;

pub trait ResultExt<T> {
    fn kernel_not_found_err(self, kernel: &'static str) -> SimResult<T>;
    fn kernel_process_err(self, kernel: &'static str) -> SimResult<T>;
}

impl<T> ResultExt<T> for Result<T, DriverError> {
    fn kernel_not_found_err(self, kernel: &'static str) -> SimResult<T> {
        self.map_err(|e| QASimError::KernelNotFound { kernel, e })
    }
    fn kernel_process_err(self, kernel: &'static str) -> SimResult<T> {
        self.map_err(|e| QASimError::KernelLaunchFailed { kernel, e })
    }
}