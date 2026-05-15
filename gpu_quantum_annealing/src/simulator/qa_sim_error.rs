use cudarc::{driver::DriverError, nvrtc::CompileError};
use thiserror::Error;

use crate::simulator::annealer::AnnealingError;

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

    #[error("Annealer error: {0} ")]
    Annealer(#[from] AnnealingError),
}

pub type SimResult<T> = Result<T, QASimError>;