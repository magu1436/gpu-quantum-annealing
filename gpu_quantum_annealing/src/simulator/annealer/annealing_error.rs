use cudarc::{
    driver::DriverError
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnnealingError {
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

    #[error("Invalid thread count, threads_x: {0}. Should be a power of 2")]
    InvalidThreadCount(u32),

    #[error("Invalid thread count for warp layout, threads_x: {0}. Should be at least 32")]
    InvalidThreadCountForWarp(u32),

    #[error("CUDA driver error: {0} ")]
    CudaDriverError(#[from] DriverError),
}

pub type AnnealingResult<T> = Result<T, AnnealingError>;

pub trait ResultExt<T> {
    fn kernel_not_found_err(self, kernel: &'static str) -> AnnealingResult<T>;
    fn kernel_process_err(self, kernel: &'static str) -> AnnealingResult<T>;
}

impl<T> ResultExt<T> for Result<T, DriverError> {
    fn kernel_not_found_err(self, kernel: &'static str) -> AnnealingResult<T> {
        self.map_err(|e| AnnealingError::KernelNotFound { kernel, e })
    }
    fn kernel_process_err(self, kernel: &'static str) -> AnnealingResult<T> {
        self.map_err(|e| AnnealingError::KernelLaunchFailed { kernel, e })
    }
}