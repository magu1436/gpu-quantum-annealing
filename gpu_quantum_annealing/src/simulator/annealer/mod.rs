mod annealer;
mod quadratic_annealer;
mod launch_config;
mod annealing_error;
pub use quadratic_annealer::QuadraticAnnealer;
pub use annealer::Annealer;
use launch_config::{
    create_launch_config,
    KernelLayout,
};
pub use annealing_error::{
    AnnealingError,
    AnnealingResult,
    ResultExt,
};