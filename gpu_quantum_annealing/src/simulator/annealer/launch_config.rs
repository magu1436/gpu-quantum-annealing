use cudarc::driver::LaunchConfig;

use crate::simulator::annealer::{
    AnnealingError,
    AnnealingResult,
};

pub enum KernelLayout {
    Vector2D,
    Warp,
}

pub fn create_launch_config(n: usize, threads_x: u32, layout: KernelLayout) -> AnnealingResult<LaunchConfig> {


    // エラーチェック
    if !threads_x.is_power_of_two() {
        return Err(AnnealingError::InvalidThreadCount(threads_x));
    }
    if matches!(layout, KernelLayout::Warp) && threads_x < 32 {
        return Err(AnnealingError::InvalidThreadCountForWarp(threads_x));
    }

    let warps_per_block = threads_x / 32;
    
    let block_n = match layout {
        KernelLayout::Vector2D => (n as u32 + threads_x - 1) / threads_x,
        KernelLayout::Warp => (n as u32 + warps_per_block - 1) / warps_per_block,
    };

    let (block_dim, grid_dim) = match layout {
        KernelLayout::Vector2D => ((threads_x, 1, 1), (block_n, 1, 1)),
        KernelLayout::Warp => ((threads_x, 1, 1), (block_n, 1, 1)),
    };

    let cfg = LaunchConfig {
        block_dim,
        grid_dim,
        shared_mem_bytes: 0,
    };
    Ok(cfg)
}