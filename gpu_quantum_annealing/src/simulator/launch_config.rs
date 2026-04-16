use cudarc::driver::LaunchConfig;

pub enum KernelLayout {
    Vector2D,
    Matrix2D,
    Warp,
}

pub fn create_launch_config(n: usize, threads_x: u32, layout: KernelLayout) -> LaunchConfig {
    let warps_per_block = threads_x / 32;
    
    let block_n = match layout {
        KernelLayout::Vector2D => (n as u32 + threads_x - 1) / threads_x,
        KernelLayout::Matrix2D => (n as u32 + threads_x - 1) / threads_x,
        KernelLayout::Warp => (n as u32 + warps_per_block - 1) / warps_per_block,
    };

    let (block_dim, grid_dim) = match layout {
        KernelLayout::Vector2D => ((threads_x, 1, 1), (block_n, 1, 1)),
        KernelLayout::Matrix2D => ((threads_x, threads_x, 1), (block_n, block_n, 1)),
        KernelLayout::Warp => ((threads_x, 1, 1), (block_n, 1, 1)),
    };

    return LaunchConfig {
        block_dim,
        grid_dim,
        shared_mem_bytes: 0,
    };
}