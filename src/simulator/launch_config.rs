use cudarc::driver::LaunchConfig;

pub fn create_launch_config(n: usize, threads_x: u32) -> LaunchConfig {
    let block_n = (n as u32 + threads_x - 1) / threads_x;
    return LaunchConfig {
        block_dim: (threads_x, threads_x, 1),
        grid_dim: (block_n, block_n, 1),
        shared_mem_bytes: 0,
    };
}