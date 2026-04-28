use cudarc::driver::{LaunchConfig, PushKernelArg};
use gpu_quantum_annealing::Worker;

pub fn create_prisoners_dillemma_hll(
    num_players: u32,
    num_pen: u32,
    num_slack: u32,
    start_slack: u32,
    hyper_params: Vec<i32>,
) -> Vec<f64> {
    let kernel_dir = "src/sample/prisoners_dilemma/kernels/";
    let kernel_file = "prisoners_dilemma.cu";
    let worker = match Worker::new(kernel_dir, kernel_file) {
        Ok(w) => w,
        Err(e) => panic!("{}", e),
    };
    let func = match worker.module.load_function("create_prisoners_dilemma_diag") {
        Ok(f) => f,
        Err(e) => panic!("{}", e),
    };

    let bit_count = (num_players + num_slack * num_pen) as u32;
    let n: u32 = 2f64.powi(bit_count as i32) as u32;
    let threads_x: u32 = 256;
    let cfg = LaunchConfig {
        block_dim: (threads_x, 1, 1),
        grid_dim: ((n + threads_x - 1) / threads_x, 1, 1),
        shared_mem_bytes: 0,
    };

    let hyper_params_dev = match worker.stream.clone_htod(&hyper_params) {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    let hll_dev = match worker.stream.alloc_zeros::<f64>(n as usize) {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };

    unsafe {
        match worker.stream
            .launch_builder(&func)
            .arg(&num_players)
            .arg(&num_pen)
            .arg(&num_slack)
            .arg(&start_slack)
            .arg(&hyper_params_dev)
            .arg(&hll_dev)
            .arg(&n)
            .arg(&(hyper_params.len() as u32))
            .arg(&bit_count)
            .launch(cfg) {
                Ok(_) => (),
                Err(e) => panic!("{}", e),
        }
    };

    let hll = match worker.stream.clone_dtoh(&hll_dev) {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };

    hll

}