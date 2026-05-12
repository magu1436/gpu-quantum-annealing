use cudarc::driver::{LaunchConfig, PushKernelArg};
use gpu_quantum_annealing::Worker;



pub fn create_integer_pertition_hll(nums: Vec<i32>) -> Vec<f64> {
    let bit_count = nums.len() as u32;
    let n = 2f64.powi(bit_count as i32) as usize;

    let kernel_dir = "src/sample/integer_pertition/kernels/";
    let kernel_file = "integer_pertition.cu";
    let w = match Worker::new(kernel_dir, kernel_file) {
        Ok(w) => w,
        Err(e) => panic!("{}", e),
    };
    let func = w.module.load_function("integer_partition").unwrap();

    let nums_dev = w.stream.clone_htod(&nums).unwrap();
    let diag_dev = w.stream.alloc_zeros::<f64>(n).unwrap();

    let threads_x: u32 = 256;
    let cfg = LaunchConfig {
        block_dim: (threads_x, 1, 1),
        grid_dim: (((n as u32) + threads_x - 1) / threads_x, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        match w.stream
            .launch_builder(&func)
            .arg(&nums_dev)
            .arg(&bit_count)
            .arg(&diag_dev)
            .launch(cfg) {
                Ok(_) => (),
                Err(e) => panic!("{}", e),
            }
    };

    let diag = w.stream.clone_dtoh(&diag_dev).unwrap();
    diag
}