use std::time;

use cudarc::driver::{CudaContext, PushKernelArg};

use crate::simulator::{compile_ptx::compile_ptx, complex::Complex64, launch_config::create_launch_config};


pub fn excute() -> Vec<Complex64>{

    let start_time = time::Instant::now();
    print!("\nExecuting quantum annealing simulation...\n");

    // パラメータ
    let bit_amount = 5;

    // 定数
    let dt = 1e-3;
    let tau = 20.0;
    let step = (tau / dt) as u32;
    let b0 = 10.0;
    let threads_x = 2u32;

    let n = 2u64.pow(bit_amount as u32) as usize;

    // 対角成分
    let mut diag = vec![0.0; n as usize];
    for i in 0..(n as usize) {
        diag[i] = objective_function(i);
    }

    let f0 = vec![Complex64::new(1.0f64 / (n as f64).sqrt(), 0.0); n];

    
    let ptx = compile_ptx("kernels/execute.cu");
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = ctx.load_module(ptx).unwrap();

    let develop_time = module.load_function("kernels/develop_time.cu").unwrap();
    let calc_norm = module.load_function("kernels/norm.cu").unwrap();
    let update_f0 = module.load_function("kernels/update_f0.cu").unwrap();
    let amplitudes_to_probabilities = module.load_function("kernels/amplitudes_to_probabilities.cu").unwrap();

    let f0_dev = stream.clone_htod(&f0).unwrap();
    let f_temp_dev = stream.alloc_zeros::<Complex64>(n).unwrap();
    let diag_dev = stream.clone_htod(&diag).unwrap();
    let mut sum = stream.alloc_zeros::<f64>(1).unwrap();

    let cfg = create_launch_config(n, threads_x);

    let mut t: f64;
    for i in 0..step {
        t = (i as f64) * dt;
        let a = t / tau;
        let b = b0 * (1.0 - a);

        unsafe  {
            stream
                .launch_builder(&develop_time)
                .arg(&f0_dev)
                .arg(&a)
                .arg(&b)
                .arg(&diag_dev)
                .arg(&n)
                .arg(&f_temp_dev)
                .launch(cfg)
                .unwrap();

            stream.memcpy_htod(&[0.0], &mut sum).unwrap();

            stream
                .launch_builder(&calc_norm)
                .arg(&f_temp_dev)
                .arg(&sum)
                .arg(&n)
                .launch(cfg)
                .unwrap();

            stream
                .launch_builder(&update_f0)
                .arg(&f_temp_dev)
                .arg(&sum)
                .arg(&n)
                .arg(&f0_dev)
                .launch(cfg)
                .unwrap();
        }
    }


    unsafe {
        stream
            .launch_builder(&amplitudes_to_probabilities)
            .arg(&f0_dev)
            .arg(&f_temp_dev)
            .launch(cfg)
            .unwrap();
    }

    let prob = stream.clone_dtoh(&f_temp_dev).unwrap();

    let elapsed = start_time.elapsed();
    print!(
        "Quantum annealing simulation completed in {:.2?} seconds.\n",
        elapsed
    );

    prob

}

fn objective_function(idx: usize) -> f64 {
    const NUMS: [i32; 3] = [1, 2, 3];
    let mut result = 0.0;

    let bit = |decimal: u32, idx: usize| -> u32 {
        let shift = NUMS.len() as u32 - 1 - (idx as u32);
        ((decimal >> shift) & 1) as u32
    };

    for i in 0..NUMS.len() {
        for j in i+1..NUMS.len() {
            let a = 2 * bit(idx as u32, i) - 1;
            let b = 2 * bit(idx as u32, j) - 1;
            result += (a * b * ((NUMS[i] * NUMS[j]) as u32)) as f64;
        }
    }
    result
}