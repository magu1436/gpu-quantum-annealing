use std::time;

use cudarc::driver::CudaContext;

use crate::simulator::{compile_ptx::compile_ptx, complex::Complex64};


pub fn excute() {

    let start_time = time::Instant::now();
    print!("\nExecuting quantum annealing simulation...\n");

    // パラメータ
    let bit_amount = 5;

    // 定数
    let dt = 1e-3;
    let tau = 20.0;
    let step = (tau / dt) as u32;
    let b0 = 10.0;

    let n = 2u64.pow(bit_amount as u32) as u32;

    // 対角成分
    let mut diag = vec![0.0; n as usize];
    for i in 0..(n as usize) {
        diag[i] = objective_function(i);
    }

    let mut f0 = vec![vec![Complex64::new(1.0f64 / (n as f64).sqrt(), 0.0); n as usize]; n as usize]
    let mut f_temp = vec![vec![Complex64::default(); n as usize]; n as usize];

    
    let ptx = compile_ptx("kernels/execute.cu");
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = ctx.load_module(ptx).unwrap();
    let f = module.load_function("execute").unwrap();

    let f0_dev = stream.clone_htod(&f0).unwrap();

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