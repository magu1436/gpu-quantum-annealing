use std::time;

use cudarc::driver::{CudaContext, PushKernelArg};

use crate::simulator::{compile_ptx::compile_ptx, complex::Complex64, launch_config::{KernelLayout, create_launch_config}};


pub fn excute<F>(bit_count: usize, objective_function: F) -> Vec<f64>
where
    F: Fn(usize) -> f64,
{

    let start_time = time::Instant::now();
    print!("\nExecuting quantum annealing simulation...\n");

    // 定数
    let dt = 1e-3;
    let tau = 20.0;
    let step = (tau / dt) as u32;
    let b0 = 10.0;
    let threads_x = 2u32;

    let n = 2u64.pow(bit_count as u32) as usize;

    // 対角成分
    let mut diag = vec![0.0; n as usize];
    for i in 0..(n as usize) {
        diag[i] = objective_function(i);
    }

    let f0 = vec![Complex64::new(1.0f64 / (n as f64).sqrt(), 0.0); n];

    
    let ptx = compile_ptx("kernels/modules.cu");
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = ctx.load_module(ptx).unwrap();

    let develop_time = module.load_function("develop_time").unwrap();
    let calc_norm = module.load_function("add_to_calc_norm").unwrap();
    let update_f0 = module.load_function("update_f0").unwrap();

    let mut f0_dev = stream.clone_htod(&f0).unwrap();
    let mut f1_dev = stream.alloc_zeros::<Complex64>(n).unwrap();
    let diag_dev = stream.clone_htod(&diag).unwrap();
    let mut sum = stream.alloc_zeros::<f64>(1).unwrap();

    let cfg_for_vector = create_launch_config(n, threads_x, KernelLayout::Vector2D);

    let mut t: f64;
    for i in 0..step {
        t = (i as f64) * dt;
        let a = t / tau;
        let b = b0 * (1.0 - a);

        unsafe  {

            match stream
                .launch_builder(&develop_time)
                .arg(&a)
                .arg(&b)
                .arg(&dt)
                .arg(&diag_dev)
                .arg(&f0_dev)
                .arg(&n)
                .arg(&(bit_count as u32))
                .arg(&f1_dev)
                .launch(cfg_for_vector) {
                    Ok(_) => {},
                    Err(e) => panic!("Develop time error: {}", e)
                };
            stream.synchronize().unwrap();

            match stream.memcpy_htod(&[0.0], &mut sum) {
                Ok(_) => {},
                Err(e) => panic!("Memcpy error: {}", e)
            };
            match stream.memcpy_dtod(&f1_dev, &mut f0_dev) {
                Ok(_) => {},
                Err(e) => panic!("Memcpy error: {}", e)
            }
            stream.synchronize().unwrap();
            stream.memcpy_htod(&vec![Complex64::default(); n], &mut f1_dev).unwrap();

            match stream
                .launch_builder(&calc_norm)
                .arg(&f0_dev)
                .arg(&sum)
                .arg(&n)
                .launch(cfg_for_vector) {
                    Ok(_) => {},
                    Err(e) => panic!("Calc norm error: {}", e)
                };
            stream.synchronize().unwrap();

            match stream
                .launch_builder(&update_f0)
                .arg(&f0_dev)
                .arg(&sum)
                .arg(&n)
                .launch(cfg_for_vector) {
                    Ok(_) => {},
                    Err(e) => panic!("Update f0 error: {}", e)
                };
            stream.synchronize().unwrap();
        }
    }

    let elapsed = start_time.elapsed();
    print!(
        "Quantum annealing simulation completed in {:.2?} seconds.\n",
        elapsed
    );

    let result = stream.clone_dtoh(&f0_dev).unwrap();
    let prob = amplitudes_to_probabilities(result);
    prob

}

fn amplitudes_to_probabilities(amplitudes: Vec<Complex64>) -> Vec<f64> {
    let mut probabilities = vec![0.0; amplitudes.len()];
    for (i, v) in amplitudes.iter().enumerate() {
        probabilities[i] = v.abs().powi(2);
    }
    probabilities
}