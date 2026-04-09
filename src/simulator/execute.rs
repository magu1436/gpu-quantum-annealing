use std::time;

use cudarc::driver::{CudaContext, PushKernelArg};

use crate::simulator::{compile_ptx::compile_ptx, complex::Complex64, config::AnnealingConfig, launch_config::{KernelLayout, create_launch_config}};


pub fn excute<F>(bit_count: usize, objective_function: F, config: AnnealingConfig) -> Vec<f64>
where
    F: Fn(usize) -> f64,
{

    let start_time = time::Instant::now();
    print!("\nExecuting quantum annealing simulation...\n");

    // 定数
    let step = (config.tau / config.dt) as u32;
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

    let develop_time = match config.threads_x < 32 {
        true => module.load_function("develop_time").unwrap(),
        false => module.load_function("develop_time_warp").unwrap(),
    };
    let calc_norm = module.load_function("add_to_calc_norm").unwrap();
    let update_f0 = module.load_function("update_f0").unwrap();

    let mut f0_dev = stream.clone_htod(&f0).unwrap();
    let mut f1_dev = stream.alloc_zeros::<Complex64>(n).unwrap();
    let diag_dev = stream.clone_htod(&diag).unwrap();
    let mut sum = stream.alloc_zeros::<f64>(1).unwrap();

    let cfg_for_vector = create_launch_config(n, config.threads_x, KernelLayout::Vector2D);
    let cfg_for_develop_time = match config.threads_x < 32 {
        true => create_launch_config(n, config.threads_x, KernelLayout::Vector2D),
        false => create_launch_config(n, config.threads_x, KernelLayout::Warp),
    };

    let mut t: f64;
    for i in 0..step {
        t = (i as f64) * config.dt;
        let a = t / config.tau;
        let b = config.b0 * (1.0 - a);

        unsafe  {

            match stream
                .launch_builder(&develop_time)
                .arg(&a)
                .arg(&b)
                .arg(&config.dt)
                .arg(&diag_dev)
                .arg(&f0_dev)
                .arg(&n)
                .arg(&(bit_count as u32))
                .arg(&f1_dev)
                .launch(cfg_for_develop_time) {
                    Ok(_) => {},
                    Err(e) => panic!("Develop time error: {}", e)
                };

            match stream.memcpy_htod(&[0.0], &mut sum) {
                Ok(_) => {},
                Err(e) => panic!("Memcpy error: {}", e)
            };
            std::mem::swap(&mut f0_dev, &mut f1_dev);

            match stream
                .launch_builder(&calc_norm)
                .arg(&f0_dev)
                .arg(&sum)
                .arg(&n)
                .launch(cfg_for_vector) {
                    Ok(_) => {},
                    Err(e) => panic!("Calc norm error: {}", e)
                };

            match stream
                .launch_builder(&update_f0)
                .arg(&f0_dev)
                .arg(&sum)
                .arg(&n)
                .launch(cfg_for_vector) {
                    Ok(_) => {},
                    Err(e) => panic!("Update f0 error: {}", e)
                };
        }
    }

    let elapsed = start_time.elapsed();
    print!(
        "Quantum annealing simulation completed in {:.2?} seconds.\n",
        elapsed
    );

    stream.synchronize().unwrap();
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