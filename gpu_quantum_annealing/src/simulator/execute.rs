use std::time;

use crate::{config::{AnnealingConfig, DevelopTimeMethod}, simulator::{
    annealer::Annealer, compile_ptx::compile_ptx, complex::Complex64, launch_config::{KernelLayout, create_launch_config}, qa_sim_error::{ResultExt, SimResult}
}};

pub fn execute(diag: &Vec<f64>, config: AnnealingConfig) -> SimResult<Vec<f64>>
{

    let start_time = time::Instant::now();
    print!("\nExecuting quantum annealing simulation...\n");

    // 定数
    let step = (config.tau / config.dt) as u32;

    let ptx = compile_ptx("modules.cu")?;
    let mut annealer = Annealer::new(diag, &ptx, &config)?;


    let mut t: f64;
    for i in 0..step {
        t = (i as f64) * config.dt;
        let a = t / config.tau;
        let b = config.b0 * (1.0 - a);

        unsafe  {
            annealer.develop_time(&a, &b)?;
            annealer.swap();
            annealer.calc_norm()?;
            annealer.update_f0()?;
        }
    }
    annealer.stream.synchronize()?;

    let elapsed = start_time.elapsed();
    print!(
        "Quantum annealing simulation completed in {:.2?} seconds.\n",
        elapsed
    );

    let result = annealer.stream.clone_dtoh(&annealer.f0_dev)?;
    let prob = amplitudes_to_probabilities(result);
    Ok(prob)

}

fn use_warp(config: &AnnealingConfig) -> bool {
    match config.develop_time_method {
        DevelopTimeMethod::DevelopTime => false,
        DevelopTimeMethod::DevelopTimeWarp => true,
        DevelopTimeMethod::Default => config.threads_x < 32
    }
}

fn amplitudes_to_probabilities(amplitudes: Vec<Complex64>) -> Vec<f64> {
    let mut probabilities = vec![0.0; amplitudes.len()];
    for (i, v) in amplitudes.iter().enumerate() {
        probabilities[i] = v.abs().powi(2);
    }
    probabilities
}