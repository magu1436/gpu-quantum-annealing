use std::time;

use crate::{
    config::AnnealingConfig,
    simulator::{
        analyze::{
            AnalysisResult,
            analyze,
            AnalysisConfig,
        },
        annealer::Annealer,
        compile_ptx::compile_ptx,
        qa_sim_error::SimResult
    },
};

pub fn execute(diag: &Vec<f64>, annealing_config: AnnealingConfig, analyze_config: AnalysisConfig) -> SimResult<AnalysisResult>
{

    let start_time = time::Instant::now();
    print!("\nExecuting quantum annealing simulation...\n");

    // 定数
    let step = (annealing_config.tau / annealing_config.dt) as u32;

    let ptx = compile_ptx("modules.cu")?;
    let mut annealer = Annealer::new(diag, &ptx, &annealing_config)?;


    let mut t: f64;
    for i in 0..step {
        t = (i as f64) * annealing_config.dt;
        let a = t / annealing_config.tau;
        let b = annealing_config.b0 * (1.0 - a);

        unsafe  {
            annealer.develop_time(&a, &b)?;
            annealer.swap();
            annealer.calc_norm()?;
        }
    }
    annealer.stream.synchronize()?;

    let elapsed = start_time.elapsed();
    print!(
        "Quantum annealing simulation completed in {:.2?} seconds.\n",
        elapsed
    );

    let amplitudes = annealer.stream.clone_dtoh(&annealer.f0_dev)?;
    let result = analyze(amplitudes, analyze_config);
    Ok(result)

}