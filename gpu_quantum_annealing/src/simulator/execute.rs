use std::{sync::{Arc, Mutex}, thread};

use crate::{
    ObserverConfig, ProgressData, config::AnnealingConfig, simulator::{
        analyze::{
            AnalysisConfig, AnalysisResult, analyze
        }, annealer::Annealer, compile_ptx::compile_ptx, observe::{Observer, ObserverState}, qa_sim_error::SimResult
    }
};

pub fn execute<T>(diag: &Vec<f64>, annealing_config: AnnealingConfig, analyze_config: AnalysisConfig, observer_config: ObserverConfig<T>) -> SimResult<AnalysisResult>
where
    T: Clone + Send + 'static,
{

    // 定数
    let step = (annealing_config.tau / annealing_config.dt) as u64;

    // observer の実行
    let state = Arc::new(Mutex::new(ObserverState {
        progress: ProgressData{total_step: step, current_step: 0, is_finished: false},
        user_data: observer_config.user_data.clone(),
    }));
    let state_for_observer = Arc::clone(&state);

    let observer_thread = thread::spawn(move || {
        let observer = Observer::new(observer_config, state_for_observer);
        observer.run();
    });

    // annealer の実行
    let ptx = compile_ptx("modules.cu")?;
    let mut annealer = Annealer::new(diag, &ptx, &annealing_config)?;

    let mut t: f64;
    let mut ratio: u8 = 0;
    {
        let mut p = state.lock().unwrap();
        p.progress.current_step = 0;
    }
    for i in 0..step {
        t = (i as f64) * annealing_config.dt;
        let a = t / annealing_config.tau;
        let b = annealing_config.b0 * (1.0 - a);

        unsafe  {
            annealer.develop_time(&a, &b)?;
            annealer.swap();
            annealer.calc_norm()?;
        }

        let new_ratio = (i * 100 / step) as u8;
        if ratio != new_ratio {
            let mut p = state.lock().unwrap();
            p.progress.current_step = i;
        }
        ratio = new_ratio;
    }
    annealer.stream.synchronize()?;
    {
        let mut p = state.lock().unwrap();
        p.progress.is_finished = true;
    }

    observer_thread.join().unwrap();

    let amplitudes = annealer.stream.clone_dtoh(&annealer.f0_dev)?;
    let result = analyze(amplitudes, analyze_config);
    Ok(result)

}