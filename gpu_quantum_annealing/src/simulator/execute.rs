use std::{sync::{Arc, Mutex}, thread};

use crate::{
    DevelopTimeMethod, ObserverConfig, ProgressData, config::AnnealingConfig, simulator::{
        analyze::{
            AnalysisConfig, AnalysisResult, analyze
        }, annealer::{
            Annealer,
            QuadraticAnnealer
        }, compile_ptx::compile_ptx, complex::Complex64, observe::Observer, qa_sim_error::SimResult
    }
};

pub fn execute(diag: &Vec<f64>, annealing_config: AnnealingConfig, analyze_config: AnalysisConfig, observer_config: ObserverConfig) -> SimResult<AnalysisResult>{

    // 定数
    let step = (annealing_config.tau / annealing_config.dt) as u64;

    // observer の実行
    let progress = Arc::new(Mutex::new(ProgressData {
        total_step: step,
        current_step: 0,
        is_finished: false
    }));
    let progress_for_observer = Arc::clone(&progress);

    let observer_thread = thread::spawn(move || {
        let mut observer = Observer::new(observer_config, progress_for_observer);
        observer.run();
    });

    // annealer の実行
    let ptx = compile_ptx("modules.cu")?;
    // let mut annealer = Annealer::new(diag, &ptx, &annealing_config)?;
    // let mut annealer = QuadraticAnnealer::new(diag, &ptx, &annealing_config)?;

    let amplitudes = match is_linear(&annealing_config) {
        true => {
            let mut annealer = Annealer::new(diag, &ptx, &annealing_config)?;
            linear_develop_loop(&mut annealer, step, &annealing_config, &progress)?
        },
        false => {
            let mut annealer = QuadraticAnnealer::new(diag, &ptx, &annealing_config)?; 
            quadratic_develop_loop(&mut annealer, step, &annealing_config, &progress)?
        }
    };
    observer_thread.join().unwrap();

    let result = analyze(amplitudes, analyze_config);
    Ok(result)

}

fn linear_develop_loop(annealer: &mut Annealer, step: u64, annealing_config: &AnnealingConfig, progress: &Arc<Mutex<ProgressData>>) -> SimResult<Vec<Complex64>> {
    let mut t: f64;
    let mut ratio: u8 = 0;
    for i in 0..step {
        t = (i as f64) * annealing_config.dt;
        let a = t / annealing_config.tau;
        let b = annealing_config.b0 * (1.0 - a);

        unsafe  {
            annealer.develop_time(&a, &b)?;
            annealer.swap();
            if i % annealing_config.norm_interval == 0 {
                annealer.calc_norm()?;
            }

            let new_ratio = (i * 100 / step) as u8;
            if ratio != new_ratio {
                let mut p = progress.lock().unwrap();
                p.current_step = i;
            }
            ratio = new_ratio;
        }
    }
    {
        let mut p = progress.lock().unwrap();
        p.is_finished = true;
    }
    annealer.stream.synchronize()?;
    
    let amp = annealer.stream.clone_dtoh(&annealer.f0_dev)?;
    Ok(amp)
}

fn quadratic_develop_loop(annealer: &mut QuadraticAnnealer, step: u64, annealing_config: &AnnealingConfig, progress: &Arc<Mutex<ProgressData>>) -> SimResult<Vec<Complex64>> {
        let mut t: f64;
    let mut ratio: u8 = 0;
    unsafe {
        annealer.pre_develop_time(&0.0, &1.0)?;
        annealer.swap();
        annealer.calc_norm()?;
    }
    for i in 1..step {
        t = (i as f64) * annealing_config.dt;
        let a = t / annealing_config.tau;
        let b = annealing_config.b0 * (1.0 - a);

        unsafe  {
            annealer.develop_time(&a, &b)?;
            annealer.swap();
            if i % annealing_config.norm_interval == 0 {
                annealer.calc_norm()?;
            }
        }

        let new_ratio = (i * 100 / step) as u8;
        if ratio != new_ratio {
            let mut p = progress.lock().unwrap();
            p.current_step = i;
        }
        ratio = new_ratio;
    }
    {
        let mut p = progress.lock().unwrap();
        p.is_finished = true;
    }
    annealer.stream.synchronize()?;

    let amp = annealer.stream.clone_dtoh(&annealer.f_current_dev)?;
    Ok(amp)
}

fn is_linear(annealing_config: &AnnealingConfig) -> bool {
    match annealing_config.develop_time_method {
        DevelopTimeMethod::DevelopTime => true,
        DevelopTimeMethod::DevelopTimeWarp => true,
        DevelopTimeMethod::QuadraticDevelopTimeWarp => false,
        DevelopTimeMethod::Default => false,
    }
}