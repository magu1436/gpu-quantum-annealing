use std::time::{self, Instant};

use gpu_quantum_annealing::{
    AnnealingConfig, ObserverConfig, ObserverState, execute::execute
};

use crate::sample::{create_integer_pertition_hll};
mod sample;

fn main() {

    let nums = vec![0; 5];
    let diag = create_integer_pertition_hll(nums);

    let cfg = AnnealingConfig::default();

    let r = execute(
        &diag,
        cfg,
        gpu_quantum_annealing::AnalysisConfig::default(),
        ObserverConfig::new(1.0, time::Instant::now(), observe_func),
    );
    match r {
        Ok(prob) => {
            println!("{:#?}, \n{:#?}, \n{:#?}", prob.sorted_probabilities[0], prob.sorted_probabilities[1], prob.sorted_probabilities[2]);
            // write_f64_bin("app/results/probabilities.bin", &prob.probabilities).unwrap();
            println!("{:#?}", prob.probabilities.len());
        },
        Err(e) => panic!("{}", e),
    };
}

fn observe_func(p: &mut ObserverState<Instant>) {
    let progress = (p.progress.current_step as f64) / (p.progress.total_step as f64) * 100.0;
    println!("progress: {}%", progress);
    println!("current_step: {}", p.progress.current_step);
    if p.progress.current_step == 0 {
        p.user_data = time::Instant::now();
    }

    if p.progress.is_finished {
        println!("elapsed: {}", p.user_data.elapsed().as_secs_f64());
    }
}