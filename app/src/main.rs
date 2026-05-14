use gpu_quantum_annealing::{
    AnnealingConfig, ObserverConfig, ProgressData, execute::execute
};

use crate::sample::{create_integer_pertition_hll};
mod sample;

fn main() {

    let nums = vec![1, 2, 3];
    let diag = create_integer_pertition_hll(nums);

    let cfg = AnnealingConfig {
        tau: 2.0,
        dt: 2e-6,
        ..Default::default()
    };

    let start_time = std::time::Instant::now();
    let observe_func = move |p: &ProgressData| {
        println!("progress: {}%", (p.current_step as f64) / (p.total_step as f64) * 100.0);

        if p.is_finished {
            let elapsed = start_time.elapsed().as_secs_f64();
            println!("elapsed: {}s", elapsed);
        }
    };

    let r = execute(
        &diag,
        cfg,
        gpu_quantum_annealing::AnalysisConfig::default(),
        ObserverConfig::new(1.0, observe_func),
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