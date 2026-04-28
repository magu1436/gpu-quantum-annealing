use gpu_quantum_annealing::{
    AnnealingConfig,
    execute::execute,
};

use crate::sample::create_prisoners_dillemma_hll;
mod sample;

fn main() {

    let num_players = 3;
    let num_pen = 6;
    let num_slack = 3;
    let start_slack = 3;
    let hyper_params = vec![-5, -5, -5];
    let diag = create_prisoners_dillemma_hll(num_players, num_pen, num_slack, start_slack, hyper_params);

    let cfg = AnnealingConfig {
        threads_x: 128,
        tau: 2.0,
        dt: 2.0,
        develop_time_method: gpu_quantum_annealing::DevelopTimeMethod::DevelopTime,
        b0: 1.0,
        ..Default::default()
    };

    // println!("{:?}", diag);

    let r = execute(
        &diag,
        cfg,
        gpu_quantum_annealing::AnalysisConfig::default()
    );
    match r {
        Ok(prob) => println!("{:#?}, \n{:#?}, \n{:#?}", prob.sorted_probabilities[0], prob.sorted_probabilities[1], prob.sorted_probabilities[2]),
        Err(e) => panic!("{}", e),
    };
}