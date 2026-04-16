use gpu_quantum_annealing::{
    AnnealingConfig,
    execute::execute,
};

fn main() {

    let nums = [1, 2, 3, 4];
    let bit_count: usize = nums.len();

    let bit = |decimal: u32, idx: usize| -> i32 {
        let shift = nums.len() as u32 - 1 - (idx as u32);
        ((decimal >> shift) & 1) as i32
    };

    let objective_function = | idx: usize | -> f64 {
        let mut result = 0.0;
        for i in 0..nums.len() {
            for j in i+1..nums.len() {
                let a = 2 * bit(idx as u32, i) - 1;
                let b = 2 * bit(idx as u32, j) - 1;
                result += (a * b * ((nums[i] * nums[j]))) as f64;
            }
        }
        result
    };

    let mut diag = vec![0.0f64; 2u64.pow(bit_count as u32) as usize];
    for i in 0..2u64.pow(bit_count as u32) as usize {
        diag[i] = objective_function(i);
    }

    let cfg = AnnealingConfig {
        threads_x: 128,
        ..Default::default()
    };

    println!("{:?}", diag);

    let r = execute(&diag, cfg);
    match r {
        Ok(prob) => println!("{:?}", prob),
        Err(e) => panic!("{}", e),
    };
}
