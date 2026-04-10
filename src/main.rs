mod simulator;

fn main() {

    let nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
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

    let cfg = simulator::config::AnnealingConfig {
        tau: 100.0,
        threads_x: 128,
        ..Default::default()
    };

    let r = simulator::execute::excute(bit_count, objective_function, cfg);
    println!("{:?}", r);
}
