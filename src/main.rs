mod simulator;

fn main() {

    const NUMS: [i32; 5] = [1, 2, 3, 4, 6];
    const BIT_COUNT: usize = NUMS.len();

    let bit = |decimal: u32, idx: usize| -> i32 {
        let shift = NUMS.len() as u32 - 1 - (idx as u32);
        ((decimal >> shift) & 1) as i32
    };

    let objective_function = | idx: usize | -> f64 {
        let mut result = 0.0;
        for i in 0..NUMS.len() {
            for j in i+1..NUMS.len() {
                let a = 2 * bit(idx as u32, i) - 1;
                let b = 2 * bit(idx as u32, j) - 1;
                result += (a * b * ((NUMS[i] * NUMS[j]))) as f64;
            }
        }
        result
    };

    let r = simulator::execute::excute(BIT_COUNT, objective_function);
    println!("{:?}", r);
}
