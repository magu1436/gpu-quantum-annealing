use crate::simulator::complex::Complex64;

mod simulator;

fn main() {
    let r = simulator::execute_scalar_multiple::execute_scalar_multiple();
    println!("{:?}", r);

    let expected = vec![
        Complex64 { re: 2.0f64, im: 2.0f64 }, Complex64 { re: 4.0f64, im: 4.0f64 }, Complex64 { re: 6.0f64, im: 6.0f64 },
        Complex64 { re: 8.0f64, im: 8.0f64 }, Complex64 { re: 10.0f64, im: 10.0f64 }, Complex64 { re: 12.0f64, im: 12.0f64 },
        Complex64 { re: 14.0f64, im: 14.0f64 }, Complex64 { re: 16.0f64, im: 16.0f64 }, Complex64 { re: 18.0f64, im: 18.0f64 },
    ];

    assert_eq!(r, expected);
    println!("{r:?}");
}
