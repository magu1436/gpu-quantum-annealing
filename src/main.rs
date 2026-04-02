mod simulator;

fn main() {
    let r = simulator::execute_scalar_multiple::execute_scalar_multiple();
    println!("{:?}", r);

    let expected = vec![
        2.0, 4.0, 6.0,
        8.0, 10.0, 12.0,
        14.0, 16.0, 18.0,
    ];

    assert_eq!(r, expected);
    println!("{r:?}");
}
