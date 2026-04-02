use cudarc::driver::{CudaContext, PushKernelArg};

use crate::simulator::{compile_ptx::compile_ptx, complex::Complex64, launch_config::create_launch_config};



pub fn execute_scalar_multiple() -> Vec<Complex64>{

    let ptx = compile_ptx("kernels/scalar_multiple.cu");

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let module = ctx.load_module(ptx).unwrap();
    let f = module.load_function("scalar_multiple_complex").unwrap();
    
    let array: Vec<Complex64> = vec![
        Complex64 { re: 1.0f64, im: 1.0f64 }, Complex64 { re: 2.0f64, im: 2.0f64 }, Complex64 { re: 3.0f64, im: 3.0f64 },
        Complex64 { re: 4.0f64, im: 4.0f64 }, Complex64 { re: 5.0f64, im: 5.0f64 }, Complex64 { re: 6.0f64, im: 6.0f64 },
        Complex64 { re: 7.0f64, im: 7.0f64 }, Complex64 { re: 8.0f64, im: 8.0f64 }, Complex64 { re: 9.0f64, im: 9.0f64 },
    ];
    let scalar = 2.0f32;
    const N: usize = 3;

    let array_dev = stream.clone_htod(&array).unwrap();
    let mut result_dev = stream.alloc_zeros::<Complex64>(N * N).unwrap();

    let threads_x = 2u32;

    let cfg = create_launch_config(N, threads_x);

    unsafe  {
        stream
            .launch_builder(&f)
            .arg(&scalar)
            .arg(&array_dev)
            .arg(&mut result_dev)
            .arg(&(N as i32))
            .launch(cfg)
            .unwrap();
    }

    let result = stream.clone_dtoh(&result_dev).unwrap();
    
    result
}