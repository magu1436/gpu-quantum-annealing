use cudarc::driver::{CudaContext, PushKernelArg};

use crate::simulator::{compile_ptx::compile_ptx, launch_config::create_launch_config};



pub fn execute_scalar_multiple() -> Vec<f32>{

    let ptx = compile_ptx("kernels/scalar_multiple.cu");

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let module = ctx.load_module(ptx).unwrap();
    let f = module.load_function("scalar_multiple").unwrap();
    
    let array: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let scalar = 2.0f32;
    const N: usize = 3;

    let array_dev = stream.clone_htod(&array).unwrap();
    let mut result_dev = stream.alloc_zeros::<f32>(N * N).unwrap();

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
    
    return result;
}