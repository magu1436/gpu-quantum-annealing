use std::sync::Arc;

use cudarc::{driver::{
    CudaContext,
    CudaFunction,
    CudaSlice,
    CudaStream,
    LaunchConfig
}, nvrtc::Ptx};

use crate::{AnnealingConfig, SimResult, simulator::{complex::Complex64, launch_config::{KernelLayout, create_launch_config}}};



pub struct Annealer {
    pub config: AnnealingConfig,

    pub n: u32,
    pub bit_count: u32,
    
    pub f0_host: Vec<Complex64>,
    pub diag_host: Vec<f64>,

    pub f0_dev: CudaSlice<Complex64>,
    pub f1_dev: CudaSlice<Complex64>,
    pub diag_dev: CudaSlice<f64>,
    pub norm_dev: CudaSlice<f64>,

    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,

    pub cfg_for_vec: LaunchConfig,
    pub cfg_for_develop_time: LaunchConfig,
    pub cfg_for_norm: LaunchConfig,

    pub develop_time_func: CudaFunction,
    pub calc_norm_func: CudaFunction,
    pub update_f0_func: CudaFunction,
}

impl Annealer {
    pub fn new(
        diag: &Vec<f64>,
        ptx: &Ptx,
        config: &AnnealingConfig,
    ) -> SimResult<Self> {
        let n = diag.len() as u32;
        let bit_count = (n as f64).log2() as u32;

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(*ptx)?;

        let develop_time_func = match use_warp(&config) {
            true => module.load_function("develop_time").kernel_not_found_err("develop_time")?,
            false => module.load_function("develop_time_warp").kernel_not_found_err("develop_time_warp")?,
        };
        let calc_norm_func = module.load_function("add_to_calc_norm").kernel_not_found_err("add_to_calc_norm")?;
        let update_f0_func = module.load_function("update_f0").kernel_not_found_err("update_f0")?;

        let f0_host = vec![Complex64::new(1.0f64 / (n as f64).sqrt(), 0.0f64); n as usize];
        
        let mut f0_dev = stream.clone_htod(&f0_host)?;
        let mut f1_dev = stream.alloc_zeros::<Complex64>(n as usize)?; 
        let diag_dev = stream.clone_htod(diag)?; 
        let norm_dev = stream.alloc_zeros::<f64>(1)?;

        let cfg_for_vec = create_launch_config(n as usize, config.threads_x, KernelLayout::Vector2D);
        let cfg_for_develop_time = create_launch_config(n as usize, config.threads_x, KernelLayout::Warp);
        let cfg_for_norm = create_launch_config(n as usize, config.threads_x, KernelLayout::Warp);
        cfg_for_norm.shared_mem_bytes = config.threads_x * (std::mem::size_of::<f64>() as u32);

        let annealer = Annealer {
            config: config.clone(),
            n,
            bit_count,
            f0_host,
            diag_host: diag.clone(),
            f0_dev,
            f1_dev,
            diag_dev,
            norm_dev,
            ctx,
            stream,
            cfg_for_vec,
            cfg_for_develop_time,
            cfg_for_norm,
            develop_time_func,
            calc_norm_func,
            update_f0_func,
        };
        Ok(annealer)

    }
}

fn use_warp(config: &AnnealingConfig) -> bool {
    match config.develop_time_method {
        DevelopTimeMethod::DevelopTime => false,
        DevelopTimeMethod::DevelopTimeWarp => true,
        DevelopTimeMethod::Default => config.threads_x < 32
    }
}