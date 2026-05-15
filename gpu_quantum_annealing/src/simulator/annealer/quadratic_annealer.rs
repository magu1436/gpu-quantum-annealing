use std::sync::Arc;

use cudarc::{
    driver::{
        CudaContext,
        CudaFunction,
        CudaSlice,
        CudaStream,
        LaunchConfig,
        PushKernelArg
    },
    nvrtc::Ptx
};

use crate::{
    AnnealingConfig,
    config::DevelopTimeMethod,
    simulator::{
        annealer::{
            AnnealingResult,
            ResultExt,
            AnnealingError,
            KernelLayout,
            create_launch_config
        },
        complex::Complex64,
    },
};


/// GPU上で量子アニーリングシミュレーションを実行するワーカークラス
pub struct QuadraticAnnealer {
    /// アニーリングシミュレーションの設定
    pub config: AnnealingConfig,

    /// 状態ベクトルの大きさ
    pub n: u32,
    /// 量子ビット数
    pub bit_count: u32,

    /// GPU上の状態ベクトル
    pub f_current_dev: CudaSlice<Complex64>,
    /// GPU上の過去の状態ベクトル
    pub f_prev_dev: CudaSlice<Complex64>,
    /// GPU上の状態ベクトルの仮置場
    pub f_developed_dev: CudaSlice<Complex64>,
    /// GPU上の対角行列
    pub diag_dev: CudaSlice<f64>,
    /// GPU上の状態ベクトルのノルム
    pub norm_dev: CudaSlice<f64>,

    /// GPU上のストリーム
    pub stream: Arc<CudaStream>,

    /// ベクトルを扱うカーネルが使用するコンフィグ
    pub cfg_for_vec: LaunchConfig,
    /// 時間発展カーネルが使用するコンフィグ
    pub cfg_for_develop_time: LaunchConfig,
    /// ノルムを計算するカーネルが使用するコンフィグ
    pub cfg_for_norm: LaunchConfig,

    /// 時間発展カーネル
    pub develop_time_func: CudaFunction,
    pub quadratic_develop_time_func: CudaFunction,
    /// ノルムを計算するカーネル
    pub calc_norm_func: CudaFunction,
    /// 事前に計算された状態ベクトルを更新するカーネル
    pub update_f0_func: CudaFunction,
}

impl QuadraticAnnealer {
    pub fn new(
        diag: &Vec<f64>,
        ptx: &Ptx,
        config: &AnnealingConfig,
    ) -> AnnealingResult<Self> {
        let n = diag.len() as u32;
        let bit_count = (n as f64).log2() as u32;

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(ptx.clone())?;

        let develop_time_func = match use_warp(&config) {
            true => module.load_function("develop_time").kernel_not_found_err("develop_time")?,
            false => module.load_function("develop_time_warp").kernel_not_found_err("develop_time_warp")?,
        };
        let quadratic_develop_time_func = module.load_function("quadratic_develop_time_warp").kernel_not_found_err("quadratic_develop_time")?;
        let calc_norm_func = module.load_function("add_to_calc_norm").kernel_not_found_err("add_to_calc_norm")?;
        let update_f0_func = module.load_function("update_f0").kernel_not_found_err("update_f0")?;

        let f0_host = vec![Complex64::new(1.0f64 / (n as f64).sqrt(), 0.0f64); n as usize];
        
        let f_current_dev = stream.clone_htod(&f0_host)?;
        let f_prev_dev = stream.alloc_zeros::<Complex64>(n as usize)?;
        let f_developed_dev = stream.alloc_zeros::<Complex64>(n as usize)?; 
        let diag_dev = stream.clone_htod(diag)?; 
        let norm_dev = stream.alloc_zeros::<f64>(1)?;

        let cfg_for_vec = create_launch_config(n as usize, config.threads_x, KernelLayout::Vector2D)?;
        let cfg_for_develop_time = create_launch_config(n as usize, config.threads_x, KernelLayout::Warp)?;
        let mut cfg_for_norm = create_launch_config(n as usize, config.threads_x, KernelLayout::Warp)?;
        cfg_for_norm.shared_mem_bytes = config.threads_x * (std::mem::size_of::<f64>() as u32);

        let annealer = QuadraticAnnealer {
            config: config.clone(),
            n,
            bit_count,
            f_current_dev,
            f_prev_dev,
            f_developed_dev,
            diag_dev,
            norm_dev,
            stream,
            cfg_for_vec,
            cfg_for_develop_time,
            cfg_for_norm,
            develop_time_func,
            quadratic_develop_time_func,
            calc_norm_func,
            update_f0_func,
        };
        Ok(annealer)

    }

    pub unsafe fn pre_develop_time(&self, &a: &f64, &b: &f64) -> AnnealingResult<()>{
        unsafe {
            self.stream
                .launch_builder(&self.develop_time_func)
                .arg(&a)
                .arg(&b)
                .arg(&self.config.dt)
                .arg(&self.diag_dev)
                .arg(&self.f_current_dev)
                .arg(&self.n)
                .arg(&self.bit_count)
                .arg(&self.f_developed_dev)
                .launch(self.cfg_for_develop_time)
                .kernel_process_err("develop time kernel")?;
        }
        Ok(())
    }

    /// 時間発展を実行する
    pub unsafe fn develop_time(&self, &a: &f64, &b: &f64) -> AnnealingResult<()>{
        unsafe {
            self.stream
                .launch_builder(&self.quadratic_develop_time_func)
                .arg(&a)
                .arg(&b)
                .arg(&self.config.dt)
                .arg(&self.diag_dev)
                .arg(&self.f_current_dev)
                .arg(&self.f_prev_dev)
                .arg(&self.n)
                .arg(&self.bit_count)
                .arg(&self.f_developed_dev)
                .launch(self.cfg_for_develop_time)
                .kernel_process_err("quadratic develop time kernel")?;
        }
        Ok(())
    }

    /// 状態ベクトルを入れ替える
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.f_current_dev, &mut self.f_prev_dev);
        std::mem::swap(&mut self.f_current_dev, &mut self.f_developed_dev);
    }

    /// ノルムを計算する
    pub unsafe fn calc_norm(&self) -> AnnealingResult<()> {
        unsafe {
            self.stream
                .launch_builder(&self.calc_norm_func)
                .arg(&self.f_current_dev)
                .arg(&self.norm_dev)
                .arg(&self.n)
                .launch(self.cfg_for_norm)
                .kernel_process_err("calc norm kernel")?;

            self.stream
                .launch_builder(&self.update_f0_func)
                .arg(&self.f_current_dev)
                .arg(&self.norm_dev)
                .arg(&self.f_developed_dev)
                .arg(&self.n)
                .launch(self.cfg_for_vec)
                .kernel_process_err("update f0 kernel")?;
        }
        Ok(())
    }
}

fn use_warp(config: &AnnealingConfig) -> bool {
    match config.develop_time_method {
        DevelopTimeMethod::DevelopTime => false,
        DevelopTimeMethod::DevelopTimeWarp => true,
        DevelopTimeMethod::QuadraticDevelopTimeWarp => true,
        DevelopTimeMethod::Default => config.threads_x < 32
    }
}