use crate::config::DevelopTimeMethod;

/// 量子アニーリングシミュレーションの設定を表す。
/// 
/// Exmple:
/// ```rust
/// use simulator::config::AnnealingConfig;
/// let config = AnnealingConfig {
///     dt: 1e-3,
///     ..Default::default()
/// };
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AnnealingConfig {
    /// 時間変化量
    pub dt: f64,
    /// 終端時間
    pub tau: f64,
    /// 初期横磁場
    pub b0: f64,
    /// スレッド数
    pub threads_x: u32,
    /// 何回に一度正規化を行うか
    pub norm_interval: u64,
    /// 時間発展関数の指定
    pub develop_time_method: DevelopTimeMethod,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            dt: 1e-3,
            tau: 20.0,
            b0: 10.0,
            threads_x: 128,
            norm_interval: 10,
            develop_time_method: DevelopTimeMethod::Default,
        }
    }
}