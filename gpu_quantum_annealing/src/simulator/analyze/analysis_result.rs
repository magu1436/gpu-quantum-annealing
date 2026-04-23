use crate::simulator::complex::Complex64;

/// 状態ベクトルの分析結果を格納する構造体
#[derive(Debug)]
pub struct AnalysisResult {
    /// 生の状態ベクトル
    pub amplitudes: Vec<Complex64>,
    /// 状態ベクトルの各要素の大きさをとった確率分布
    pub probabilities: Vec<f64>,
    /// 確率を降順に並び替えたベクトル
    pub sorted_probabilities: Vec<ProbSet>,
}

/// 確率と状態番号を格納した構造体
#[derive(Debug)]
pub struct ProbSet {
    /// 状態(振幅)
    pub amplitude: Complex64,
    /// 確率
    pub probability: f64,
    /// 状態番号
    pub status_number: u32,
}