use crate::simulator::{analyze::{AnalysisConfig, AnalysisResult, analysis_result::ProbSet}, complex::Complex64};


/// 時間発展させた結果得られた状態ベクトルを分析し, 確率が大きい順に並び替えたベクトルを含む構造体を返す
pub fn analyze(amplitudes: Vec<Complex64>, config: AnalysisConfig) -> AnalysisResult {
    let mut probabilities = vec![0.0; amplitudes.len()];
    let mut sorted_probabilities: Vec<ProbSet> = vec![];
    for (i, amplitude) in amplitudes.iter().enumerate() {
        let prob = __round(amplitude.abs().powi(2), config.round);
        probabilities[i] = prob;
        sorted_probabilities.push(ProbSet {
            amplitude: amplitude.clone(),
            probability: prob,
            status_number: i as u32,
        });
    }
    sorted_probabilities.sort_by(|a, b| b.probability.total_cmp(&a.probability));

    AnalysisResult {
        amplitudes,
        probabilities,
        sorted_probabilities,
    }
}

fn __round(val: f64, round: u32) -> f64 {
    (val * 10f64.powi(round as i32)).round() / 10f64.powi(round as i32)
}