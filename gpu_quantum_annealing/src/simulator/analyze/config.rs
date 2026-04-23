/// 分析の設定を格納する構造体
#[derive(Debug)]
pub struct AnalysisConfig {
    /// 確率を丸める桁数
    pub round: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self { round: 5 }
    }
}