
/// 進捗状況を格納する構造体.
/// observe_func の引数として使用される.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ProgressData{
    pub total_step: u64,
    pub current_step: u64,
    pub is_finished: bool,
}

impl ProgressData {
    pub fn new(total_step: u64, current_step: u64, is_finished: bool) -> Self {
        Self { total_step, current_step, is_finished }
    }
}