use crate::ProgressData;


#[derive(Debug, Clone)]
pub struct ObserverState<T> {
    pub progress: ProgressData,
    pub user_data: T
}