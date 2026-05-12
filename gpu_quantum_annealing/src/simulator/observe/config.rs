use crate::simulator::observe::{ObserverState};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ObserverConfig<T> {
    pub sleep_time_secs: f64,
    pub user_data: T,
    pub observe_func: fn(&mut ObserverState<T>),
}

impl<T> ObserverConfig<T> {
    pub fn new(sleep_time_secs: f64, user_data: T, observe_func: fn(&mut ObserverState<T>)) -> Self {
        Self { sleep_time_secs, user_data, observe_func }
    }
}

impl Default for ObserverConfig<()> {
    fn default() -> Self {
        Self {
            sleep_time_secs: 1.0,
            user_data: (),
            observe_func: |_: &mut ObserverState<()>| {},
        }
    }
}