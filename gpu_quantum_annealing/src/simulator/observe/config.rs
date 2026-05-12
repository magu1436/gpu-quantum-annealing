use crate::{ProgressData};

pub struct ObserverConfig {
    pub sleep_time_secs: f64,
    pub observe_func: Box<dyn FnMut(&ProgressData) + Send + 'static>,
}

impl ObserverConfig {
    pub fn new<F>(sleep_time_secs: f64, observe_func: F) -> Self
    where F: FnMut(&ProgressData) + Send + 'static {
        Self { sleep_time_secs, observe_func: Box::new(observe_func) }
    }
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            sleep_time_secs: 1.0,
            observe_func: Box::new(|p: &ProgressData| {
                println!("progress: {}%", (p.current_step as f64) / (p.total_step as f64) * 100.0);
            }),
        }
    }
}