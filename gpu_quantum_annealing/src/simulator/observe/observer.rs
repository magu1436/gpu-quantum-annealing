use std::{sync::{Arc, Mutex}, thread::sleep, time::Duration};

use crate::simulator::observe::{ObserverConfig, ProgressData};


#[repr(C)]
pub struct Observer {
    config: ObserverConfig,
    progress_data_arc: Arc<Mutex<ProgressData>>,
}

impl Observer
{
    pub fn new(config: ObserverConfig, progress_data_arc: Arc<Mutex<ProgressData>>) -> Self {
        Self {
            config,
            progress_data_arc,
        }
    }

    pub fn run(&mut self) {
        let mut last_saved_progress: Option<ProgressData> = None;
        
        loop {
            let snapshot ={
                let p = self.progress_data_arc.lock().unwrap();
                p.clone()
            };
            
            if last_saved_progress != Some(snapshot) {
                last_saved_progress = Some(snapshot);
                (self.config.observe_func)(&snapshot);
            }

            if snapshot.is_finished {
                break;
            }
            sleep(Duration::from_secs_f64(self.config.sleep_time_secs));
        }
    }
}