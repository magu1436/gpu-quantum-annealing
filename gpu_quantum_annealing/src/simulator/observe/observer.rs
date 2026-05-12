use std::{sync::{Arc, Mutex}, thread::sleep, time::Duration};

use crate::simulator::observe::{ObserverConfig, ObserverState, ProgressData};


#[repr(C)]
pub struct Observer<T> {
    config: ObserverConfig<T>,
    progress_state_arc: Arc<Mutex<ObserverState<T>>>,
}

impl<T> Observer<T> 
where 
    T: Clone + Send + 'static,
{
    pub fn new(config: ObserverConfig<T>, progress_state_arc: Arc<Mutex<ObserverState<T>>>) -> Self {
        Self {
            config,
            progress_state_arc,
        }
    }

    pub fn run(&self) {
        let mut last_saved_progress: Option<ProgressData> = None;
        
        loop {
            let is_finished ={
                let mut snapshot = self.progress_state_arc.lock().unwrap();

                if last_saved_progress != Some(snapshot.progress) {
                    (self.config.observe_func)(&mut snapshot);
                    last_saved_progress = Some(snapshot.progress);
                };

                snapshot.progress.is_finished
            };

            if is_finished {
                break;
            }

            sleep(Duration::from_secs_f64(self.config.sleep_time_secs));
        }

        {
            let mut snapshot = self.progress_state_arc.lock().unwrap();
            (self.config.observe_func)(&mut snapshot);
        }
    }
}