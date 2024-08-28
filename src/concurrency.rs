use std::{sync::{mpsc, Arc, Mutex}, thread};

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<mpsc::Sender<Job>>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    pub fn new(size: usize) -> ThreadPool {
        assert!(size > 0);

        let (sender, receiver): (mpsc::Sender<Box<dyn FnOnce() + Send>>, mpsc::Receiver<Box<dyn FnOnce() + Send>>) = mpsc::channel();
        let receiver: Arc<Mutex<mpsc::Receiver<Box<dyn FnOnce() + Send>>>> = Arc::new(Mutex::new(receiver));

        let mut workers: Vec<Worker> = Vec::with_capacity(size);
        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }
        ThreadPool { workers, sender: Some(sender) }
    }
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job: Box<F> = Box::new(f);
        self.sender.as_ref().unwrap().send(job).unwrap();
    }
}

struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Job>>>) -> Worker {
        let thread: thread::JoinHandle<()> = thread::spawn(move || loop {
            let message: Result<Box<dyn FnOnce() + Send>, mpsc::RecvError> = receiver.lock().unwrap().recv();

            match message {
                Ok(job) => {
                    println!("Worker {id} got a job; executing.");

                    job();
                }
                Err(_) => {
                    println!("Worker {id} disconnected; shutting down.");
                    break;
                }
            }
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.sender.take());

        for worker in &mut self.workers {
            println!("Shutting down worker {}", worker.id);

            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_creation() {
        let pool: ThreadPool = ThreadPool::new(4);
        assert_eq!(pool.workers.len(), 4);
    }

    #[test]
    #[should_panic]
    fn test_invalid_thread_pool_creation() {
        ThreadPool::new(0);
    }

    #[test]
    fn test_job_execution() {
        let pool: ThreadPool = ThreadPool::new(2);
        let shared_var: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

        let shared_var_clone: Arc<Mutex<bool>> = Arc::clone(&shared_var);
        pool.execute(move || {
            *shared_var_clone.lock().unwrap() = true;
        });

        thread::sleep(std::time::Duration::from_millis(500));

        assert!(*shared_var.lock().unwrap());
    }

    #[test]
    fn test_multiple_job_execution() {
        let pool: ThreadPool = ThreadPool::new(3);
        let messages: Vec<&str> = vec!["Job 1", "Job 2", "Job 3"];
        for message in messages {
            pool.execute(move || println!("{}", message));
        }
    }

    #[test]
    fn test_worker_1_thread() {
        let pool: ThreadPool = ThreadPool::new(1);
        let shared_var: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

        let shared_var_clone: Arc<Mutex<bool>> = Arc::clone(&shared_var);
        pool.execute(move || {
            *shared_var_clone.lock().unwrap() = true;
        });

        thread::sleep(std::time::Duration::from_millis(1));

        assert!(*shared_var.lock().unwrap());
    }

    #[test]
    fn test_thread_pool_shutdown() {
        let pool: ThreadPool = ThreadPool::new(2);
        drop(pool);
    }
}
