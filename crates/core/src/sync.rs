//! Locking that survives a panic in another thread.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// `lock()` without the poisoning: when a thread panics while holding a mutex, the standard library makes every
/// later `lock()` fail, so one panic would take the whole application down. The data is still there, usually
/// consistent enough to carry on (a mask or a setting), so the guard is handed out anyway.
pub trait LockExt<T> {
    fn guard(&self) -> MutexGuard<'_, T>;
}

impl<T> LockExt<T> for Mutex<T> {
    fn guard(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_mutex_poisoned_by_a_panic_can_still_be_locked() {
        let m = Arc::new(Mutex::new(5));
        let m2 = m.clone();
        let _ = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("a thread dies while holding the lock");
        })
        .join();
        assert!(m.lock().is_err(), "the standard lock is poisoned");
        assert_eq!(*m.guard(), 5);
        *m.guard() += 1;
        assert_eq!(*m.guard(), 6);
    }
}
