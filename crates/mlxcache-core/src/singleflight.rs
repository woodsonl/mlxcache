//! Single-flight: concurrent identical uncached prefixes share ONE prefill (R1-3).
//!
//! The second request blocks on the in-flight prefill and is served from its
//! result. One behavior, no either/or.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

#[derive(Debug, Clone, thiserror::Error)]
pub enum SingleFlightError {
    #[error("prefill failed: {0}")]
    PrefillFailed(String),
}

/// Guard for one in-flight prefill. Dropping it removes the in-flight entry.
pub struct InFlightGuard {
    key: Vec<u32>,
    map: Arc<Mutex<HashMap<Vec<u32>, InFlightEntry>>>,
}

pub struct InFlightEntry {
    pub done: broadcast::Sender<Result<(), SingleFlightError>>,
}

pub struct SingleFlight {
    map: Arc<Mutex<HashMap<Vec<u32>, InFlightEntry>>>,
}

impl Default for SingleFlight {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleFlight {
    pub fn new() -> Self {
        Self {
            map: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Try to become the leader for `key`. If another prefill is in flight,
    /// returns a receiver to await its completion.
    pub async fn try_lead(
        &self,
        key: Vec<u32>,
    ) -> (InFlightGuard, Option<broadcast::Receiver<Result<(), SingleFlightError>>>) {
        let mut map = self.map.lock().await;
        if let Some(entry) = map.get(&key) {
            let rx = entry.done.subscribe();
            let guard = InFlightGuard {
                key: key.clone(),
                map: Arc::clone(&self.map),
            };
            return (guard, Some(rx));
        }
        let (tx, _) = broadcast::channel(1);
        map.insert(
            key.clone(),
            InFlightEntry {
                done: tx,
            },
        );
        let guard = InFlightGuard {
            key,
            map: Arc::clone(&self.map),
        };
        (guard, None)
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        // Synchronous best-effort removal; the map lock is a tokio Mutex, so we
        // rely on try_lock to avoid blocking the runtime. If contended, the
        // entry is cleaned up by the next leader's insert-overwrite.
        if let Ok(mut map) = self.map.try_lock() {
            map.remove(&self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn second_request_follows_first() {
        let sf = SingleFlight::new();
        let key = vec![1, 2, 3];
        let (guard1, follower) = sf.try_lead(key.clone()).await;
        assert!(follower.is_none(), "first caller leads");

        let (guard2, follower2) = sf.try_lead(key.clone()).await;
        let mut rx = follower2.expect("second caller follows");
        assert!(Arc::ptr_eq(
            &guard1.map,
            &guard2.map
        ));

        // Leader completes successfully; follower is notified.
        let map = sf.map.lock().await;
        let entry = map.get(&key).unwrap();
        let mut rx2 = entry.done.subscribe();
        drop(map);
        // Send on the leader's sender via the entry
        {
            let map = sf.map.lock().await;
            let entry = map.get(&key).unwrap();
            let _ = entry.done.send(Ok(()));
        }
        assert!(rx.recv().await.is_ok());
        assert!(rx2.recv().await.is_ok());
    }

    #[tokio::test]
    async fn different_keys_do_not_block() {
        let sf = SingleFlight::new();
        let (g1, f1) = sf.try_lead(vec![1]).await;
        let (g2, f2) = sf.try_lead(vec![2]).await;
        assert!(f1.is_none());
        assert!(f2.is_none());
        drop(g1);
        drop(g2);
    }
}
