//! Single-flight: concurrent identical uncached prefixes share ONE prefill (R1-3).
//!
//! Exactly one caller becomes the leader for a key and runs the prefill; every
//! concurrent follower awaits that same result and is served from it. The entry
//! is removed by the leader alone when it finishes, so a follower's guard drop
//! can never evict an in-flight leader.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{watch, Mutex};

/// Outcome of a shared prefill: the published blob path, or the failure.
pub type PrefillResult = Result<String, String>;

struct InFlightEntry {
    /// Carries the leader's result to all followers. `None` until the leader
    /// publishes; the sender is dropped when the leader's guard drops, which
    /// wakes followers that were still waiting.
    done: watch::Sender<Option<PrefillResult>>,
}

/// Leader token. Only the leader holds one, so only the leader removes the
/// entry on drop; followers hold no guard.
pub struct Leadership {
    key: Vec<u32>,
    map: Arc<Mutex<HashMap<Vec<u32>, InFlightEntry>>>,
    done: watch::Sender<Option<PrefillResult>>,
}

impl Leadership {
    /// Publish the shared result to all current and future followers.
    pub fn complete(&self, result: PrefillResult) {
        // Ignore send errors: no followers is not a failure.
        let _ = self.done.send(Some(result));
    }
}

impl Drop for Leadership {
    fn drop(&mut self) {
        // The leader alone removes the entry. Use a blocking lock via spawn:
        // Drop cannot await, so hand the removal to the runtime; if the runtime
        // is gone the map dies with it. Removal must not be skipped, or later
        // followers subscribe to a finished entry and block.
        let key = std::mem::take(&mut self.key);
        let map = Arc::clone(&self.map);
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                map.lock().await.remove(&key);
            });
        }
    }
}

pub struct SingleFlight {
    map: Arc<Mutex<HashMap<Vec<u32>, InFlightEntry>>>,
}

impl Default for SingleFlight {
    fn default() -> Self {
        Self::new()
    }
}

/// What the caller becomes for a key.
pub enum Role {
    /// This caller runs the prefill; call `complete()` with the result.
    Leader(Leadership),
    /// Another prefill is in flight; await its result.
    Follower(watch::Receiver<Option<PrefillResult>>),
}

impl SingleFlight {
    pub fn new() -> Self {
        Self {
            map: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Become the leader for `key`, or a follower of the in-flight leader.
    pub async fn enter(&self, key: Vec<u32>) -> Role {
        let mut map = self.map.lock().await;
        if let Some(entry) = map.get(&key) {
            return Role::Follower(entry.done.subscribe());
        }
        let (tx, _rx) = watch::channel(None);
        map.insert(key.clone(), InFlightEntry { done: tx.clone() });
        Role::Leader(Leadership {
            key,
            map: Arc::clone(&self.map),
            done: tx,
        })
    }
}

/// Await a follower's result. Returns `None` if the leader vanished without
/// publishing (sender dropped) — the caller should fall back to its own path.
pub async fn await_result(mut rx: watch::Receiver<Option<PrefillResult>>) -> Option<PrefillResult> {
    loop {
        // Already published?
        if let Some(result) = rx.borrow().clone() {
            return Some(result);
        }
        // Wait for a change; Err means the leader's sender was dropped.
        if rx.changed().await.is_err() {
            // One last read: the value may have been set before the drop.
            return rx.borrow().clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn one_leader_many_followers_share_result() {
        let sf = SingleFlight::new();
        let key = vec![1, 2, 3];

        let leader = match sf.enter(key.clone()).await {
            Role::Leader(l) => l,
            Role::Follower(_) => panic!("first caller must lead"),
        };

        let mut followers = Vec::new();
        for _ in 0..4 {
            match sf.enter(key.clone()).await {
                Role::Follower(rx) => followers.push(rx),
                Role::Leader(_) => panic!("only one leader allowed"),
            }
        }

        leader.complete(Ok("blob-abc".into()));
        for rx in followers {
            let got = await_result(rx).await;
            assert_eq!(got, Some(Ok("blob-abc".to_string())));
        }
    }

    #[tokio::test]
    async fn leader_error_propagates_to_followers() {
        let sf = SingleFlight::new();
        let key = vec![9, 9];
        let leader = match sf.enter(key.clone()).await {
            Role::Leader(l) => l,
            Role::Follower(_) => panic!(),
        };
        let follower = match sf.enter(key).await {
            Role::Follower(rx) => rx,
            Role::Leader(_) => panic!(),
        };
        leader.complete(Err("prefill blew up".into()));
        assert_eq!(
            await_result(follower).await,
            Some(Err("prefill blew up".to_string()))
        );
    }

    #[tokio::test]
    async fn new_key_after_leader_drops_can_lead() {
        let sf = SingleFlight::new();
        let key = vec![5];
        {
            let leader = match sf.enter(key.clone()).await {
                Role::Leader(l) => l,
                Role::Follower(_) => panic!(),
            };
            leader.complete(Ok("b".into()));
        }
        // Give the spawned removal a chance to run.
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        match sf.enter(key).await {
            Role::Leader(_) => {}
            Role::Follower(_) => panic!("entry must be removed after leader drop"),
        }
    }

    #[tokio::test]
    async fn different_keys_do_not_block() {
        let sf = SingleFlight::new();
        let a = matches!(sf.enter(vec![1]).await, Role::Leader(_));
        let b = matches!(sf.enter(vec![2]).await, Role::Leader(_));
        assert!(a && b, "distinct keys both lead");
    }

    #[tokio::test]
    async fn follower_hanging_when_leader_drops_without_complete() {
        let sf = SingleFlight::new();
        let key = vec![7];
        let follower = {
            let leader = match sf.enter(key.clone()).await {
                Role::Leader(l) => l,
                Role::Follower(_) => panic!(),
            };
            let rx = match sf.enter(key).await {
                Role::Follower(rx) => rx,
                Role::Leader(_) => panic!(),
            };
            drop(leader); // leader vanishes without completing
            rx
        };
        // Must NOT hang: returns None so the caller can fall back.
        assert_eq!(await_result(follower).await, None);
    }
}
