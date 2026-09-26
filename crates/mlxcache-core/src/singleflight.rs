//! Single-flight: concurrent identical uncached prefixes share ONE prefill (R1-3).
//!
//! Exactly one caller becomes the leader for a key and runs the prefill; every
//! concurrent follower awaits that same result and is served from it. The entry
//! is removed by the leader alone when it finishes, so a follower's guard drop
//! can never evict an in-flight leader.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{watch, Mutex};

/// Outcome of a shared prefill: the published blob path, or the failure.
pub type PrefillResult = Result<String, String>;

struct InFlightEntry {
    /// Carries the leader's result to all followers. `None` until the leader
    /// publishes; the sender is dropped when the leader's guard drops, which
    /// wakes followers that were still waiting.
    done: watch::Sender<Option<PrefillResult>>,
    /// Followers currently attached to this in-flight prefill. Used to observe
    /// coalescing (and by tests to know followers have registered before the
    /// leader completes). Decremented when a follower guard drops.
    waiters: Arc<AtomicUsize>,
}

/// Leader token. Only the leader holds one, so only the leader removes the
/// entry on drop; followers hold no guard.
pub struct Leadership {
    key: Vec<u32>,
    map: Arc<Mutex<HashMap<Vec<u32>, InFlightEntry>>>,
    done: watch::Sender<Option<PrefillResult>>,
    waiters: Arc<AtomicUsize>,
}

impl Leadership {
    /// Publish the shared result to all current and future followers.
    pub fn complete(&self, result: PrefillResult) {
        // Ignore send errors: no followers is not a failure.
        let _ = self.done.send(Some(result));
    }

    /// Followers currently attached to this prefill.
    pub fn waiter_count(&self) -> usize {
        self.waiters.load(Ordering::SeqCst)
    }
}

impl Drop for Leadership {
    fn drop(&mut self) {
        // The leader alone removes the entry. Removal must not be skipped, or a
        // later follower subscribes to a finished entry. Drop cannot await, so
        // hand the removal to the runtime; if there is no runtime (e.g. a test
        // thread), fall back to a try_lock so the entry does not leak.
        let key = std::mem::take(&mut self.key);
        let map = Arc::clone(&self.map);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    map.lock().await.remove(&key);
                });
            }
            Err(_) => {
                // ponytail: no runtime to spawn on; best-effort try_lock only.
                // A contended lock here leaks one entry until process exit,
                // which only affects non-runtime drop paths (tests, teardown).
                if let Ok(mut m) = map.try_lock() {
                    m.remove(&key);
                }
            }
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

/// A follower's subscription to an in-flight prefill. Dropping it decrements
/// the leader's waiter count (so coalescing can be observed).
pub struct Follower {
    rx: watch::Receiver<Option<PrefillResult>>,
    waiters: Arc<AtomicUsize>,
}

impl Follower {
    /// The receiver to await the leader's result on.
    pub fn receiver(&mut self) -> &mut watch::Receiver<Option<PrefillResult>> {
        &mut self.rx
    }
}

impl Drop for Follower {
    fn drop(&mut self) {
        self.waiters.fetch_sub(1, Ordering::SeqCst);
    }
}

/// What the caller becomes for a key.
pub enum Role {
    /// This caller runs the prefill; call `complete()` with the result.
    Leader(Leadership),
    /// Another prefill is in flight; await its result.
    Follower(Follower),
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
            // Subscribe BEFORE counting: a leader that observes waiter_count
            // must know the receiver already exists, or it could publish before
            // we subscribe and the send would find no receiver.
            let rx = entry.done.subscribe();
            entry.waiters.fetch_add(1, Ordering::SeqCst);
            return Role::Follower(Follower {
                rx,
                waiters: Arc::clone(&entry.waiters),
            });
        }
        let (tx, _rx) = watch::channel(None);
        let waiters = Arc::new(AtomicUsize::new(0));
        map.insert(
            key.clone(),
            InFlightEntry {
                done: tx.clone(),
                waiters: Arc::clone(&waiters),
            },
        );
        Role::Leader(Leadership {
            key,
            map: Arc::clone(&self.map),
            done: tx,
            waiters,
        })
    }

    /// Followers currently attached to an in-flight prefill for `key`, or 0 if
    /// no prefill is in flight. Test/observability hook for coalescing.
    pub async fn waiter_count(&self, key: &[u32]) -> usize {
        let map = self.map.lock().await;
        map.get(key)
            .map(|e| e.waiters.load(Ordering::SeqCst))
            .unwrap_or(0)
    }
}

/// Await a follower's result. Returns `None` if the leader vanished without
/// publishing (sender dropped) — the caller should fall back to its own path.
pub async fn await_result(follower: &mut Follower) -> Option<PrefillResult> {
    let rx = follower.receiver();
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
                Role::Follower(f) => followers.push(f),
                Role::Leader(_) => panic!("only one leader allowed"),
            }
        }
        assert_eq!(leader.waiter_count(), 4, "followers must be counted");

        leader.complete(Ok("blob-abc".into()));
        for mut f in followers.drain(..) {
            let got = await_result(&mut f).await;
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
        let mut follower = match sf.enter(key).await {
            Role::Follower(f) => f,
            Role::Leader(_) => panic!(),
        };
        leader.complete(Err("prefill blew up".into()));
        assert_eq!(
            await_result(&mut follower).await,
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
        let mut follower = {
            let leader = match sf.enter(key.clone()).await {
                Role::Leader(l) => l,
                Role::Follower(_) => panic!(),
            };
            let follower = match sf.enter(key).await {
                Role::Follower(f) => f,
                Role::Leader(_) => panic!(),
            };
            drop(leader); // leader vanishes without completing
            follower
        };
        // Must NOT hang: returns None so the caller can fall back.
        assert_eq!(await_result(&mut follower).await, None);
    }
}
