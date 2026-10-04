//! Allocation gate for the SSE streaming hot path (P2b/T10).
//!
//! The daemon's SSE framer runs once per decoded token for every live stream.
//! The design contract: ONE allocation per token frame (the outgoing chunk
//! buffer; the reqwest chunk may allocate independently). Before the
//! splice-framer, every token built a full `serde_json::Value` tree and
//! re-serialized it — several allocations per token, multiplied by every
//! concurrent stream and every token. The OpenAI chunk envelope made this
//! sharper: chunks are LARGER than the raw sidecar line (id/object/created/
//! model head + choices wrapper), so the framer precomputes the head once per
//! REQUEST and splices the sidecar's own already-escaped `"text"` string and
//! `"token"` digits through by byte scan — parse-free, escape-free.
//!
//! This test runs in its OWN integration-test binary because a global
//! counting allocator is process-global: sharing a binary with the other
//! suites would count their parallel allocations and flake. It MUST also run
//! with --test-threads=1 (pinned in CI): even libtest's own harness threads
//! allocate inside the measurement window.
//!
//! The framer (per-request head) is constructed OUTSIDE the measured window,
//! matching the real caller: one ChunkFramer per stream, reused per token.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// Serializes the measurement sections: the counting allocator is process
/// global, so parallel tests would count each other's allocations.
static MEASURE: Mutex<()> = Mutex::new(());

fn measure() -> MutexGuard<'static, ()> {
    MEASURE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Warm the framer on first use: the very first serde_json parse in a process
/// lazily initializes parser state that costs one allocation, unrelated to the
/// steady-state per-frame cost this gate measures.
fn warm() {
    let framer = framer_for_gate();
    let mut frames = Vec::with_capacity(2);
    let before = ALLOCS.load(Ordering::Relaxed);
    push_line_for_gate(&framer, &mut frames, br#"{"done": true}"#);
    push_line_for_gate(&framer, &mut frames, br#"{"token": 1, "text": "w"}"#);
    let _ = ALLOCS.load(Ordering::Relaxed) - before;
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

use mlxcache_daemon::http::test_support::{framer_for_gate, push_line_for_gate};

fn frames_for(lines: &[&[u8]]) -> usize {
    // The CALLER holds the measurement lock (see `measure`): this helper is
    // only ever invoked from a test that already serialized itself.
    // The framer is built BEFORE the counter snapshot: per-request head
    // construction is not a per-frame cost.
    let framer = framer_for_gate();
    let before = ALLOCS.load(Ordering::Relaxed);
    // Capacity 1+: the daemon's real caller (the unfold loop) builds its
    // frames Vec once per chunk and reuses it across tokens, so the Vec
    // growth is NOT a per-frame cost. Measuring with an empty Vec would
    // charge the (first-push) growth to the frame accounting.
    let mut frames = Vec::with_capacity(lines.len() + 1);
    for line in lines {
        push_line_for_gate(&framer, &mut frames, line);
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    after.saturating_sub(before)
}

#[test]
fn token_frame_allocates_at_most_two_per_frame() {
    let _g = measure();
    warm();
    let line = br#"{"token": 12345, "text": "hello world"}"#;
    let allocs = frames_for(&[line, line, line, line]);
    assert!(
        allocs <= 3 * 2 + 2, // 3 token frames (≤2 each) + one Vec growth for the frames vec
        "SSE token framing allocated {allocs} times for 3 frames (budget 8): the hot path \
         regressed to per-frame Value trees or string churn"
    );
}

#[test]
fn done_frame_costs_at_most_one_allocation() {
    // The done line emits TWO frames: the final OpenAI chunk (empty delta +
    // finish_reason "stop" — one buffer) and the static [DONE] terminator
    // (zero allocations).
    let _g = measure();
    warm();
    let framer = framer_for_gate();
    let mut frames = Vec::with_capacity(2);
    let before = ALLOCS.load(Ordering::Relaxed);
    push_line_for_gate(&framer, &mut frames, br#"{"done": true, "tokens": 64}"#);
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;
    assert_eq!(
        frames.len(),
        2,
        "a done line must produce the final chunk + [DONE]"
    );
    assert!(
        allocs <= 1,
        "the done path allocated {allocs} times — the final chunk buffer is the only allowed cost"
    );
}

#[test]
fn token_frame_allocates_exactly_one() {
    // The token frame's single allocation is the outgoing chunk buffer itself
    // (head + spliced text/token + SSE framing, frozen into a Bytes).
    // Anything beyond one per frame means the parse path is materializing
    // intermediates again (Value trees, re-escaped strings, format! churn).
    let _g = measure();
    warm();
    let framer = framer_for_gate();
    let mut frames = Vec::with_capacity(1);
    let before = ALLOCS.load(Ordering::Relaxed);
    let line = br#"{"token": 7, "text": "tok"}"#;
    push_line_for_gate(&framer, &mut frames, line);
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;
    assert_eq!(frames.len(), 1);
    assert!(
        allocs <= 1,
        "one token frame must cost exactly ≤1 allocation, got {allocs}"
    );
}
