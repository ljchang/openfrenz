//! Host clocks. All timestamps in this crate are taken from one process-wide monotonic anchor.

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct Anchor {
    instant: Instant,
    wall_ns: i64,
}

fn anchor() -> &'static Anchor {
    static ANCHOR: OnceLock<Anchor> = OnceLock::new();
    ANCHOR.get_or_init(|| Anchor {
        instant: Instant::now(),
        wall_ns: wall_ns(),
    })
}

/// Monotonic nanoseconds since the process anchor. Never goes backwards.
pub fn mono_ns() -> i64 {
    i64::try_from(anchor().instant.elapsed().as_nanos()).unwrap_or(i64::MAX)
}

/// Wall-clock (Unix) nanoseconds. May jump if the system clock is adjusted.
pub fn wall_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_nanos()).ok())
        .unwrap_or_default()
}

/// Wall-clock nanoseconds at which `mono_ns() == 0`.
pub fn anchor_wall_ns() -> i64 {
    anchor().wall_ns
}
