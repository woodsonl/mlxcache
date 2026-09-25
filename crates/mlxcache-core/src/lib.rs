//! mlxcache-core: cache contract, policy, prefix index, single-flight.
//!
//! The engine-agnostic cache contract lives here. See docs/contract-spec.md.

pub mod contract;
pub mod index;
pub mod policy;
pub mod singleflight;
