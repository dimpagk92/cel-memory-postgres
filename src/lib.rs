//! PostgreSQL + pgvector backend for [`cel_memory::MemoryProvider`].
//!
//! Phase 0–1 starter: connect/migrate, write/get, hybrid vector + FTS
//! retrieval, session lifecycle, and stats. Summarization, aging, export,
//! and governance mutations return [`cel_memory::MemoryError::NotImplemented`]
//! until later phases — see the crate README.

mod error;
mod provider;
mod util;

pub use error::PostgresMemoryError;
pub use provider::PostgresMemoryProvider;

pub use cel_memory::{Embedder, MockEmbedder};
