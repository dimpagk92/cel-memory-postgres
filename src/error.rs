//! Storage-side error type. Converted to [`MemoryError`] on the way out
//! of the [`PostgresMemoryProvider`](crate::PostgresMemoryProvider).

use cel_memory::MemoryError;
use thiserror::Error;

/// Errors that originate inside the PostgreSQL memory backend.
#[derive(Debug, Error)]
pub enum PostgresMemoryError {
    /// sqlx returned an error.
    #[error("postgres error: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// Migration failed to apply.
    #[error("migration failed: {0}")]
    Migration(String),

    /// The embedder dimension does not match the schema (`vector(384)`).
    #[error("embedding dim mismatch: provider expects {expected}, embedder declares {actual}")]
    DimMismatch {
        /// Dim the schema expects.
        expected: usize,
        /// Dim the embedder declares.
        actual: usize,
    },

    /// JSON serialization failed inside the storage layer.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<PostgresMemoryError> for MemoryError {
    fn from(e: PostgresMemoryError) -> Self {
        match e {
            PostgresMemoryError::DimMismatch { expected, actual } => MemoryError::InvalidArgument(
                format!("embedding dim mismatch: expected {expected}, got {actual}"),
            ),
            PostgresMemoryError::Json(err) => MemoryError::Storage(format!("json: {err}")),
            other => MemoryError::Storage(other.to_string()),
        }
    }
}
