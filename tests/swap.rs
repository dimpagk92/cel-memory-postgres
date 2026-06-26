//! Crate-level conformance test: `PostgresMemoryProvider` must work through
//! the locked `cel_memory::MemoryProvider` trait surface.
//!
//! Requires PostgreSQL with pgvector. Set `CEL_MEMORY_POSTGRES_URL`, or run
//! CI which starts `pgvector/pgvector:pg16` as a service.

use std::sync::Arc;

use cel_memory::{
    assert_retrieve_finds_written, assert_session_lifecycle, assert_write_get_stats, MemoryProvider,
};
use cel_memory_postgres::{MockEmbedder, PostgresMemoryProvider};

fn postgres_url() -> Option<String> {
    std::env::var("CEL_MEMORY_POSTGRES_URL").ok()
}

#[tokio::test]
async fn postgres_provider_works_through_locked_trait() {
    let Some(url) = postgres_url() else {
        eprintln!("CEL_MEMORY_POSTGRES_URL not set — skipping postgres conformance");
        return;
    };

    let embedder = Arc::new(MockEmbedder::new());
    let provider = PostgresMemoryProvider::connect(&url, embedder)
        .await
        .expect("connect to postgres");
    sqlx::query("TRUNCATE memory_chunks, memory_vectors, memory_sessions CASCADE")
        .execute(provider.pool())
        .await
        .expect("truncate test tables");

    let memory: Arc<dyn MemoryProvider> = Arc::new(provider);

    let (_chunk, stats) = assert_write_get_stats(memory.clone(), "user asked about the Q4 report")
        .await
        .unwrap();
    assert_eq!(stats.total_chunks, 1);
    assert_eq!(stats.embedding_model.as_deref(), Some("mock-384"));

    assert_retrieve_finds_written(memory.clone(), "Q4 revenue forecast")
        .await
        .unwrap();
    assert_session_lifecycle(memory).await.unwrap();
}
