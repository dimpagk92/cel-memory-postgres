//! Minimal `cel-memory-postgres` example.
//!
//! Requires PostgreSQL with pgvector and `CEL_MEMORY_POSTGRES_URL`.
//!
//! Run with:
//!   CEL_MEMORY_POSTGRES_URL=postgres://postgres:postgres@localhost:5432/cel_memory \
//!     cargo run --example basic

use std::sync::Arc;

use cel_memory::{
    ChunkKind, ChunkSource, MemoryProvider, NewMemoryChunk, NewMemorySession, SessionOutcome,
};
use cel_memory_postgres::{MockEmbedder, PostgresMemoryProvider};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = std::env::var("CEL_MEMORY_POSTGRES_URL")
        .expect("set CEL_MEMORY_POSTGRES_URL to a postgres:// URL with pgvector enabled");

    let embedder = Arc::new(MockEmbedder::new());
    let provider = PostgresMemoryProvider::connect(&url, embedder).await?;
    println!("connected PostgresMemoryProvider");

    let session = provider
        .open_session(NewMemorySession {
            caller_id: "example".into(),
            title: Some("basic-example".into()),
            metadata: serde_json::json!({}),
        })
        .await?;
    println!("opened session {}", session.id);

    let mut written = Vec::with_capacity(10);
    for i in 0..10 {
        let chunk = provider
            .write(NewMemoryChunk {
                kind: ChunkKind::Chat,
                source: ChunkSource::Embedded,
                session_id: Some(session.id.clone()),
                project_root: None,
                caller_id: "example".into(),
                content: format!("chunk number {i} — synthetic content for the basic example"),
                metadata: serde_json::json!({ "index": i }),
                importance: None,
                shareable: false,
                pinned: false,
            })
            .await?;
        written.push(chunk);
    }
    println!("wrote {} chunks", written.len());

    let stats = provider.stats().await?;
    println!(
        "stats: total_chunks={}, total_sessions={}, embedding_model={:?}",
        stats.total_chunks, stats.total_sessions, stats.embedding_model
    );

    provider
        .close_session(&session.id, SessionOutcome::Success)
        .await?;
    println!("closed session and finished cleanly");

    Ok(())
}
