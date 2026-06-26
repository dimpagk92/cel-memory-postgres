# cel-memory-postgres

[![crates.io](https://img.shields.io/crates/v/cel-memory-postgres.svg)](https://crates.io/crates/cel-memory-postgres)
[![docs.rs](https://docs.rs/cel-memory-postgres/badge.svg)](https://docs.rs/cel-memory-postgres)
[![CI](https://github.com/dimpagk92/cel-memory-postgres/actions/workflows/ci.yml/badge.svg)](https://github.com/dimpagk92/cel-memory-postgres/actions/workflows/ci.yml)

PostgreSQL + pgvector memory backend for AI agents. Implements
[`cel-memory`](https://crates.io/crates/cel-memory)'s `MemoryProvider` trait with
hybrid vector + full-text retrieval.

**Status:** v0.1.0 starter — Phase 0–1: connect/migrate, write/get, hybrid
retrieve, sessions, stats. Summarization, aging, export, and bulk mutations
return `NotImplemented` until later phases (see [BACKENDS.md](https://github.com/dimpagk92/cel-memory/blob/main/BACKENDS.md)).

## Purpose

Use `cel-memory-postgres` when you need durable agent memory in a shared
PostgreSQL deployment — multi-tenant services, team backends, or production
hosts that already run Postgres. Swap it in wherever code holds
`Arc<dyn MemoryProvider>` today (e.g. beside [`cel-memory-sqlite`](https://crates.io/crates/cel-memory-sqlite)).

## Requirements

- PostgreSQL 14+ with the [pgvector](https://github.com/pgvector/pgvector) extension
- Embedding width **384** (matches [`MockEmbedder`](https://docs.rs/cel-memory/latest/cel_memory/struct.MockEmbedder.html) / bge-small)

## Example

```rust
use std::sync::Arc;
use cel_memory_postgres::{MockEmbedder, PostgresMemoryProvider};

let provider = PostgresMemoryProvider::connect(
    "postgres://postgres:postgres@localhost:5432/cel_memory",
    Arc::new(MockEmbedder::new()),
).await?;
// Use as cel_memory::MemoryProvider — same trait as BasicMemoryProvider.
```

Run the complete example (requires `CEL_MEMORY_POSTGRES_URL`):

```sh
CEL_MEMORY_POSTGRES_URL=postgres://postgres:postgres@localhost:5432/cel_memory \
  cargo run --example basic
```

## Testing

Integration tests use the same URL:

```sh
docker run -d --name cel-pg-test \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=cel_memory_test \
  -p 54329:5432 \
  pgvector/pgvector:pg16

export CEL_MEMORY_POSTGRES_URL=postgres://postgres:postgres@localhost:54329/cel_memory_test
cargo test
```

## What's included (Phase 0–1)

- `PostgresMemoryProvider::connect` — pool + sqlx migrations
- `write` / `get` / `write_batch`
- Hybrid `retrieve` — pgvector cosine distance + `tsvector` FTS, fused with RRF
- Session lifecycle — `open_session`, `close_session`, `rename_session`, `get_session`, `list_sessions`
- `stats`

## Roadmap

| Phase | Methods |
|-------|---------|
| 2 | `pin`, `delete`, `purge_all`, `run_aging_sweep`, `export`, … |
| 3 | `summarize_session`, `rollup_day`, `rollup_rule_week` (via injected [`Summarizer`](https://docs.rs/cel-memory/latest/cel_memory/trait.Summarizer.html)) |
| 4 | `re_embed_all`, HNSW/IVFFlat indexes |

## License

Apache-2.0
