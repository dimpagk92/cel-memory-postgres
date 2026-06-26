-- cel-memory-postgres initial schema (Phase 0–1 starter).
-- Requires pgvector. Embedding width matches MockEmbedder / bge-small (384).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE memory_chunks (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL,
    kind            TEXT NOT NULL,
    tier            TEXT NOT NULL,
    source          TEXT NOT NULL,
    session_id      TEXT,
    project_root    TEXT,
    caller_id       TEXT NOT NULL,
    content         TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    importance      REAL NOT NULL DEFAULT 0.5,
    pinned          BOOLEAN NOT NULL DEFAULT FALSE,
    shareable       BOOLEAN NOT NULL DEFAULT FALSE,
    superseded_by   TEXT,
    embedding_model TEXT NOT NULL,
    embedding_dim   INTEGER NOT NULL
);

CREATE INDEX idx_memory_chunks_kind_tier ON memory_chunks (kind, tier);
CREATE INDEX idx_memory_chunks_session ON memory_chunks (session_id);
CREATE INDEX idx_memory_chunks_caller ON memory_chunks (caller_id);
CREATE INDEX idx_memory_chunks_created ON memory_chunks (created_at DESC);
CREATE INDEX idx_memory_chunks_project ON memory_chunks (project_root);
CREATE INDEX idx_memory_chunks_content_fts
    ON memory_chunks USING gin (to_tsvector('english', content));

CREATE TABLE memory_vectors (
    chunk_id   TEXT PRIMARY KEY REFERENCES memory_chunks (id) ON DELETE CASCADE,
    embedding  vector(384) NOT NULL
);

CREATE TABLE memory_sessions (
    id          TEXT PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL,
    ended_at    TIMESTAMPTZ,
    caller_id   TEXT NOT NULL,
    title       TEXT,
    summary     TEXT,
    outcome     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_memory_sessions_caller ON memory_sessions (caller_id);
CREATE INDEX idx_memory_sessions_outcome ON memory_sessions (outcome);
CREATE INDEX idx_memory_sessions_started ON memory_sessions (started_at DESC);

CREATE TABLE memory_summary_members (
    rollup_id  TEXT NOT NULL,
    member_id  TEXT NOT NULL,
    PRIMARY KEY (rollup_id, member_id)
);

CREATE TABLE memory_access_log (
    ts            TIMESTAMPTZ NOT NULL,
    chunk_id      TEXT NOT NULL,
    retrieved_by  TEXT NOT NULL,
    query_hash    TEXT NOT NULL,
    rank          INTEGER NOT NULL,
    used          BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_memory_access_ts ON memory_access_log (ts DESC);

CREATE TABLE memory_eviction_log (
    ts        TIMESTAMPTZ NOT NULL,
    chunk_id  TEXT NOT NULL,
    reason    TEXT NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_memory_eviction_ts ON memory_eviction_log (ts DESC);
