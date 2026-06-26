//! `PostgresMemoryProvider` — PostgreSQL + pgvector backing storage.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use cel_memory::{
    CallerScope, MemoryChunk, MemoryError, MemoryProvider, MemoryQuery, MemorySession, MemoryStats,
    MemoryTier, NewMemoryChunk, NewMemorySession, Result as MemoryResult, SessionFilter,
    SessionOutcome,
};
use chrono::Utc;
use pgvector::Vector;
use sqlx::postgres::PgRow;
use sqlx::{PgPool, Postgres, QueryBuilder, Row};
use uuid::Uuid;

use crate::error::PostgresMemoryError;
use crate::util::{
    chunk_matches_query, kind_str, outcome_str, row_to_chunk, rrf, source_str, str_to_outcome,
    tier_str, EMBEDDING_DIM,
};
use cel_memory::Embedder;


const GET_CHUNK_SQL: &str =
    "SELECT id, created_at, kind, tier, source, session_id, project_root, caller_id, content, \
     metadata, importance, pinned, shareable, superseded_by, embedding_model, embedding_dim \
     FROM memory_chunks WHERE id = $1";

const CHUNKS_BY_IDS_SQL: &str =
    "SELECT id, created_at, kind, tier, source, session_id, project_root, caller_id, content, \
     metadata, importance, pinned, shareable, superseded_by, embedding_model, embedding_dim \
     FROM memory_chunks WHERE id = ANY($1)";

/// PostgreSQL-backed [`MemoryProvider`] using pgvector + tsvector retrieval.
pub struct PostgresMemoryProvider {
    pool: PgPool,
    embedder: Arc<dyn Embedder>,
    write_hook: Option<Arc<dyn cel_memory::MemoryWriteHook>>,
    summarizer: Option<Arc<dyn cel_memory::Summarizer>>,
}

impl PostgresMemoryProvider {
    /// Connect to PostgreSQL, run migrations, and return a ready provider.
    ///
    /// The embedder must declare dimension [`EMBEDDING_DIM`] (384) to match the
    /// schema. Use [`cel_memory::MockEmbedder`] in tests.
    pub async fn connect(
        database_url: &str,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, PostgresMemoryError> {
        if embedder.dim() != EMBEDDING_DIM {
            return Err(PostgresMemoryError::DimMismatch {
                expected: EMBEDDING_DIM,
                actual: embedder.dim(),
            });
        }
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(8)
            .connect(database_url)
            .await?;
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|e| PostgresMemoryError::Migration(e.to_string()))?;
        Ok(Self {
            pool,
            embedder,
            write_hook: None,
            summarizer: None,
        })
    }

    /// Attach a write hook consulted before every persist.
    pub fn with_write_hook(mut self, hook: Arc<dyn cel_memory::MemoryWriteHook>) -> Self {
        self.write_hook = Some(hook);
        self
    }

    /// Attach a summarizer for session summaries and rollups (Phase 3).
    pub fn with_summarizer(mut self, summarizer: Arc<dyn cel_memory::Summarizer>) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// Pool accessor for integration tests.
    #[doc(hidden)]
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

#[async_trait]
impl MemoryProvider for PostgresMemoryProvider {
    async fn retrieve(&self, query: MemoryQuery) -> MemoryResult<Vec<MemoryChunk>> {
        if query.text.trim().is_empty() {
            return Err(MemoryError::InvalidArgument(
                "query.text must not be empty".into(),
            ));
        }
        let k = query.k.max(1);
        let candidate_k = (3 * k).max(16) as i64;
        let q_embedding = self.embedder.embed(&query.text).await?;
        let q_vec = Vector::from(q_embedding);
        let vector_sql = match query.caller_scope {
            CallerScope::Global => {
                "SELECT c.id
                 FROM memory_chunks c
                 INNER JOIN memory_vectors v ON v.chunk_id = c.id
                 ORDER BY v.embedding <=> $1
                 LIMIT $2"
            }
            CallerScope::Own | CallerScope::OwnPlusShared => {
                "SELECT c.id
                 FROM memory_chunks c
                 INNER JOIN memory_vectors v ON v.chunk_id = c.id
                 WHERE c.caller_id = $2
                 ORDER BY v.embedding <=> $1
                 LIMIT $3"
            }
        };
        let vector_rows = match query.caller_scope {
            CallerScope::Global => {
                sqlx::query(vector_sql)
                    .bind(q_vec.clone())
                    .bind(candidate_k)
                    .fetch_all(&self.pool)
                    .await
            }
            CallerScope::Own => {
                sqlx::query(vector_sql)
                    .bind(q_vec.clone())
                    .bind(&query.caller_id)
                    .bind(candidate_k)
                    .fetch_all(&self.pool)
                    .await
            }
            CallerScope::OwnPlusShared => {
                sqlx::query(
                    "SELECT c.id
                     FROM memory_chunks c
                     INNER JOIN memory_vectors v ON v.chunk_id = c.id
                     WHERE (c.caller_id = $2 OR c.shareable = TRUE)
                     ORDER BY v.embedding <=> $1
                     LIMIT $3",
                )
                .bind(q_vec.clone())
                .bind(&query.caller_id)
                .bind(candidate_k)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let fts_rows = match query.caller_scope {
            CallerScope::Global => {
                sqlx::query(
                    "SELECT c.id
                     FROM memory_chunks c
                     WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', $1)
                     ORDER BY ts_rank(
                         to_tsvector('english', c.content),
                         plainto_tsquery('english', $1)
                     ) DESC
                     LIMIT $2",
                )
                .bind(&query.text)
                .bind(candidate_k)
                .fetch_all(&self.pool)
                .await
            }
            CallerScope::Own => {
                sqlx::query(
                    "SELECT c.id
                     FROM memory_chunks c
                     WHERE c.caller_id = $1
                       AND to_tsvector('english', c.content) @@ plainto_tsquery('english', $2)
                     ORDER BY ts_rank(
                         to_tsvector('english', c.content),
                         plainto_tsquery('english', $2)
                     ) DESC
                     LIMIT $3",
                )
                .bind(&query.caller_id)
                .bind(&query.text)
                .bind(candidate_k)
                .fetch_all(&self.pool)
                .await
            }
            CallerScope::OwnPlusShared => {
                sqlx::query(
                    "SELECT c.id
                     FROM memory_chunks c
                     WHERE (c.caller_id = $1 OR c.shareable = TRUE)
                       AND to_tsvector('english', c.content) @@ plainto_tsquery('english', $2)
                     ORDER BY ts_rank(
                         to_tsvector('english', c.content),
                         plainto_tsquery('english', $2)
                     ) DESC
                     LIMIT $3",
                )
                .bind(&query.caller_id)
                .bind(&query.text)
                .bind(candidate_k)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let mut scores: HashMap<String, f32> = HashMap::new();
        for (rank, row) in vector_rows.into_iter().enumerate() {
            let id: String = row
                .try_get("id")
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            *scores.entry(id).or_insert(0.0) += rrf(rank, 60.0);
        }
        for (rank, row) in fts_rows.into_iter().enumerate() {
            let id: String = row
                .try_get("id")
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            *scores.entry(id).or_insert(0.0) += rrf(rank, 60.0);
        }

        if scores.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<String> = scores.keys().cloned().collect();
        let rows = sqlx::query(CHUNKS_BY_IDS_SQL)
            .bind(&ids)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let mut filtered: Vec<MemoryChunk> = rows
            .iter()
            .filter_map(|row| row_to_chunk(row).ok())
            .filter(|c| chunk_matches_query(c, &query))
            .collect();

        filtered.sort_by(|a, b| {
            let sa = scores.get(&a.id).copied().unwrap_or(0.0);
            let sb = scores.get(&b.id).copied().unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        filtered.truncate(k);
        Ok(filtered)
    }

    async fn get(&self, chunk_id: &str) -> MemoryResult<Option<MemoryChunk>> {
        let row = sqlx::query(GET_CHUNK_SQL)
            .bind(chunk_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        row.map(|r| row_to_chunk(&r)).transpose()
    }

    async fn get_session(&self, session_id: &str) -> MemoryResult<Option<MemorySession>> {
        let row = sqlx::query(
            "SELECT id, started_at, ended_at, caller_id, title, summary, outcome, metadata
             FROM memory_sessions WHERE id = $1",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| MemoryError::Storage(e.to_string()))?;
        row.map(|r| row_to_session(&r)).transpose()
    }

    async fn list_sessions(&self, filter: SessionFilter) -> MemoryResult<Vec<MemorySession>> {
        let mut builder: QueryBuilder<Postgres> = QueryBuilder::new(
            "SELECT id, started_at, ended_at, caller_id, title, summary, outcome, metadata \
             FROM memory_sessions WHERE TRUE",
        );
        if let Some(caller) = &filter.caller_id {
            builder.push(" AND caller_id = ");
            builder.push_bind(caller);
        }
        if filter.open_only {
            builder.push(" AND outcome = 'open'");
        } else if let Some(outcome) = filter.outcome {
            builder.push(" AND outcome = ");
            builder.push_bind(outcome_str(outcome));
        }
        builder.push(" ORDER BY started_at DESC");
        let rows = builder
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        rows.iter().map(row_to_session).collect()
    }

    async fn write(&self, new_chunk: NewMemoryChunk) -> MemoryResult<MemoryChunk> {
        if new_chunk.content.trim().is_empty() {
            return Err(MemoryError::InvalidArgument(
                "content must not be empty".into(),
            ));
        }
        if let Some(hook) = &self.write_hook {
            match hook.before_write(&new_chunk).await? {
                cel_memory::WriteDecision::Allow => {}
                cel_memory::WriteDecision::Redact { reason } => {
                    return Ok(MemoryChunk {
                        id: Uuid::now_v7().to_string(),
                        created_at: Utc::now(),
                        kind: new_chunk.kind,
                        tier: MemoryTier::Session,
                        source: new_chunk.source,
                        session_id: new_chunk.session_id,
                        project_root: new_chunk.project_root,
                        caller_id: new_chunk.caller_id,
                        content: format!("<redacted: {reason}>"),
                        metadata: serde_json::json!({"redacted": true, "reason": reason}),
                        importance: 0.0,
                        pinned: false,
                        shareable: false,
                        superseded_by: None,
                        embedding_model: "none".into(),
                        embedding_dim: 0,
                    });
                }
            }
        }

        let id = Uuid::now_v7().to_string();
        let created_at = Utc::now();
        let importance = cel_memory::score_importance(&new_chunk);
        let embedding = self.embedder.embed(&new_chunk.content).await?;
        if embedding.len() != EMBEDDING_DIM {
            return Err(MemoryError::Internal(format!(
                "embedder produced dim {}, expected {EMBEDDING_DIM}",
                embedding.len()
            )));
        }

        let chunk = MemoryChunk {
            id: id.clone(),
            created_at,
            kind: new_chunk.kind,
            tier: MemoryTier::Session,
            source: new_chunk.source,
            session_id: new_chunk.session_id.clone(),
            project_root: new_chunk.project_root.clone(),
            caller_id: new_chunk.caller_id.clone(),
            content: new_chunk.content.clone(),
            metadata: new_chunk.metadata.clone(),
            importance,
            pinned: new_chunk.pinned,
            shareable: new_chunk.shareable,
            superseded_by: None,
            embedding_model: self.embedder.model_name().to_string(),
            embedding_dim: EMBEDDING_DIM as u32,
        };

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        sqlx::query(
            "INSERT INTO memory_chunks(
                id, created_at, kind, tier, source, session_id, project_root,
                caller_id, content, metadata, importance, pinned, shareable,
                superseded_by, embedding_model, embedding_dim
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)",
        )
        .bind(&chunk.id)
        .bind(chunk.created_at)
        .bind(kind_str(chunk.kind))
        .bind(tier_str(chunk.tier))
        .bind(source_str(chunk.source))
        .bind(&chunk.session_id)
        .bind(&chunk.project_root)
        .bind(&chunk.caller_id)
        .bind(&chunk.content)
        .bind(&chunk.metadata)
        .bind(chunk.importance)
        .bind(chunk.pinned)
        .bind(chunk.shareable)
        .bind::<Option<String>>(None)
        .bind(&chunk.embedding_model)
        .bind(chunk.embedding_dim as i32)
        .execute(&mut *tx)
        .await
        .map_err(|e| MemoryError::Storage(e.to_string()))?;

        sqlx::query("INSERT INTO memory_vectors (chunk_id, embedding) VALUES ($1, $2)")
            .bind(&chunk.id)
            .bind(Vector::from(embedding))
            .execute(&mut *tx)
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(chunk)
    }

    async fn write_batch(&self, chunks: Vec<NewMemoryChunk>) -> MemoryResult<Vec<MemoryChunk>> {
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            out.push(self.write(chunk).await?);
        }
        Ok(out)
    }

    async fn open_session(&self, init: NewMemorySession) -> MemoryResult<MemorySession> {
        let session = MemorySession {
            id: Uuid::now_v7().to_string(),
            started_at: Utc::now(),
            ended_at: None,
            caller_id: init.caller_id,
            title: init.title,
            summary: None,
            outcome: SessionOutcome::Open,
            metadata: init.metadata,
        };
        sqlx::query(
            "INSERT INTO memory_sessions (id, started_at, ended_at, caller_id, title, summary, outcome, metadata)
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
        )
        .bind(&session.id)
        .bind(session.started_at)
        .bind(session.ended_at)
        .bind(&session.caller_id)
        .bind(&session.title)
        .bind(&session.summary)
        .bind(outcome_str(session.outcome))
        .bind(&session.metadata)
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(session)
    }

    async fn close_session(&self, session_id: &str, outcome: SessionOutcome) -> MemoryResult<()> {
        let ended_at = Utc::now();
        let rows =
            sqlx::query("UPDATE memory_sessions SET ended_at = $1, outcome = $2 WHERE id = $3")
                .bind(ended_at)
                .bind(outcome_str(outcome))
                .bind(session_id)
                .execute(&self.pool)
                .await
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
        if rows.rows_affected() == 0 {
            return Err(MemoryError::NotFound(session_id.into()));
        }
        Ok(())
    }

    async fn rename_session(&self, session_id: &str, title: &str) -> MemoryResult<()> {
        let rows = sqlx::query("UPDATE memory_sessions SET title = $1 WHERE id = $2")
            .bind(title)
            .bind(session_id)
            .execute(&self.pool)
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        if rows.rows_affected() == 0 {
            return Err(MemoryError::NotFound(session_id.into()));
        }
        Ok(())
    }

    async fn stats(&self) -> MemoryResult<MemoryStats> {
        let total_chunks: i64 = sqlx::query_scalar("SELECT COUNT(*)::bigint FROM memory_chunks")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        let total_sessions: i64 =
            sqlx::query_scalar("SELECT COUNT(*)::bigint FROM memory_sessions")
                .fetch_one(&self.pool)
                .await
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
        let embedding_model: Option<String> = sqlx::query_scalar(
            "SELECT embedding_model FROM memory_chunks ORDER BY created_at DESC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(MemoryStats {
            total_chunks: total_chunks as usize,
            total_sessions: total_sessions as usize,
            embedding_model,
            ..MemoryStats::default()
        })
    }

    async fn summarize_session(&self, _session_id: &str) -> MemoryResult<MemoryChunk> {
        if self.summarizer.is_none() {
            return Err(MemoryError::NotImplemented(
                "PostgresMemoryProvider::summarize_session — attach summarizer via with_summarizer",
            ));
        }
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::summarize_session — Phase 3",
        ))
    }

    async fn rollup_day(&self, _date: chrono::NaiveDate) -> MemoryResult<Vec<MemoryChunk>> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::rollup_day — Phase 3",
        ))
    }

    async fn rollup_rule_week(
        &self,
        _rule_id: &str,
        _week_start: chrono::NaiveDate,
    ) -> MemoryResult<MemoryChunk> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::rollup_rule_week — Phase 3",
        ))
    }

    async fn run_aging_sweep(&self) -> MemoryResult<cel_memory::AgingReport> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::run_aging_sweep — Phase 2",
        ))
    }

    async fn re_embed_all(&self, _target_model: &str) -> MemoryResult<cel_memory::ReEmbedReport> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::re_embed_all — Phase 4",
        ))
    }

    async fn export(
        &self,
        _filter: cel_memory::ExportFilter,
    ) -> MemoryResult<cel_memory::ExportBundle> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::export — Phase 2",
        ))
    }

    async fn pin(&self, _chunk_id: &str, _pinned: bool) -> MemoryResult<()> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::pin — Phase 2",
        ))
    }

    async fn update_importance(&self, _chunk_id: &str, _importance: f32) -> MemoryResult<()> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::update_importance — Phase 2",
        ))
    }

    async fn supersede(&self, _old_id: &str, _new_id: &str) -> MemoryResult<()> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::supersede — Phase 2",
        ))
    }

    async fn record_access(
        &self,
        _chunk_id: &str,
        _retrieved_by: &str,
        _used: bool,
    ) -> MemoryResult<()> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::record_access — Phase 2",
        ))
    }

    async fn delete(
        &self,
        _chunk_id: &str,
        _reason: cel_memory::EvictionReason,
    ) -> MemoryResult<()> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::delete — Phase 2",
        ))
    }

    async fn delete_matching(
        &self,
        _predicate: cel_memory::MemoryPredicate,
        _reason: cel_memory::EvictionReason,
    ) -> MemoryResult<usize> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::delete_matching — Phase 2",
        ))
    }

    async fn purge_all(&self) -> MemoryResult<cel_memory::PurgeReport> {
        Err(MemoryError::NotImplemented(
            "PostgresMemoryProvider::purge_all — Phase 2",
        ))
    }
}

fn row_to_session(row: &PgRow) -> MemoryResult<MemorySession> {
    let metadata: serde_json::Value = row.try_get("metadata").unwrap_or(serde_json::json!({}));
    Ok(MemorySession {
        id: row
            .try_get("id")
            .map_err(|e| MemoryError::Storage(e.to_string()))?,
        started_at: row
            .try_get("started_at")
            .map_err(|e| MemoryError::Storage(e.to_string()))?,
        ended_at: row.try_get("ended_at").ok(),
        caller_id: row
            .try_get("caller_id")
            .map_err(|e| MemoryError::Storage(e.to_string()))?,
        title: row.try_get("title").ok(),
        summary: row.try_get("summary").ok(),
        outcome: str_to_outcome(
            &row.try_get::<String, _>("outcome")
                .map_err(|e| MemoryError::Storage(e.to_string()))?,
        )?,
        metadata,
    })
}
