//! Row mapping and query-filter helpers shared by the provider.

use cel_memory::{
    CallerScope, ChunkKind, ChunkSource, MemoryChunk, MemoryError, MemoryQuery, MemoryTier,
    Result as MemoryResult,
};
use chrono::{DateTime, Utc};
use sqlx::postgres::PgRow;
use sqlx::Row;

pub(crate) const EMBEDDING_DIM: usize = 384;

pub(crate) fn kind_str(k: ChunkKind) -> &'static str {
    match k {
        ChunkKind::Chat => "chat",
        ChunkKind::Action => "action",
        ChunkKind::Fire => "fire",
        ChunkKind::Observation => "observation",
        ChunkKind::Correction => "correction",
        ChunkKind::JobSummary => "job_summary",
        ChunkKind::Context => "context",
        ChunkKind::Rollup => "rollup",
    }
}

pub(crate) fn str_to_kind(s: &str) -> Option<ChunkKind> {
    match s {
        "chat" => Some(ChunkKind::Chat),
        "action" => Some(ChunkKind::Action),
        "fire" => Some(ChunkKind::Fire),
        "observation" => Some(ChunkKind::Observation),
        "correction" => Some(ChunkKind::Correction),
        "job_summary" => Some(ChunkKind::JobSummary),
        "context" => Some(ChunkKind::Context),
        "rollup" => Some(ChunkKind::Rollup),
        _ => None,
    }
}

pub(crate) fn tier_str(t: MemoryTier) -> &'static str {
    match t {
        MemoryTier::Session => "session",
        MemoryTier::LongTerm => "long_term",
    }
}

pub(crate) fn str_to_tier(s: &str) -> Option<MemoryTier> {
    match s {
        "session" => Some(MemoryTier::Session),
        "long_term" => Some(MemoryTier::LongTerm),
        _ => None,
    }
}

pub(crate) fn source_str(s: ChunkSource) -> &'static str {
    match s {
        ChunkSource::Embedded => "embedded",
        ChunkSource::Mcp => "mcp",
        ChunkSource::Gateway => "gateway",
        ChunkSource::Matcher => "matcher",
        ChunkSource::Perception => "cortex",
        ChunkSource::System => "system",
    }
}

pub(crate) fn str_to_source(s: &str) -> Option<ChunkSource> {
    match s {
        "embedded" => Some(ChunkSource::Embedded),
        "mcp" => Some(ChunkSource::Mcp),
        "gateway" => Some(ChunkSource::Gateway),
        "matcher" => Some(ChunkSource::Matcher),
        "cortex" | "perception" => Some(ChunkSource::Perception),
        "system" => Some(ChunkSource::System),
        _ => None,
    }
}

pub(crate) fn outcome_str(o: cel_memory::SessionOutcome) -> &'static str {
    use cel_memory::SessionOutcome;
    match o {
        SessionOutcome::Open => "open",
        SessionOutcome::Success => "success",
        SessionOutcome::Failure => "failure",
        SessionOutcome::Aborted => "aborted",
    }
}

pub(crate) fn str_to_outcome(s: &str) -> MemoryResult<cel_memory::SessionOutcome> {
    use cel_memory::SessionOutcome;
    match s {
        "open" => Ok(SessionOutcome::Open),
        "success" => Ok(SessionOutcome::Success),
        "failure" => Ok(SessionOutcome::Failure),
        "aborted" => Ok(SessionOutcome::Aborted),
        other => Err(MemoryError::Storage(format!("unknown outcome: {other}"))),
    }
}

pub(crate) fn row_to_chunk(row: &PgRow) -> MemoryResult<MemoryChunk> {
    let metadata: serde_json::Value = row.try_get("metadata").unwrap_or(serde_json::json!({}));
    Ok(MemoryChunk {
        id: row
            .try_get("id")
            .map_err(|e| MemoryError::Storage(e.to_string()))?,
        created_at: row
            .try_get::<DateTime<Utc>, _>("created_at")
            .map_err(|e| MemoryError::Storage(e.to_string()))?,
        kind: str_to_kind(
            &row.try_get::<String, _>("kind")
                .map_err(|e| MemoryError::Storage(e.to_string()))?,
        )
        .unwrap_or(ChunkKind::Chat),
        tier: str_to_tier(
            &row.try_get::<String, _>("tier")
                .map_err(|e| MemoryError::Storage(e.to_string()))?,
        )
        .unwrap_or(MemoryTier::Session),
        source: str_to_source(
            &row.try_get::<String, _>("source")
                .map_err(|e| MemoryError::Storage(e.to_string()))?,
        )
        .unwrap_or(ChunkSource::System),
        session_id: row.try_get("session_id").ok(),
        project_root: row.try_get("project_root").ok(),
        caller_id: row
            .try_get("caller_id")
            .map_err(|e| MemoryError::Storage(e.to_string()))?,
        content: row
            .try_get("content")
            .map_err(|e| MemoryError::Storage(e.to_string()))?,
        metadata,
        importance: row.try_get::<f32, _>("importance").unwrap_or(0.5),
        pinned: row.try_get::<bool, _>("pinned").unwrap_or(false),
        shareable: row.try_get::<bool, _>("shareable").unwrap_or(false),
        superseded_by: row.try_get("superseded_by").ok(),
        embedding_model: row
            .try_get("embedding_model")
            .map_err(|e| MemoryError::Storage(e.to_string()))?,
        embedding_dim: row.try_get::<i32, _>("embedding_dim").unwrap_or(0) as u32,
    })
}

pub(crate) fn chunk_matches_query(c: &MemoryChunk, q: &MemoryQuery) -> bool {
    if let Some(kinds) = &q.kinds {
        if !kinds.contains(&c.kind) {
            return false;
        }
    }
    if !q.include_rollups && c.kind == ChunkKind::Rollup {
        return false;
    }
    if let Some(since) = q.since {
        if c.created_at < since {
            return false;
        }
    }
    if let Some(until) = q.until {
        if c.created_at > until {
            return false;
        }
    }
    if let Some(sid) = &q.session_id {
        if c.session_id.as_deref() != Some(sid.as_str()) {
            return false;
        }
    }
    if let Some(prefix) = &q.project_root_prefix {
        match &c.project_root {
            Some(root) if root.starts_with(prefix.as_str()) => {}
            _ => return false,
        }
    }
    if let Some(min) = q.min_importance {
        if c.importance < min {
            return false;
        }
    }
    match q.caller_scope {
        CallerScope::Global => true,
        CallerScope::Own => c.caller_id == q.caller_id,
        CallerScope::OwnPlusShared => c.caller_id == q.caller_id || c.shareable,
    }
}

pub(crate) fn rrf(rank: usize, k: f32) -> f32 {
    1.0 / (k + rank as f32)
}
