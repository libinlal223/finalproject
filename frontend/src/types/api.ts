/**
 * TypeScript interface contracts mirroring backend schemas (app/models/schemas.py).
 */

export type AgentRole =
  | 'planner'
  | 'executor_a'
  | 'executor_b'
  | 'critic'
  | 'judge'
  | 'baseline';

export type DebateStatus =
  | 'pending'
  | 'running'
  | 'converged'
  | 'max_rounds'
  | 'completed'
  | 'error';

export type LLMProvider = 'groq' | 'gemini' | 'mock';

export interface EvidenceItem {
  title: string;
  url: string;
  snippet: string;
  domain?: string | null;
  source?: string | null;
  score?: number | null;
  retrieved_at?: string;
}

export interface CritiqueIssue {
  issue_id: string;
  claim_id?: string | null;
  verification_status?: string | null;
  target: 'executor_a' | 'executor_b' | 'both';
  severity: 'low' | 'medium' | 'high';
  affected_claim: string;
  explanation: string;
  recommended_correction: string;
}

export interface DebateRound {
  round_number: number;
  executor_a_answer: string;
  executor_b_answer: string;
  critique: string;
  similarity_score?: number | null;
  converged: boolean;
  issues?: CritiqueIssue[];
  addressed_issues?: string[];
  executor_a_notes?: string | null;
  executor_b_notes?: string | null;
}

export interface AgentTraceEntry {
  agent: AgentRole;
  round_number: number;
  output: string;
  evidence: EvidenceItem[];
  unresolved_issues: string[];
  timestamp: string;
}

export interface QueryRequest {
  query: string;
  max_rounds?: number | null;
  llm_provider?: LLMProvider | null;
}

// ─── Claim Verification Types ─────────────────────────────────────────────────

export type ClaimStatus =
  | 'supported'
  | 'contradicted'
  | 'insufficient_evidence'
  | 'not_checked';

export type VerificationReportStatus =
  | 'completed'
  | 'partial'
  | 'failed'
  | 'unavailable'
  | 'no_claims';

export interface VerificationEvidenceRef {
  title: string;
  url: string;
  excerpt: string;
  relationship: string;
  retrieval_status?: string | null;
}

export interface ClaimVerificationResult {
  claim_id: string;
  claim_text: string;
  status: ClaimStatus;
  explanation: string;
  evidence: VerificationEvidenceRef[];
  limitations?: string | null;
}

export interface ClaimVerificationReport {
  report_status: VerificationReportStatus;
  results: ClaimVerificationResult[];
  summary: string;
  evidence_count: number;
  claims_checked: number;
  claims_not_checked: number;
}

// ─── API Response ─────────────────────────────────────────────────────────────

export interface QueryResponse {
  query: string;
  status: DebateStatus;
  rounds_completed: number;
  final_answer: string | null;
  subtasks: string[];
  debate_rounds: DebateRound[];
  trace: AgentTraceEntry[];
  sources: EvidenceItem[];
  verification_report?: ClaimVerificationReport | null;
  verdict_summary?: string | null;
  unresolved_uncertainties?: string[];
  error_message?: string | null;
}

export interface ErrorDetail {
  error: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  timestamp: string;
  llm_provider: string;
  llm_key_present: boolean;
}
