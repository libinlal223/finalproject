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

export interface DebateRound {
  round_number: number;
  executor_a_answer: string;
  executor_b_answer: string;
  critique: string;
  similarity_score?: number | null;
  converged: boolean;
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

export interface QueryResponse {
  query: string;
  status: DebateStatus;
  rounds_completed: number;
  final_answer: string | null;
  subtasks: string[];
  debate_rounds: DebateRound[];
  trace: AgentTraceEntry[];
  sources: EvidenceItem[];
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
