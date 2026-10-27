import React, { useState } from 'react';
import type { DebateStatus } from '../types/api';

interface FinalAnswerCardProps {
  finalAnswer: string | null;
  status: DebateStatus;
  roundsCompleted: number;
  subtasks: string[];
  verdictSummary?: string | null;
  unresolvedUncertainties?: string[];
  onReset: () => void;
}

export const FinalAnswerCard: React.FC<FinalAnswerCardProps> = ({
  finalAnswer,
  status,
  roundsCompleted,
  subtasks,
  verdictSummary,
  unresolvedUncertainties,
  onReset,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!finalAnswer) return;
    try {
      await navigator.clipboard.writeText(finalAnswer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  const getStatusBadge = (s: DebateStatus) => {
    switch (s) {
      case 'completed':
        return <span className="status-badge badge-success">Adjudicated by Judge</span>;
      case 'converged':
        return <span className="status-badge badge-converged">Debate Converged</span>;
      case 'max_rounds':
        return <span className="status-badge badge-warning">Max Rounds Reached</span>;
      case 'error':
        return <span className="status-badge badge-error">Halted on Error</span>;
      default:
        return <span className="status-badge badge-neutral">{s}</span>;
    }
  };

  return (
    <div className="final-answer-card">
      <div className="card-top-bar">
        <div className="card-heading-group">
          <div className="judge-badge-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
              <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
          </div>
          <div>
            <h2 className="card-main-title">
              {status === 'error' ? 'Candidate Output (Unadjudicated Fallback)' : 'Final Synthesis & Adjudication'}
            </h2>
            <div className="badge-row">
              {getStatusBadge(status)}
              <span className="round-badge">
                {roundsCompleted} {roundsCompleted === 1 ? 'Round' : 'Rounds'} Completed
              </span>
            </div>
          </div>
        </div>

        <div className="card-actions-row">
          <button
            type="button"
            className="action-btn"
            onClick={handleCopy}
            disabled={!finalAnswer}
            title="Copy answer to clipboard"
          >
            {copied ? (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span>Copied</span>
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                <span>Copy</span>
              </>
            )}
          </button>
          <button
            type="button"
            className="action-btn action-btn-primary"
            onClick={onReset}
            title="Submit another analytical query"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>New Debate</span>
          </button>
        </div>
      </div>

      <div className="final-answer-body">
        {finalAnswer ? (
          <div className="answer-text-content">
            {finalAnswer.startsWith('[UNADJUDICATED FALLBACK CANDIDATE') && (
              <div
                className="unadjudicated-fallback-warning"
                data-testid="unadjudicated-warning"
                style={{
                  padding: '12px 16px',
                  marginBottom: '16px',
                  borderRadius: '8px',
                  background: 'rgba(239, 68, 68, 0.12)',
                  border: '1px solid rgba(239, 68, 68, 0.35)',
                  color: '#FCA5A5',
                  fontSize: '0.875rem',
                  lineHeight: '1.4',
                }}
              >
                <strong>Unadjudicated Fallback Notice:</strong> Debate stopped before adjudication.
                The following candidate output has NOT been evaluated, verified, or synthesized by the Judge.
              </div>
            )}
            {finalAnswer.split('\n\n').map((paragraph, idx) => (
              <p key={idx} className="answer-paragraph">
                {paragraph}
              </p>
            ))}
          </div>
        ) : (
          <div className="no-answer-notice">
            No final synthesis was produced. Please review the trace logs for details.
          </div>
        )}
      </div>

      {verdictSummary && (
        <div className="verdict-summary-callout" data-testid="verdict-summary">
          <div className="verdict-summary-header">
            <span className="verdict-summary-label">
              {status === 'completed' || status === 'converged'
                ? 'Judge Adjudication Rationale'
                : 'Adjudication Fallback & Status Rationale'}
            </span>
          </div>
          <p className="verdict-summary-text">{verdictSummary}</p>
        </div>
      )}

      {unresolvedUncertainties && unresolvedUncertainties.length > 0 && (
        <div className="unresolved-uncertainties-section" data-testid="unresolved-uncertainties">
          <h4 className="uncertainties-title">Epistemic Uncertainty & Unresolved Boundaries</h4>
          <ul className="uncertainties-list">
            {unresolvedUncertainties.map((item, idx) => (
              <li key={idx} className="uncertainty-item">
                <span className="uncertainty-bullet" />
                <span className="uncertainty-text">{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {subtasks && subtasks.length > 0 && (
        <div className="subtasks-section">
          <h4 className="subtasks-title">Planner Analytical Decomposition</h4>
          <ul className="subtasks-list">
            {subtasks.map((subtask, index) => (
              <li key={index} className="subtask-item">
                <span className="subtask-num">{index + 1}</span>
                <span className="subtask-text">{subtask}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
