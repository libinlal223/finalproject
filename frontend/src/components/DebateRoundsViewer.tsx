import React, { useState } from 'react';
import type { DebateRound } from '../types/api';

interface DebateRoundsViewerProps {
  rounds: DebateRound[];
}

export const DebateRoundsViewer: React.FC<DebateRoundsViewerProps> = ({ rounds }) => {
  const [selectedRoundIndex, setSelectedRoundIndex] = useState(0);

  if (!rounds || rounds.length === 0) {
    return null;
  }

  const currentRound = rounds[selectedRoundIndex] || rounds[0];
  const simPercent = currentRound.similarity_score !== null && currentRound.similarity_score !== undefined
    ? Math.round(currentRound.similarity_score * 100)
    : null;

  return (
    <div className="debate-rounds-section">
      <div className="section-header-row">
        <div>
          <h3 className="section-title">Adversarial Debate Rounds</h3>
          <p className="section-subtitle">
            Compare divergent candidate answers, adversarial challenge points, and convergence progression.
          </p>
        </div>

        {rounds.length > 1 && (
          <div className="round-tabs" role="tablist" aria-label="Debate Rounds">
            {rounds.map((r, idx) => (
              <button
                key={r.round_number}
                type="button"
                role="tab"
                aria-selected={idx === selectedRoundIndex}
                className={`round-tab-btn ${idx === selectedRoundIndex ? 'active' : ''}`}
                onClick={() => setSelectedRoundIndex(idx)}
              >
                Round {r.round_number}
                {r.converged && <span className="tab-converged-dot" title="Converged" />}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="round-content-container">
        {/* Metric banner */}
        <div className="round-metrics-bar">
          <div className="metric-item">
            <span className="metric-label">Active Round:</span>
            <span className="metric-val">Round {currentRound.round_number} of {rounds.length}</span>
          </div>

          {simPercent !== null && (
            <div className="metric-item">
              <span className="metric-label">Semantic Similarity Signal:</span>
              <div className="similarity-meter-wrap">
                <div className="similarity-bar-bg">
                  <div
                    className="similarity-bar-fill"
                    style={{ width: `${simPercent}%` }}
                  />
                </div>
                <span className="metric-val">{simPercent}%</span>
              </div>
              <span className="similarity-disclaimer">
                (Observational alignment signal; not proof of factual correctness)
              </span>
            </div>
          )}

          <div className="metric-item">
            <span className="metric-label">Round Status:</span>
            {currentRound.converged ? (
              <span className="badge badge-converged">Converged</span>
            ) : (
              <span className="badge badge-amber">Material Issues Present</span>
            )}
          </div>
        </div>

        {/* Executor Comparison Grid */}
        <div className="executors-grid">
          {/* Executor A */}
          <div className="executor-card card-executor-a">
            <div className="executor-card-header">
              <div className="executor-title-row">
                <span className="executor-role-badge badge-role-a">Executor A</span>
                <span className="strategy-tag">Deductive & First-Principles</span>
              </div>
              <p className="executor-desc">Analyzes foundational axioms, definitions, and causal mechanisms.</p>
            </div>
            <div className="executor-card-body">
              <div className="answer-prose">
                {currentRound.executor_a_answer}
              </div>
              {currentRound.executor_a_notes && (
                <div className="executor-revision-notes" data-testid="executor-a-notes">
                  <span className="revision-notes-label">Revision Strategy:</span>{' '}
                  <span className="revision-notes-text">{currentRound.executor_a_notes}</span>
                </div>
              )}
            </div>
          </div>

          {/* Executor B */}
          <div className="executor-card card-executor-b">
            <div className="executor-card-header">
              <div className="executor-title-row">
                <span className="executor-role-badge badge-role-b">Executor B</span>
                <span className="strategy-tag">Empirical & Practical Trade-offs</span>
              </div>
              <p className="executor-desc">Analyzes boundary conditions, real-world friction, and edge cases.</p>
            </div>
            <div className="executor-card-body">
              <div className="answer-prose">
                {currentRound.executor_b_answer}
              </div>
              {currentRound.executor_b_notes && (
                <div className="executor-revision-notes" data-testid="executor-b-notes">
                  <span className="revision-notes-label">Revision Strategy:</span>{' '}
                  <span className="revision-notes-text">{currentRound.executor_b_notes}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Critic Evaluation for this round */}
        <div className="critic-evaluation-card">
          <div className="critic-card-header">
            <div className="critic-title-row">
              <span className="executor-role-badge badge-role-critic">Critic Agent Audit</span>
              <span className="critic-subtitle">Adversarial evaluation & contradiction detection</span>
            </div>
            {currentRound.addressed_issues && currentRound.addressed_issues.length > 0 && (
              <div className="addressed-issues-container" data-testid="addressed-issues-list">
                <span className="addressed-issues-label">Addressed in Revisions:</span>
                <div className="addressed-pills">
                  {currentRound.addressed_issues.map((issId) => (
                    <span key={issId} className="addressed-pill">
                      ✓ {issId}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
          <div className="critic-card-body">
            <p className="critic-text">{currentRound.critique}</p>

            {/* Structured findings if present */}
            {currentRound.issues && currentRound.issues.length > 0 && (
              <div className="critic-issues-list" data-testid="critic-issues-list">
                <h5 className="issues-list-heading">Specific Actionable Findings:</h5>
                <div className="issues-grid">
                  {currentRound.issues.map((issue) => (
                    <div
                      key={issue.issue_id}
                      className={`issue-item-card severity-${issue.severity}`}
                      data-testid={`critic-issue-${issue.issue_id}`}
                    >
                      <div className="issue-item-header">
                        <div className="issue-tags">
                          <span className={`issue-severity-badge badge-${issue.severity}`}>
                            {issue.severity.toUpperCase()}
                          </span>
                          <span className="issue-id-tag">{issue.issue_id}</span>
                          {issue.claim_id && (
                            <span className="issue-claim-link" title="Linked to Verified Claim">
                              🔗 {issue.claim_id}
                            </span>
                          )}
                          <span className="issue-target-tag">Target: {issue.target}</span>
                        </div>
                      </div>
                      <div className="issue-claim-name">
                        <strong>Affected:</strong> {issue.affected_claim}
                      </div>
                      <div className="issue-explanation">{issue.explanation}</div>
                      {issue.recommended_correction && (
                        <div className="issue-recommendation">
                          <strong>Correction:</strong> {issue.recommended_correction}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
