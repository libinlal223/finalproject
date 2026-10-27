import React, { useState } from 'react';
import type {
  ClaimVerificationReport,
  ClaimVerificationResult,
  ClaimStatus,
  VerificationReportStatus,
  VerificationEvidenceRef,
} from '../types/api';

interface ClaimVerificationPanelProps {
  report: ClaimVerificationReport | null | undefined;
}

// ─── Status helpers ───────────────────────────────────────────────────────────

function getStatusConfig(status: ClaimStatus): {
  label: string;
  className: string;
  icon: React.ReactNode;
} {
  switch (status) {
    case 'supported':
      return {
        label: 'Supported',
        className: 'cvp-badge cvp-badge--supported',
        icon: (
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        ),
      };
    case 'contradicted':
      return {
        label: 'Contradicted',
        className: 'cvp-badge cvp-badge--contradicted',
        icon: (
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ),
      };
    case 'insufficient_evidence':
      return {
        label: 'Inconclusive',
        className: 'cvp-badge cvp-badge--inconclusive',
        icon: (
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        ),
      };
    case 'not_checked':
    default:
      return {
        label: 'Not Checked',
        className: 'cvp-badge cvp-badge--not-checked',
        icon: (
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="12" cy="12" r="10" />
            <line x1="8" y1="12" x2="16" y2="12" />
          </svg>
        ),
      };
  }
}

function getReportStatusLabel(status: VerificationReportStatus): {
  label: string;
  description: string;
  className: string;
} {
  switch (status) {
    case 'completed':
      return {
        label: 'Verification Completed',
        description: 'All eligible claims were evaluated against retrieved evidence.',
        className: 'cvp-report-status cvp-report-status--completed',
      };
    case 'partial':
      return {
        label: 'Partial Verification',
        description: 'Some claims were evaluated; others could not be checked.',
        className: 'cvp-report-status cvp-report-status--partial',
      };
    case 'failed':
      return {
        label: 'Verification Failed',
        description: 'The verification service encountered an error. Results are unreliable.',
        className: 'cvp-report-status cvp-report-status--failed',
      };
    case 'unavailable':
      return {
        label: 'Verification Unavailable',
        description: 'Evidence retrieval was unavailable. Claims could not be checked.',
        className: 'cvp-report-status cvp-report-status--unavailable',
      };
    case 'no_claims':
    default:
      return {
        label: 'No Claims to Verify',
        description: 'No factual claims were extracted for verification in this query.',
        className: 'cvp-report-status cvp-report-status--no-claims',
      };
  }
}

function getRelationshipLabel(rel: string): string {
  switch (rel.toLowerCase()) {
    case 'supports': return 'Supports claim';
    case 'contradicts': return 'Contradicts claim';
    default: return 'Inconclusive';
  }
}

// ─── Evidence reference row ───────────────────────────────────────────────────

function EvidenceRefRow({ evidence }: { evidence: VerificationEvidenceRef }) {
  const [expanded, setExpanded] = useState(false);
  let host = 'external';
  try {
    host = new URL(evidence.url).hostname;
  } catch {
    /* ignore */
  }

  const relLabel = getRelationshipLabel(evidence.relationship);
  const relClass =
    evidence.relationship === 'supports'
      ? 'cvp-rel-tag cvp-rel-tag--supports'
      : evidence.relationship === 'contradicts'
      ? 'cvp-rel-tag cvp-rel-tag--contradicts'
      : 'cvp-rel-tag cvp-rel-tag--inconclusive';

  return (
    <div className="cvp-evidence-ref">
      <div className="cvp-evidence-ref-header">
        <span className={relClass}>{relLabel}</span>
        <a
          href={evidence.url}
          target="_blank"
          rel="noopener noreferrer"
          className="cvp-evidence-link"
          title={evidence.title}
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </svg>
          <span className="cvp-evidence-host">{host}</span>
          <span className="cvp-evidence-title">{evidence.title}</span>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
            <polyline points="15 3 21 3 21 9" />
            <line x1="10" y1="14" x2="21" y2="3" />
          </svg>
        </a>
        {evidence.excerpt && (
          <button
            type="button"
            className="cvp-expand-btn"
            onClick={() => setExpanded(e => !e)}
            aria-expanded={expanded}
            title={expanded ? 'Hide excerpt' : 'Show excerpt'}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
        )}
      </div>
      {expanded && evidence.excerpt && (
        <blockquote className="cvp-evidence-excerpt">{evidence.excerpt}</blockquote>
      )}
    </div>
  );
}

// ─── Individual claim card ────────────────────────────────────────────────────

function ClaimCard({ result }: { result: ClaimVerificationResult }) {
  const [open, setOpen] = useState(false);
  const { label, className, icon } = getStatusConfig(result.status);

  return (
    <div className={`cvp-claim-card cvp-claim-card--${result.status}`}>
      <div className="cvp-claim-header" onClick={() => setOpen(o => !o)} role="button" tabIndex={0}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') setOpen(o => !o); }}
        aria-expanded={open}
      >
        <div className="cvp-claim-left">
          <span className="cvp-claim-id">{result.claim_id}</span>
          <span className={className} aria-label={`Status: ${label}`}>
            {icon}
            {label}
          </span>
        </div>
        <p className="cvp-claim-text">{result.claim_text}</p>
        <svg
          className="cvp-chevron"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', flexShrink: 0 }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {open && (
        <div className="cvp-claim-body">
          <div className="cvp-claim-explanation">
            <h5 className="cvp-section-label">Verdict Explanation</h5>
            <p className="cvp-explanation-text">{result.explanation}</p>
          </div>

          {result.evidence && result.evidence.length > 0 && (
            <div className="cvp-claim-evidence">
              <h5 className="cvp-section-label">Evidence ({result.evidence.length})</h5>
              <div className="cvp-evidence-list">
                {result.evidence.map((evRef, idx) => (
                  <EvidenceRefRow key={idx} evidence={evRef} />
                ))}
              </div>
            </div>
          )}

          {result.limitations && (
            <div className="cvp-claim-limitations">
              <h5 className="cvp-section-label">Limitations &amp; Uncertainty</h5>
              <p className="cvp-limitations-text">{result.limitations}</p>
            </div>
          )}

          {result.status === 'not_checked' && (
            <div className="cvp-not-checked-notice">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              This claim was not checked. Not checked ≠ false.
            </div>
          )}

          {result.status === 'insufficient_evidence' && (
            <div className="cvp-inconclusive-notice">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              Insufficient evidence ≠ false. The system found no reliable sources to confirm or deny this claim.
            </div>
          )}

          {result.status === 'supported' && (
            <div className="cvp-supported-notice">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              Supported by retrieved evidence — does not guarantee absolute accuracy.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main panel ───────────────────────────────────────────────────────────────

export const ClaimVerificationPanel: React.FC<ClaimVerificationPanelProps> = ({ report }) => {
  if (!report) {
    return (
      <div className="cvp-root">
        <div className="cvp-header-row">
          <div className="cvp-title-group">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 11l3 3L22 4" />
              <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
            </svg>
            <h3 className="cvp-title">Claim Verification Report</h3>
          </div>
        </div>
        <div className="cvp-empty-notice">
          <p>Claim verification was not performed for this query (debate failed before retrieval).</p>
        </div>
      </div>
    );
  }

  const statusCfg = getReportStatusLabel(report.report_status);
  const totalClaims = report.results.length;

  const countByStatus = (s: ClaimStatus) => report.results.filter(r => r.status === s).length;
  const supported = countByStatus('supported');
  const contradicted = countByStatus('contradicted');
  const inconclusive = countByStatus('insufficient_evidence');
  const notChecked = countByStatus('not_checked');

  return (
    <div className="cvp-root">
      {/* Header */}
      <div className="cvp-header-row">
        <div className="cvp-title-group">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 11l3 3L22 4" />
            <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
          </svg>
          <h3 className="cvp-title">Claim Verification Report</h3>
        </div>
        <span className={statusCfg.className} title={statusCfg.description}>
          {statusCfg.label}
        </span>
      </div>

      {/* Summary bar */}
      <div className="cvp-summary-bar">
        <p className="cvp-summary-text">{report.summary}</p>
        <p className="cvp-summary-note">{statusCfg.description}</p>
      </div>

      {/* Stats row */}
      {totalClaims > 0 && (
        <div className="cvp-stats-row" role="list" aria-label="Verification statistics">
          <div className="cvp-stat" role="listitem">
            <span className="cvp-stat-count cvp-stat-count--total">{totalClaims}</span>
            <span className="cvp-stat-label">Total Claims</span>
          </div>
          {supported > 0 && (
            <div className="cvp-stat" role="listitem">
              <span className="cvp-stat-count cvp-stat-count--supported">{supported}</span>
              <span className="cvp-stat-label">Supported</span>
            </div>
          )}
          {contradicted > 0 && (
            <div className="cvp-stat" role="listitem">
              <span className="cvp-stat-count cvp-stat-count--contradicted">{contradicted}</span>
              <span className="cvp-stat-label">Contradicted</span>
            </div>
          )}
          {inconclusive > 0 && (
            <div className="cvp-stat" role="listitem">
              <span className="cvp-stat-count cvp-stat-count--inconclusive">{inconclusive}</span>
              <span className="cvp-stat-label">Inconclusive</span>
            </div>
          )}
          {notChecked > 0 && (
            <div className="cvp-stat" role="listitem">
              <span className="cvp-stat-count cvp-stat-count--not-checked">{notChecked}</span>
              <span className="cvp-stat-label">Not Checked</span>
            </div>
          )}
          {report.evidence_count > 0 && (
            <div className="cvp-stat" role="listitem">
              <span className="cvp-stat-count cvp-stat-count--evidence">{report.evidence_count}</span>
              <span className="cvp-stat-label">Evidence Items</span>
            </div>
          )}
        </div>
      )}

      {/* Disclaimer */}
      <div className="cvp-disclaimer">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <span>
          <strong>Supported</strong> means evidence was found — not a guarantee of truth.{' '}
          <strong>Inconclusive</strong> means evidence was insufficient — not proof of falsehood.
          VADER does not claim to eliminate hallucinations.
        </span>
      </div>

      {/* Claim cards */}
      {totalClaims > 0 ? (
        <div className="cvp-claims-list" aria-label="Claim verification results">
          {report.results.map(result => (
            <ClaimCard key={result.claim_id} result={result} />
          ))}
        </div>
      ) : (
        <div className="cvp-empty-notice">
          {report.report_status === 'no_claims' ? (
            <p>
              No factual claims were extracted from the executor candidates for this query.
              This can happen when the query is conceptual, mathematical, or when the LLM
              did not produce claims to verify.
            </p>
          ) : (
            <p>No claim results are available. See the report status above for details.</p>
          )}
        </div>
      )}
    </div>
  );
};
