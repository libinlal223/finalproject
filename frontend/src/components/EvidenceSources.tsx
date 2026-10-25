import React from 'react';
import type { EvidenceItem } from '../types/api';

interface EvidenceSourcesProps {
  sources: EvidenceItem[];
}

export const EvidenceSources: React.FC<EvidenceSourcesProps> = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return (
      <div className="evidence-section">
        <h3 className="section-title">Grounded Evidence & Citations</h3>
        <p className="section-subtitle">
          Retrieved empirical sources audited during claim verification.
        </p>
        <div className="empty-sources-notice">
          No external web evidence was retrieved or cited for this query. The synthesis was resolved using foundational axioms and comparative debate.
        </div>
      </div>
    );
  }

  return (
    <div className="evidence-section">
      <div className="section-header-row">
        <div>
          <h3 className="section-title">Grounded Evidence & Citations</h3>
          <p className="section-subtitle">
            External empirical documents retrieved, normalized, and verified by the system.
          </p>
        </div>
        <span className="badge badge-accent">
          {sources.length} Verified {sources.length === 1 ? 'Source' : 'Sources'}
        </span>
      </div>

      <div className="sources-grid">
        {sources.map((item, idx) => {
          let host = item.domain;
          if (!host && item.url) {
            try {
              host = new URL(item.url).hostname;
            } catch {
              host = 'external';
            }
          }

          return (
            <div key={idx} className="source-card">
              <div className="source-card-header">
                <div className="source-domain-tag">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="2" y1="12" x2="22" y2="12"></line>
                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                  </svg>
                  <span>{host}</span>
                </div>
                {item.source && (
                  <span className="source-provider-tag">{item.source}</span>
                )}
              </div>

              <h4 className="source-title">
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="source-link"
                  title={item.title}
                >
                  {item.title}
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                  </svg>
                </a>
              </h4>

              <div className="source-snippet">
                {item.snippet}
              </div>

              <div className="source-card-footer">
                <span className="source-url-text" title={item.url}>
                  {item.url}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
