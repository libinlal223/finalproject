import React from 'react';

interface ErrorMessageProps {
  message: string;
  code?: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  message,
  code,
  onRetry,
  onDismiss,
}) => {
  return (
    <div className="error-banner" role="alert">
      <div className="error-icon" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
      </div>

      <div className="error-body">
        <div className="error-title-row">
          <span className="error-heading">Debate Execution Error</span>
          {code && <span className="error-code-badge">{code}</span>}
        </div>
        <p className="error-message-text">{message}</p>
      </div>

      <div className="error-actions">
        {onRetry && (
          <button type="button" className="error-btn-retry" onClick={onRetry}>
            Retry
          </button>
        )}
        {onDismiss && (
          <button
            type="button"
            className="error-btn-dismiss"
            onClick={onDismiss}
            aria-label="Dismiss error"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};
