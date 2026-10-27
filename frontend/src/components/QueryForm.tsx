import React, { useState } from 'react';
import type { LLMProvider } from '../types/api';

const EXAMPLE_QUERIES = [
  'What are the primary macroeconomic mechanisms driving stagflation, and what policy trade-offs exist?',
  'Can 100% renewable energy grids maintain baseload stability without nuclear power or fossil fuels?',
  'Does quantum computing pose an immediate threat to modern RSA and ECC cryptographic standards?',
  'Is market competition alone sufficient to prevent monopolistic AI model dominance?',
];

interface QueryFormProps {
  onSubmit: (query: string, maxRounds: number, provider: LLMProvider | null) => void;
  isLoading: boolean;
}

export const QueryForm: React.FC<QueryFormProps> = ({ onSubmit, isLoading }) => {
  const [query, setQuery] = useState('');
  const [maxRounds, setMaxRounds] = useState<number>(3);
  const [provider, setProvider] = useState<LLMProvider | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const isSubmittingRef = React.useRef(false);

  React.useEffect(() => {
    isSubmittingRef.current = false;
  }, [isLoading]);

  const trimmed = query.trim();
  const isValidLength = trimmed.length >= 3 && trimmed.length <= 4000;
  const canSubmit = isValidLength && !isLoading;

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!canSubmit || isSubmittingRef.current) return;
    isSubmittingRef.current = true;
    onSubmit(trimmed, maxRounds, provider);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleExampleClick = (example: string) => {
    if (isLoading) return;
    setQuery(example);
  };

  return (
    <div className="query-card">
      <form onSubmit={handleSubmit} noValidate>
        <div className="query-header-row">
          <label htmlFor="vader-query-input" className="query-label">
            Research Question or Analytical Topic
          </label>
          <div className="query-tools-right">
            <button
              type="button"
              className={`settings-toggle-btn ${showSettings ? 'active' : ''}`}
              onClick={() => setShowSettings(!showSettings)}
              aria-expanded={showSettings}
              aria-controls="debate-settings-panel"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
              <span>{showSettings ? 'Hide Settings' : 'Debate Settings'}</span>
            </button>
          </div>
        </div>

        <div className="input-wrapper">
          <textarea
            id="vader-query-input"
            className="query-textarea"
            rows={3}
            placeholder="Type your complex analytical question here. The system will decompose it, assign divergent executors, retrieve grounded evidence, and conduct adversarial critique..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            maxLength={4000}
            required
            aria-invalid={query.length > 0 && !isValidLength}
            aria-describedby="query-char-count"
          />
        </div>

        {showSettings && (
          <div id="debate-settings-panel" className="settings-panel">
            <div className="setting-item">
              <div className="setting-label-row">
                <label htmlFor="rounds-slider" className="setting-title">
                  Max Debate Rounds: <strong>{maxRounds}</strong>
                </label>
                <span className="setting-hint">Server hard limit: 10</span>
              </div>
              <div className="slider-row">
                <input
                  id="rounds-slider"
                  type="range"
                  min={1}
                  max={5}
                  value={maxRounds}
                  onChange={(e) => setMaxRounds(Number(e.target.value))}
                  disabled={isLoading}
                  className="rounds-slider"
                />
                <div className="slider-ticks">
                  <span>1 (Rapid)</span>
                  <span>2</span>
                  <span>3 (Default)</span>
                  <span>4</span>
                  <span>5 (In-depth)</span>
                </div>
              </div>
            </div>

            <div className="setting-item">
              <label htmlFor="provider-select" className="setting-title">
                LLM Provider Override
              </label>
              <select
                id="provider-select"
                className="provider-select"
                value={provider || ''}
                onChange={(e) => setProvider((e.target.value as LLMProvider) || null)}
                disabled={isLoading}
              >
                <option value="">Default (Configured on Server)</option>
                <option value="mock">Offline Mock Adapter (Deterministic)</option>
                <option value="groq">Groq API (Llama 3)</option>
                <option value="gemini">Google Gemini API (Gemini Flash)</option>
              </select>
            </div>
          </div>
        )}

        <div className="query-footer-row">
          <div className="char-count" id="query-char-count">
            {query.length > 0 && trimmed.length < 3 ? (
              <span className="text-warning">Minimum 3 characters required</span>
            ) : (
              <span>{query.length} / 4000 characters</span>
            )}
          </div>

          <div className="actions-right">
            {query && !isLoading && (
              <button
                type="button"
                className="btn-clear"
                onClick={() => setQuery('')}
                title="Clear input"
              >
                Clear
              </button>
            )}
            <button
              type="submit"
              className="btn-submit"
              disabled={!canSubmit}
              title={canSubmit ? 'Submit query (Ctrl+Enter)' : 'Enter at least 3 characters'}
            >
              {isLoading ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  <span>Debating...</span>
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                  <span>Start Multi-Agent Debate</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {!isLoading && (
        <div className="example-queries-section">
          <span className="example-label">Or explore a benchmark analytical question:</span>
          <div className="example-pills">
            {EXAMPLE_QUERIES.map((example, idx) => (
              <button
                key={idx}
                type="button"
                className="example-pill"
                onClick={() => handleExampleClick(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
