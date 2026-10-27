import React, { useState } from 'react';
import { Header } from './components/Header';
import { QueryForm } from './components/QueryForm';
import { LoadingIndicator } from './components/LoadingIndicator';
import { FinalAnswerCard } from './components/FinalAnswerCard';
import { DebateRoundsViewer } from './components/DebateRoundsViewer';
import { AgentTraceViewer } from './components/AgentTraceViewer';
import { EvidenceSources } from './components/EvidenceSources';
import { ClaimVerificationPanel } from './components/ClaimVerificationPanel';
import { ErrorMessage } from './components/ErrorMessage';
import { submitQuery, ApiError } from './services/api';
import type { LLMProvider, QueryResponse } from './types/api';
import './App.css';

type TabType = 'overview' | 'verification' | 'rounds' | 'trace' | 'evidence';

export const App: React.FC = () => {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<{ message: string; code?: string } | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('overview');

  // Track last query for retry functionality
  const [lastSubmitted, setLastSubmitted] = useState<{
    query: string;
    maxRounds: number;
    provider: LLMProvider | null;
  } | null>(null);

  const handleQuerySubmit = async (
    query: string,
    maxRounds: number,
    provider: LLMProvider | null,
  ) => {
    if (isLoading) return;
    setIsLoading(true);
    setError(null);
    setLastSubmitted({ query, maxRounds, provider });

    try {
      const data = await submitQuery({
        query,
        max_rounds: maxRounds,
        llm_provider: provider || undefined,
      });
      setResponse(data);
      setActiveTab('overview');
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError({ message: err.message, code: err.code });
      } else if (err instanceof Error) {
        setError({ message: err.message, code: 'unknown_error' });
      } else {
        setError({ message: 'An unexpected error occurred during the debate.', code: 'unknown' });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    if (lastSubmitted) {
      handleQuerySubmit(lastSubmitted.query, lastSubmitted.maxRounds, lastSubmitted.provider);
    }
  };

  const handleReset = () => {
    setResponse(null);
    setError(null);
    setActiveTab('overview');
  };

  return (
    <div className="vader-app">
      <Header />

      <main className="main-content-wrapper">
        {error && (
          <ErrorMessage
            message={error.message}
            code={error.code}
            onRetry={lastSubmitted ? handleRetry : undefined}
            onDismiss={() => setError(null)}
          />
        )}

        {!response && !isLoading && (
          <QueryForm onSubmit={handleQuerySubmit} isLoading={isLoading} />
        )}

        {isLoading && <LoadingIndicator />}

        {response && !isLoading && (
          <div className="results-container">
            {/* Active Query Banner */}
            <div className="active-query-banner">
              <div className="query-banner-left">
                <span className="query-banner-tag">Research Question</span>
                <p className="query-banner-text">{response.query}</p>
              </div>
              <button
                type="button"
                className="btn-new-query"
                onClick={handleReset}
                title="Ask a different question"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="12" y1="5" x2="12" y2="19"></line>
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
                <span>New Question</span>
              </button>
            </div>

            {/* Navigation Tabs */}
            <div className="navigation-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'overview'}
                className={`nav-tab ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                <span>Final Synthesis</span>
              </button>

              <button
                type="button"
                role="tab"
                id="tab-verification"
                aria-selected={activeTab === 'verification'}
                className={`nav-tab ${activeTab === 'verification' ? 'active' : ''}`}
                onClick={() => setActiveTab('verification')}
              >
                <span>Claim Verification</span>
                {response.verification_report && (
                  <span className="tab-count-pill">
                    {response.verification_report.results.length}
                  </span>
                )}
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'rounds'}
                className={`nav-tab ${activeTab === 'rounds' ? 'active' : ''}`}
                onClick={() => setActiveTab('rounds')}
              >
                <span>Debate Rounds</span>
                <span className="tab-count-pill">{response.debate_rounds.length}</span>
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'trace'}
                className={`nav-tab ${activeTab === 'trace' ? 'active' : ''}`}
                onClick={() => setActiveTab('trace')}
              >
                <span>Agent Trace</span>
                <span className="tab-count-pill">{response.trace.length}</span>
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'evidence'}
                className={`nav-tab ${activeTab === 'evidence' ? 'active' : ''}`}
                onClick={() => setActiveTab('evidence')}
              >
                <span>Grounded Evidence</span>
                <span className="tab-count-pill">{response.sources.length}</span>
              </button>
            </div>

            {/* Tab Views */}
            <div className="tab-content">
              {activeTab === 'overview' && (
                <>
                  <FinalAnswerCard
                    finalAnswer={response.final_answer}
                    status={response.status}
                    roundsCompleted={response.rounds_completed}
                    subtasks={response.subtasks}
                    verdictSummary={response.verdict_summary}
                    unresolvedUncertainties={response.unresolved_uncertainties}
                    onReset={handleReset}
                  />

                  {/* Summary preview of debate rounds, verification, and evidence */}
                  <div className="overview-subsections">
                    <ClaimVerificationPanel report={response.verification_report} />
                    <DebateRoundsViewer rounds={response.debate_rounds} />
                    <EvidenceSources sources={response.sources} />
                  </div>
                </>
              )}

              {activeTab === 'verification' && (
                <ClaimVerificationPanel report={response.verification_report} />
              )}

              {activeTab === 'rounds' && (
                <DebateRoundsViewer rounds={response.debate_rounds} />
              )}

              {activeTab === 'trace' && (
                <AgentTraceViewer trace={response.trace} />
              )}

              {activeTab === 'evidence' && (
                <EvidenceSources sources={response.sources} />
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="footer-container">
        <div className="footer-content">
          <div className="footer-meta">
            <span>VADER — Hybrid Multi-Agent AI System</span>
            <span>Final-Year CSE Project</span>
          </div>
          <p className="footer-disclaimer">
            Disclaimer: Multiple agents and adversarial critique do not guarantee factual infallibility.
            VADER is an AI research demonstration of adversarial collaborative reasoning with verifiable evidence grounding.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
