import React, { useEffect, useState } from 'react';
import { checkHealth } from '../services/api';
import type { HealthResponse } from '../types/api';

export const Header: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;
    checkHealth()
      .then((data) => {
        if (!ignore) {
          setHealth(data);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          setHealth(null);
          setError(err instanceof Error ? err.message : 'Backend offline');
        }
      });

    return () => {
      ignore = true;
    };
  }, []);

  const handleRefresh = async () => {
    setIsChecking(true);
    setError(null);
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch (err: unknown) {
      setHealth(null);
      setError(err instanceof Error ? err.message : 'Backend offline');
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <header className="header-container">
      <div className="header-content">
        <div className="brand-section">
          <div className="brand-badge-row">
            <span className="badge badge-accent">AI Research System</span>
            <span className="badge badge-outline">Final-Year CSE Project</span>
          </div>
          <div className="logo-row">
            <div className="logo-icon" aria-hidden="true">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                <polyline points="2 17 12 22 22 17"></polyline>
                <polyline points="2 12 12 17 22 12"></polyline>
              </svg>
            </div>
            <h1 className="brand-title">VADER</h1>
          </div>
          <p className="brand-subtitle">
            Hybrid Multi-Agent AI System with <strong>Adversarial Collaborative Reasoning</strong>
          </p>
        </div>

        <div className="header-status-section">
          <div className="status-pill-container" role="status">
            <span className="status-dot-wrapper">
              <span className={`status-dot ${health?.status === 'ok' ? 'status-online' : 'status-offline'}`} />
            </span>
            <div className="status-text-block">
              <span className="status-label">
                {health?.status === 'ok' ? 'Backend Connected' : error ? 'Backend Offline' : 'Connecting...'}
              </span>
              {health && (
                <span className="status-meta">
                  Provider: <strong>{health.llm_provider.toUpperCase()}</strong> | v{health.version}
                </span>
              )}
            </div>
            <button
              type="button"
              className="refresh-btn"
              onClick={handleRefresh}
              disabled={isChecking}
              title="Refresh backend status"
              aria-label="Refresh backend status"
            >
              <svg className={isChecking ? 'spin' : ''} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
