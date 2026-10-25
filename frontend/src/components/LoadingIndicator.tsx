import React, { useEffect, useState } from 'react';

const PIPELINE_STAGES = [
  { id: 'planner', name: 'Planner Agent', desc: 'Decomposing problem into constraints & subtasks' },
  { id: 'executors', name: 'Executors A & B', desc: 'Formulating independent deductive & empirical answers' },
  { id: 'retrieval', name: 'Evidence Retrieval', desc: 'Fetching external evidence & normalising source URLs' },
  { id: 'verifier', name: 'Claim Verification', desc: 'Auditing factual assertions against retrieved evidence' },
  { id: 'critic', name: 'Critic Agent', desc: 'Challenging contradictions, omissions & logical flaws' },
  { id: 'judge', name: 'Judge Agent', desc: 'Synthesising balanced resolution and limitations' },
];

export const LoadingIndicator: React.FC = () => {
  const [activeStageIndex, setActiveStageIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStageIndex((prev) => (prev + 1) % PIPELINE_STAGES.length);
    }, 2400);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="loading-card" role="status" aria-live="polite">
      <div className="loading-header">
        <div className="pulsing-spinner" aria-hidden="true">
          <div className="inner-ring" />
          <div className="outer-ring" />
        </div>
        <div className="loading-header-text">
          <h3 className="loading-title">Orchestrating Adversarial Collaborative Debate</h3>
          <p className="loading-subtitle">
            Coordinating independent agents, verifiable claims, and bounded revisions...
          </p>
        </div>
      </div>

      <div className="pipeline-steps">
        {PIPELINE_STAGES.map((stage, idx) => {
          const isCurrent = idx === activeStageIndex;
          const isPassed = idx < activeStageIndex;

          return (
            <div
              key={stage.id}
              className={`pipeline-step ${isCurrent ? 'active' : ''} ${isPassed ? 'completed' : ''}`}
            >
              <div className="step-icon-col">
                <div className="step-bullet">
                  {isPassed ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                  ) : (
                    <span>{idx + 1}</span>
                  )}
                </div>
                {idx < PIPELINE_STAGES.length - 1 && <div className="step-line" />}
              </div>
              <div className="step-body">
                <div className="step-name">{stage.name}</div>
                <div className="step-desc">{stage.desc}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
