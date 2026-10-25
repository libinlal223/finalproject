import React, { useState } from 'react';
import type { AgentRole, AgentTraceEntry } from '../types/api';

interface AgentTraceViewerProps {
  trace: AgentTraceEntry[];
}

export const AgentTraceViewer: React.FC<AgentTraceViewerProps> = ({ trace }) => {
  const [selectedAgent, setSelectedAgent] = useState<AgentRole | 'all'>('all');
  const [expandedIndices, setExpandedIndices] = useState<Record<number, boolean>>({});

  if (!trace || trace.length === 0) {
    return null;
  }

  const toggleExpand = (idx: number) => {
    setExpandedIndices((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const filteredTrace = selectedAgent === 'all'
    ? trace
    : trace.filter((entry) => entry.agent === selectedAgent);

  const getAgentRoleClass = (agent: AgentRole) => {
    switch (agent) {
      case 'planner':
        return 'agent-planner';
      case 'executor_a':
        return 'agent-exec-a';
      case 'executor_b':
        return 'agent-exec-b';
      case 'critic':
        return 'agent-critic';
      case 'judge':
        return 'agent-judge';
      default:
        return 'agent-default';
    }
  };

  const getAgentDisplayName = (agent: AgentRole) => {
    switch (agent) {
      case 'planner':
        return 'Planner';
      case 'executor_a':
        return 'Executor A';
      case 'executor_b':
        return 'Executor B';
      case 'critic':
        return 'Critic';
      case 'judge':
        return 'Judge';
      default:
        return agent;
    }
  };

  return (
    <div className="trace-section">
      <div className="section-header-row">
        <div>
          <h3 className="section-title">Transparent Agent Execution Trace</h3>
          <p className="section-subtitle">
            Verifiable chronological log of public agent reasoning steps, critiques, and decisions.
          </p>
        </div>

        <div className="trace-filter-pills" role="group" aria-label="Filter by agent">
          {(['all', 'planner', 'executor_a', 'executor_b', 'critic', 'judge'] as const).map((agent) => (
            <button
              key={agent}
              type="button"
              className={`filter-pill ${selectedAgent === agent ? 'active' : ''}`}
              onClick={() => setSelectedAgent(agent)}
            >
              {agent === 'all' ? 'All Steps' : getAgentDisplayName(agent)}
            </button>
          ))}
        </div>
      </div>

      <div className="trace-timeline">
        {filteredTrace.map((entry, idx) => {
          const isExpanded = !!expandedIndices[idx];
          const isLongOutput = entry.output.length > 280;

          return (
            <div key={idx} className={`trace-item ${getAgentRoleClass(entry.agent)}`}>
              <div className="trace-timeline-marker">
                <div className="marker-dot" />
                <div className="marker-line" />
              </div>

              <div className="trace-card">
                <div className="trace-card-header">
                  <div className="trace-agent-info">
                    <span className={`trace-role-badge ${getAgentRoleClass(entry.agent)}`}>
                      {getAgentDisplayName(entry.agent)}
                    </span>
                    <span className="trace-round-badge">
                      {entry.round_number === 0 ? 'Plan Phase' : `Round ${entry.round_number}`}
                    </span>
                  </div>
                  <div className="trace-timestamp">
                    {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </div>
                </div>

                <div className="trace-card-body">
                  <div className="trace-output-text">
                    {isLongOutput && !isExpanded ? (
                      <>
                        {entry.output.slice(0, 280)}...
                        <button
                          type="button"
                          className="btn-toggle-expand"
                          onClick={() => toggleExpand(idx)}
                        >
                          Show full output
                        </button>
                      </>
                    ) : (
                      <>
                        {entry.output}
                        {isLongOutput && (
                          <button
                            type="button"
                            className="btn-toggle-expand"
                            onClick={() => toggleExpand(idx)}
                          >
                            Collapse output
                          </button>
                        )}
                      </>
                    )}
                  </div>

                  {entry.unresolved_issues && entry.unresolved_issues.length > 0 && (
                    <div className="unresolved-issues-block">
                      <span className="unresolved-issues-title">Unresolved Critique Findings:</span>
                      <ul className="unresolved-list">
                        {entry.unresolved_issues.map((issue, issueIdx) => (
                          <li key={issueIdx}>{issue}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
