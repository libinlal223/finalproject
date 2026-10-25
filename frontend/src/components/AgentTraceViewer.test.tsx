import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AgentTraceViewer } from './AgentTraceViewer';
import type { AgentTraceEntry } from '../types/api';

describe('AgentTraceViewer Component', () => {
  const mockTrace: AgentTraceEntry[] = [
    {
      agent: 'planner',
      round_number: 0,
      output: 'Core Question: Inflation causes. Subtasks: Monetary, Supply-side.',
      evidence: [],
      unresolved_issues: [],
      timestamp: '2026-10-16T08:00:00Z',
    },
    {
      agent: 'executor_a',
      round_number: 1,
      output: 'Executor A initial deductive derivation.',
      evidence: [],
      unresolved_issues: [],
      timestamp: '2026-10-16T08:00:05Z',
    },
    {
      agent: 'critic',
      round_number: 1,
      output: 'Critic evaluation noting contradictions in supply model.',
      evidence: [],
      unresolved_issues: ['Issue 1: Unverified monetary assertion.'],
      timestamp: '2026-10-16T08:00:10Z',
    },
  ];

  it('renders all trace entries with agent badges and timestamps', () => {
    render(<AgentTraceViewer trace={mockTrace} />);

    expect(screen.getByText(/Transparent Agent Execution Trace/i)).toBeInTheDocument();
    expect(screen.getByText(/Core Question: Inflation causes./i)).toBeInTheDocument();
    expect(screen.getByText(/Executor A initial deductive derivation./i)).toBeInTheDocument();
    expect(screen.getByText(/Critic evaluation noting contradictions/i)).toBeInTheDocument();
    expect(screen.getByText(/Issue 1: Unverified monetary assertion./i)).toBeInTheDocument();
  });

  it('filters trace entries when filter pills are clicked', () => {
    render(<AgentTraceViewer trace={mockTrace} />);

    const plannerFilter = screen.getByRole('button', { name: /^Planner$/i });
    fireEvent.click(plannerFilter);

    expect(screen.getByText(/Core Question: Inflation causes./i)).toBeInTheDocument();
    expect(screen.queryByText(/Executor A initial deductive derivation./i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Critic evaluation noting contradictions/i)).not.toBeInTheDocument();

    const allFilter = screen.getByRole('button', { name: /All Steps/i });
    fireEvent.click(allFilter);
    expect(screen.getByText(/Executor A initial deductive derivation./i)).toBeInTheDocument();
  });

  it('allows expanding long output text', () => {
    const longOutput = 'A'.repeat(350);
    const traceWithLongEntry: AgentTraceEntry[] = [
      {
        agent: 'executor_a',
        round_number: 1,
        output: longOutput,
        evidence: [],
        unresolved_issues: [],
        timestamp: '2026-10-16T08:00:00Z',
      },
    ];

    render(<AgentTraceViewer trace={traceWithLongEntry} />);

    const expandBtn = screen.getByRole('button', { name: /Show full output/i });
    expect(expandBtn).toBeInTheDocument();

    fireEvent.click(expandBtn);
    expect(screen.getByRole('button', { name: /Collapse output/i })).toBeInTheDocument();
  });
});
