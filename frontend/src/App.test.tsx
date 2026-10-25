import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';
import * as api from './services/api';
import type { QueryResponse } from './types/api';

vi.mock('./services/api', () => ({
  submitQuery: vi.fn(),
  checkHealth: vi.fn().mockResolvedValue({
    status: 'ok',
    version: '0.1.0',
    environment: 'test',
    timestamp: '2026-10-16T08:00:00Z',
    llm_provider: 'mock',
    llm_key_present: true,
  }),
  ApiError: class ApiError extends Error {
    public status: number;
    public code: string;
    constructor(message: string, status: number, code: string = 'api_error') {
      super(message);
      this.name = 'ApiError';
      this.status = status;
      this.code = code;
    }
  },
  getApiBaseUrl: vi.fn().mockReturnValue('http://localhost:8000'),
}));

describe('App Component Integration', () => {
  const mockSuccessResponse: QueryResponse = {
    query: 'What causes macroeconomic inflation?',
    status: 'completed',
    rounds_completed: 2,
    final_answer: 'Adjudicated answer resolving theoretical dispute with empirical caveats.',
    subtasks: ['Subtask 1: Define monetary velocity', 'Subtask 2: Analyze supply shocks'],
    debate_rounds: [
      {
        round_number: 1,
        executor_a_answer: 'Exec A deductive argument.',
        executor_b_answer: 'Exec B empirical argument.',
        critique: 'Critic Round 1 findings.',
        similarity_score: 0.55,
        converged: false,
      },
    ],
    trace: [
      {
        agent: 'planner',
        round_number: 0,
        output: 'Planner subtasks generated.',
        evidence: [],
        unresolved_issues: [],
        timestamp: '2026-10-16T08:00:00Z',
      },
    ],
    sources: [
      {
        title: 'Central Bank Review 2026',
        url: 'https://example.org/review',
        snippet: 'Inflation pressures moderated.',
        domain: 'example.org',
      },
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders branding header, backend health indicator, and initial query form', async () => {
    render(<App />);

    expect(screen.getByRole('heading', { level: 1, name: /VADER/i })).toBeInTheDocument();
    expect(screen.getAllByText(/Adversarial Collaborative Reasoning/i).length).toBeGreaterThan(0);
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Start Multi-Agent Debate/i })).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Backend Connected/i)).toBeInTheDocument();
    });
  });

  it('submits query and displays loading indicator, followed by final answer', async () => {
    vi.mocked(api.submitQuery).mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve(mockSuccessResponse), 50);
        }),
    );

    render(<App />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'What causes macroeconomic inflation?' } });

    const submitBtn = screen.getByRole('button', { name: /Start Multi-Agent Debate/i });
    fireEvent.click(submitBtn);

    // Should display loading indicator
    expect(
      screen.getByText(/Orchestrating Adversarial Collaborative Debate/i),
    ).toBeInTheDocument();

    // Await response rendering
    await waitFor(() => {
      expect(screen.getByText(/Final Synthesis & Adjudication/i)).toBeInTheDocument();
      expect(
        screen.getByText(/Adjudicated answer resolving theoretical dispute with empirical caveats./i),
      ).toBeInTheDocument();
      expect(screen.getByText(/2 Rounds Completed/i)).toBeInTheDocument();
    });
  });

  it('switches between navigation tabs (Rounds, Trace, Evidence)', async () => {
    vi.mocked(api.submitQuery).mockResolvedValue(mockSuccessResponse);

    render(<App />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'What causes macroeconomic inflation?' } });
    fireEvent.click(screen.getByRole('button', { name: /Start Multi-Agent Debate/i }));

    await waitFor(() => {
      expect(screen.getByText(/Final Synthesis & Adjudication/i)).toBeInTheDocument();
    });

    // Switch to Trace tab
    const traceTab = screen.getByRole('tab', { name: /Agent Trace/i });
    fireEvent.click(traceTab);
    expect(screen.getByText(/Transparent Agent Execution Trace/i)).toBeInTheDocument();
    expect(screen.getByText(/Planner subtasks generated./i)).toBeInTheDocument();

    // Switch to Evidence tab
    const evidenceTab = screen.getByRole('tab', { name: /Grounded Evidence/i });
    fireEvent.click(evidenceTab);
    expect(screen.getByText(/Grounded Evidence & Citations/i)).toBeInTheDocument();
    expect(screen.getByText(/Central Bank Review 2026/i)).toBeInTheDocument();
  });

  it('displays ErrorMessage when submitQuery rejects, and allows retry', async () => {
    vi.mocked(api.submitQuery).mockRejectedValueOnce(
      new api.ApiError('Connection refused to backend', 0, 'network_error'),
    );

    render(<App />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Query leading to error' } });
    fireEvent.click(screen.getByRole('button', { name: /Start Multi-Agent Debate/i }));

    await waitFor(() => {
      expect(screen.getByText(/Debate Execution Error/i)).toBeInTheDocument();
      expect(screen.getByText(/Connection refused to backend/i)).toBeInTheDocument();
      expect(screen.getByText(/network_error/i)).toBeInTheDocument();
    });

    // Test retry
    vi.mocked(api.submitQuery).mockResolvedValueOnce(mockSuccessResponse);
    const retryBtn = screen.getByRole('button', { name: /Retry/i });
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(screen.getByText(/Final Synthesis & Adjudication/i)).toBeInTheDocument();
    });
  });

  it('allows starting a new question from results banner', async () => {
    vi.mocked(api.submitQuery).mockResolvedValue(mockSuccessResponse);

    render(<App />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'What causes macroeconomic inflation?' } });
    fireEvent.click(screen.getByRole('button', { name: /Start Multi-Agent Debate/i }));

    await waitFor(() => {
      expect(screen.getByText(/Final Synthesis & Adjudication/i)).toBeInTheDocument();
    });

    const newQueryBtn = screen.getByRole('button', { name: /New Question/i });
    fireEvent.click(newQueryBtn);

    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.queryByText(/Final Synthesis & Adjudication/i)).not.toBeInTheDocument();
  });
});
