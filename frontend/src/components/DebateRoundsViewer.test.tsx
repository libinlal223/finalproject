import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DebateRoundsViewer } from './DebateRoundsViewer';
import type { DebateRound } from '../types/api';

describe('DebateRoundsViewer Component', () => {
  const mockRounds: DebateRound[] = [
    {
      round_number: 1,
      executor_a_answer: 'Executor A deductive proposal: Axiomatic derivation.',
      executor_b_answer: 'Executor B empirical proposal: Empirical boundary observations.',
      critique: 'Critic Round 1: Contradictions identified regarding friction.',
      similarity_score: 0.65,
      converged: false,
    },
    {
      round_number: 2,
      executor_a_answer: 'Executor A revised proposal with friction qualified.',
      executor_b_answer: 'Executor B revised proposal with unified parameters.',
      critique: 'Critic Round 2: All material issues successfully resolved.',
      similarity_score: 0.94,
      converged: true,
    },
  ];

  it('renders initial round Executor A and B answers, similarity, and critic critique', () => {
    render(<DebateRoundsViewer rounds={mockRounds} />);

    expect(screen.getByText(/Adversarial Debate Rounds/i)).toBeInTheDocument();
    expect(screen.getByText(/Executor A deductive proposal: Axiomatic derivation./i)).toBeInTheDocument();
    expect(screen.getByText(/Executor B empirical proposal: Empirical boundary observations./i)).toBeInTheDocument();
    expect(screen.getByText(/Critic Round 1: Contradictions identified regarding friction./i)).toBeInTheDocument();
    expect(screen.getByText(/65%/i)).toBeInTheDocument();
    expect(screen.getByText(/Material Issues Present/i)).toBeInTheDocument();
  });

  it('switches to round 2 when Round 2 tab is clicked', () => {
    render(<DebateRoundsViewer rounds={mockRounds} />);

    const round2Tab = screen.getByRole('tab', { name: /Round 2/i });
    fireEvent.click(round2Tab);

    expect(screen.getByText(/Executor A revised proposal with friction qualified./i)).toBeInTheDocument();
    expect(screen.getByText(/Executor B revised proposal with unified parameters./i)).toBeInTheDocument();
    expect(screen.getByText(/Critic Round 2: All material issues successfully resolved./i)).toBeInTheDocument();
    expect(screen.getByText(/94%/i)).toBeInTheDocument();
    expect(screen.getByText(/Converged/i)).toBeInTheDocument();
  });

  it('returns null when rounds array is empty', () => {
    const { container } = render(<DebateRoundsViewer rounds={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders structured critic findings, claim ID links, addressed issues, and revision notes', () => {
    const roundWithPriority2: DebateRound[] = [
      {
        round_number: 1,
        executor_a_answer: 'Exec A answer text',
        executor_b_answer: 'Exec B answer text',
        critique: 'Critic overview text',
        similarity_score: 0.72,
        converged: false,
        issues: [
          {
            issue_id: 'ISSUE-1',
            claim_id: 'CLAIM-42',
            verification_status: 'contradicted',
            target: 'executor_a',
            severity: 'high',
            affected_claim: 'Cold fusion efficiency rate',
            explanation: 'Contradicted by empirical calorimetric measurements.',
            recommended_correction: 'Retract cold fusion claim and state boundary regimes.',
          },
        ],
        addressed_issues: ['ISSUE-1'],
        executor_a_notes: 'Retracted cold fusion claim; preserved verified baseline.',
        executor_b_notes: 'Nuanced operational conditions.',
      },
    ];

    render(<DebateRoundsViewer rounds={roundWithPriority2} />);

    expect(screen.getByTestId('critic-issues-list')).toBeInTheDocument();
    expect(screen.getByText(/HIGH/i)).toBeInTheDocument();
    expect(screen.getAllByText(/ISSUE-1/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/🔗 CLAIM-42/i)).toBeInTheDocument();
    expect(screen.getByText(/Cold fusion efficiency rate/i)).toBeInTheDocument();
    expect(screen.getByText(/Contradicted by empirical calorimetric measurements./i)).toBeInTheDocument();
    expect(screen.getByText(/Retract cold fusion claim and state boundary regimes./i)).toBeInTheDocument();

    expect(screen.getByTestId('addressed-issues-list')).toBeInTheDocument();
    expect(screen.getByText(/✓ ISSUE-1/i)).toBeInTheDocument();

    expect(screen.getByTestId('executor-a-notes')).toHaveTextContent(/Retracted cold fusion claim/i);
    expect(screen.getByTestId('executor-b-notes')).toHaveTextContent(/Nuanced operational conditions/i);
  });
});
