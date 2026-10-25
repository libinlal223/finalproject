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
});
