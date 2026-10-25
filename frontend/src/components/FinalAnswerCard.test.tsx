import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { FinalAnswerCard } from './FinalAnswerCard';

describe('FinalAnswerCard Component', () => {
  it('renders final answer paragraphs, badges, and subtasks', () => {
    const onReset = vi.fn();
    render(
      <FinalAnswerCard
        finalAnswer="Paragraph 1: Core resolution.\n\nParagraph 2: Preserved empirical uncertainty."
        status="completed"
        roundsCompleted={2}
        subtasks={['Decompose primary definitions', 'Analyze causal mechanisms']}
        onReset={onReset}
      />,
    );

    expect(screen.getByText(/Final Synthesis & Adjudication/i)).toBeInTheDocument();
    expect(screen.getByText(/Paragraph 1: Core resolution./i)).toBeInTheDocument();
    expect(screen.getByText(/Paragraph 2: Preserved empirical uncertainty./i)).toBeInTheDocument();
    expect(screen.getByText(/Adjudicated by Judge/i)).toBeInTheDocument();
    expect(screen.getByText(/2 Rounds Completed/i)).toBeInTheDocument();
    expect(screen.getByText(/Decompose primary definitions/i)).toBeInTheDocument();
    expect(screen.getByText(/Analyze causal mechanisms/i)).toBeInTheDocument();
  });

  it('renders converged badge when status is converged', () => {
    render(
      <FinalAnswerCard
        finalAnswer="Consensus answer."
        status="converged"
        roundsCompleted={1}
        subtasks={[]}
        onReset={vi.fn()}
      />,
    );
    expect(screen.getByText(/Debate Converged/i)).toBeInTheDocument();
  });

  it('calls onReset when New Debate button is clicked', () => {
    const onReset = vi.fn();
    render(
      <FinalAnswerCard
        finalAnswer="Answer text."
        status="completed"
        roundsCompleted={1}
        subtasks={[]}
        onReset={onReset}
      />,
    );

    const newBtn = screen.getByRole('button', { name: /New Debate/i });
    fireEvent.click(newBtn);
    expect(onReset).toHaveBeenCalled();
  });

  it('copies answer text to clipboard when Copy button is clicked', async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    render(
      <FinalAnswerCard
        finalAnswer="Answer to be copied."
        status="completed"
        roundsCompleted={1}
        subtasks={[]}
        onReset={vi.fn()}
      />,
    );

    const copyBtn = screen.getByRole('button', { name: /Copy/i });
    await act(async () => {
      fireEvent.click(copyBtn);
    });
    expect(writeTextMock).toHaveBeenCalledWith('Answer to be copied.');
  });
});
