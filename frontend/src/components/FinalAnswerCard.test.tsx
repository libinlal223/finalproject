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

  it('renders verdict summary and unresolved uncertainties when provided', () => {
    render(
      <FinalAnswerCard
        finalAnswer="Answer with adjudication rationale."
        status="completed"
        roundsCompleted={2}
        subtasks={['Subtask 1']}
        verdictSummary="Selected Executor A candidate because it successfully retracted contradicted claims and preserved supported axioms."
        unresolvedUncertainties={[
          'Exact thermodynamic efficiency under 0K boundary conditions remains experimentally unmeasured.',
        ]}
        onReset={vi.fn()}
      />,
    );

    expect(screen.getByTestId('verdict-summary')).toBeInTheDocument();
    expect(screen.getByText(/Judge Adjudication Rationale/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Selected Executor A candidate because it successfully retracted contradicted claims/i),
    ).toBeInTheDocument();

    expect(screen.getByTestId('unresolved-uncertainties')).toBeInTheDocument();
    expect(screen.getByText(/Epistemic Uncertainty & Unresolved Boundaries/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Exact thermodynamic efficiency under 0K boundary conditions remains experimentally unmeasured./i),
    ).toBeInTheDocument();
  });

  it('renders unadjudicated fallback banner and error badge when debate halts on error', () => {
    render(
      <FinalAnswerCard
        finalAnswer="[UNADJUDICATED FALLBACK CANDIDATE - NOT VERIFIED]\n\nCandidate text from Executor A."
        status="error"
        roundsCompleted={0}
        subtasks={['Subtask 1']}
        verdictSummary="Critique step failed; adjudication was skipped and unadjudicated fallback candidate used without Judge review."
        unresolvedUncertainties={[
          'Unverified candidate: Critic did not complete critique; verification-grounded adjudication was not performed.',
        ]}
        onReset={vi.fn()}
      />,
    );

    expect(screen.getByText(/Candidate Output \(Unadjudicated Fallback\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Halted on Error/i)).toBeInTheDocument();
    expect(screen.getByText(/0 Rounds Completed/i)).toBeInTheDocument();
    expect(screen.getByTestId('unadjudicated-warning')).toBeInTheDocument();
    expect(screen.getByText(/Unadjudicated Fallback Notice:/i)).toBeInTheDocument();
    expect(screen.getByText(/Adjudication Fallback & Status Rationale/i)).toBeInTheDocument();
    expect(screen.getByText(/Critique step failed/i)).toBeInTheDocument();
  });
});
