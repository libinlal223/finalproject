import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryForm } from './QueryForm';

describe('QueryForm Component', () => {
  it('renders input with placeholder and disabled submit button initially', () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} isLoading={false} />);

    const textarea = screen.getByPlaceholderText(/Type your complex analytical question here/i);
    expect(textarea).toBeInTheDocument();

    const submitBtn = screen.getByRole('button', { name: /Start Multi-Agent Debate/i });
    expect(submitBtn).toBeDisabled();
  });

  it('enables submit button when valid input is typed and calls onSubmit', () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} isLoading={false} />);

    const textarea = screen.getByPlaceholderText(/Type your complex analytical question here/i);
    fireEvent.change(textarea, { target: { value: 'Why is the sky blue?' } });

    const submitBtn = screen.getByRole('button', { name: /Start Multi-Agent Debate/i });
    expect(submitBtn).not.toBeDisabled();

    fireEvent.click(submitBtn);
    expect(onSubmit).toHaveBeenCalledWith('Why is the sky blue?', 3, null);
  });

  it('disables submit button and shows warning if input is under 3 characters', () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} isLoading={false} />);

    const textarea = screen.getByPlaceholderText(/Type your complex analytical question here/i);
    fireEvent.change(textarea, { target: { value: 'hi' } });

    expect(screen.getByText(/Minimum 3 characters required/i)).toBeInTheDocument();
    const submitBtn = screen.getByRole('button', { name: /Start Multi-Agent Debate/i });
    expect(submitBtn).toBeDisabled();
  });

  it('populates input when an example query is clicked', () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} isLoading={false} />);

    const exampleBtn = screen.getByRole('button', {
      name: /What are the primary macroeconomic mechanisms driving stagflation/i,
    });
    fireEvent.click(exampleBtn);

    const textarea = screen.getByRole('textbox') as HTMLTextAreaElement;
    expect(textarea.value).toContain('macroeconomic mechanisms driving stagflation');

    const submitBtn = screen.getByRole('button', { name: /Start Multi-Agent Debate/i });
    expect(submitBtn).not.toBeDisabled();
  });

  it('submits on Ctrl+Enter keyboard shortcut', () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} isLoading={false} />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Complex theoretical inquiry' } });

    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true });
    expect(onSubmit).toHaveBeenCalledWith('Complex theoretical inquiry', 3, null);
  });

  it('toggles settings panel and updates maxRounds and provider', () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} isLoading={false} />);

    const settingsBtn = screen.getByRole('button', { name: /Debate Settings/i });
    fireEvent.click(settingsBtn);

    expect(screen.getByText(/Max Debate Rounds:/i)).toBeInTheDocument();
    const slider = screen.getByRole('slider');
    fireEvent.change(slider, { target: { value: '4' } });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'groq' } });

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Inquiry with custom settings' } });

    const submitBtn = screen.getByRole('button', { name: /Start Multi-Agent Debate/i });
    fireEvent.click(submitBtn);

    expect(onSubmit).toHaveBeenCalledWith('Inquiry with custom settings', 4, 'groq');
  });

  it('prevents submission and shows spinner while loading', () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} isLoading={true} />);

    const textarea = screen.getByRole('textbox');
    expect(textarea).toBeDisabled();

    const submitBtn = screen.getByRole('button', { name: /Debating/i });
    expect(submitBtn).toBeDisabled();

    fireEvent.click(submitBtn);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('prevents rapid double-click duplicate submissions', () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} isLoading={false} />);

    const textarea = screen.getByPlaceholderText(/Type your complex analytical question here/i);
    fireEvent.change(textarea, { target: { value: 'Valid test query for debounce' } });

    const submitBtn = screen.getByRole('button', { name: /Start Multi-Agent Debate/i });
    expect(submitBtn).not.toBeDisabled();

    // First click
    fireEvent.click(submitBtn);
    expect(onSubmit).toHaveBeenCalledTimes(1);

    // Immediate second click before re-render
    fireEvent.click(submitBtn);
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });
});
