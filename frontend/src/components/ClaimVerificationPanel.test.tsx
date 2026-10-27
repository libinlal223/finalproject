/**
 * ClaimVerificationPanel.test.tsx
 * ────────────────────────────────
 * Tests for the ClaimVerificationPanel component covering:
 *
 *  1. Renders nothing harmful when report is null.
 *  2. Renders "no claims" state for NO_CLAIMS report_status.
 *  3. Renders COMPLETED status badge.
 *  4. Renders UNAVAILABLE status badge.
 *  5. Renders FAILED status badge.
 *  6. Renders each claim status badge (supported, contradicted, insufficient_evidence, not_checked).
 *  7. Renders claim count statistics.
 *  8. Expands claim card to show explanation.
 *  9. Expands evidence excerpt toggle safely.
 * 10. External links have noopener noreferrer.
 * 11. Insufficient-evidence notice is not displayed as "false".
 * 12. supported notice does not claim absolute truth.
 * 13. Renders summary text.
 * 14. Renders limitations text.
 * 15. Renders multiple claims correctly.
 */

import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ClaimVerificationPanel } from './ClaimVerificationPanel';
import type { ClaimVerificationReport } from '../types/api';

// ─── Factories ────────────────────────────────────────────────────────────────

function makeReport(
  overrides: Partial<ClaimVerificationReport> = {}
): ClaimVerificationReport {
  return {
    report_status: 'completed',
    results: [],
    summary: 'Test summary.',
    evidence_count: 0,
    claims_checked: 0,
    claims_not_checked: 0,
    ...overrides,
  };
}

const supportedResult = {
  claim_id: 'CLAIM-1',
  claim_text: 'Earth is the third planet from the Sun.',
  status: 'supported' as const,
  explanation: 'Confirmed by astronomy sources.',
  evidence: [
    {
      title: 'NASA Solar System',
      url: 'https://nasa.gov/solar-system',
      excerpt: 'Earth is the third planet.',
      relationship: 'supports',
    },
  ],
  limitations: null,
};

const contradictedResult = {
  claim_id: 'CLAIM-2',
  claim_text: 'The Moon is made of cheese.',
  status: 'contradicted' as const,
  explanation: 'All scientific sources confirm the Moon is rocky.',
  evidence: [
    {
      title: 'Lunar Science',
      url: 'https://example.org/moon',
      excerpt: 'The Moon is composed of rock and regolith.',
      relationship: 'contradicts',
    },
  ],
  limitations: null,
};

const inconclusiveResult = {
  claim_id: 'CLAIM-3',
  claim_text: 'AI will solve all problems by 2030.',
  status: 'insufficient_evidence' as const,
  explanation: 'No conclusive evidence found for this timeline.',
  evidence: [],
  limitations: 'No reliable sources retrieved for this specific prediction.',
};

const notCheckedResult = {
  claim_id: 'CLAIM-4',
  claim_text: 'Some unverified claim.',
  status: 'not_checked' as const,
  explanation: 'Claim could not be verified due to retrieval error.',
  evidence: [],
  limitations: 'Retrieval status: error',
};

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('ClaimVerificationPanel', () => {

  // 1. Null report
  it('renders gracefully when report is null', () => {
    render(<ClaimVerificationPanel report={null} />);
    expect(screen.getByText('Claim Verification Report')).toBeTruthy();
    expect(screen.getByText(/not performed/i)).toBeTruthy();
  });

  // 2. No claims state
  it('renders no_claims empty state correctly', () => {
    const report = makeReport({
      report_status: 'no_claims',
      summary: 'No claims to verify.',
    });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getAllByText(/No Claims to Verify/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/No factual claims were extracted/i).length).toBeGreaterThan(0);
  });

  // 3. COMPLETED status badge
  it('renders Verification Completed badge for completed status', () => {
    const report = makeReport({ report_status: 'completed' });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getByText('Verification Completed')).toBeTruthy();
  });

  // 4. UNAVAILABLE status badge
  it('renders Verification Unavailable badge for unavailable status', () => {
    const report = makeReport({ report_status: 'unavailable' });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getByText('Verification Unavailable')).toBeTruthy();
  });

  // 5. FAILED status badge
  it('renders Verification Failed badge for failed status', () => {
    const report = makeReport({ report_status: 'failed' });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getByText('Verification Failed')).toBeTruthy();
  });

  // 6a. Supported badge
  it('renders Supported status badge', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [supportedResult],
      claims_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getAllByText('Supported').length).toBeGreaterThan(0);
  });

  // 6b. Contradicted badge
  it('renders Contradicted status badge', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [contradictedResult],
      claims_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getAllByText('Contradicted').length).toBeGreaterThan(0);
  });

  // 6c. Inconclusive badge
  it('renders Inconclusive badge for insufficient_evidence', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [inconclusiveResult],
      claims_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getAllByText('Inconclusive').length).toBeGreaterThan(0);
  });

  // 6d. Not Checked badge
  it('renders Not Checked badge for not_checked status', () => {
    const report = makeReport({
      report_status: 'unavailable',
      results: [notCheckedResult],
      claims_not_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getAllByText('Not Checked').length).toBeGreaterThan(0);
  });

  // 7. Statistics
  it('renders claim count statistics correctly', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [supportedResult, contradictedResult, inconclusiveResult],
      evidence_count: 5,
      claims_checked: 3,
    });
    render(<ClaimVerificationPanel report={report} />);
    // Stats row should show total claims count = 3 and evidence items = 5
    expect(screen.getByText('Total Claims')).toBeTruthy();
    expect(screen.getByText('3')).toBeTruthy();
    expect(screen.getByText('5')).toBeTruthy();
  });

  // 8. Expand claim card
  it('expands claim card to show explanation on click', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [supportedResult],
      claims_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);

    // Explanation should not be visible initially (body hidden)
    const card = screen.getByRole('button', { name: /CLAIM-1/i });
    fireEvent.click(card);
    expect(screen.getByText('Confirmed by astronomy sources.')).toBeTruthy();
  });

  // 9. Evidence excerpt toggle
  it('shows evidence excerpt when expand button is clicked', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [supportedResult],
      claims_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);

    // Expand the card first
    const card = screen.getByRole('button', { name: /CLAIM-1/i });
    fireEvent.click(card);

    // Now click the excerpt expand button
    const expandBtn = screen.getByTitle('Show excerpt');
    fireEvent.click(expandBtn);
    expect(screen.getByText('Earth is the third planet.')).toBeTruthy();
  });

  // 10. External links security
  it('renders external links with noopener noreferrer', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [supportedResult],
      claims_checked: 1,
    });
    const { container } = render(<ClaimVerificationPanel report={report} />);

    // Expand the card to see evidence
    const card = screen.getByRole('button', { name: /CLAIM-1/i });
    fireEvent.click(card);

    const links = container.querySelectorAll('a[href]');
    links.forEach(link => {
      expect(link.getAttribute('rel')).toContain('noopener');
      expect(link.getAttribute('rel')).toContain('noreferrer');
      expect(link.getAttribute('target')).toBe('_blank');
    });
  });

  // 11. Insufficient evidence is not "false"
  it('shows inconclusive notice that does not equate insufficient_evidence with false', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [inconclusiveResult],
      claims_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);

    // Expand to see the notice
    const card = screen.getByRole('button', { name: /CLAIM-3/i });
    fireEvent.click(card);

    expect(screen.getByText(/does not prove the claim is false|not proof of falsehood/i)).toBeTruthy();
  });

  // 12. Supported does not claim absolute truth
  it('shows notice that supported does not guarantee absolute accuracy', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [supportedResult],
      claims_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);

    const card = screen.getByRole('button', { name: /CLAIM-1/i });
    fireEvent.click(card);

    expect(screen.getByText(/does not guarantee absolute accuracy/i)).toBeTruthy();
  });

  // 13. Summary text rendered
  it('renders the report summary text', () => {
    const report = makeReport({ summary: 'All claims reviewed and evaluated.' });
    render(<ClaimVerificationPanel report={report} />);
    expect(screen.getByText('All claims reviewed and evaluated.')).toBeTruthy();
  });

  // 14. Limitations text rendered
  it('renders claim limitations when present', () => {
    const report = makeReport({
      report_status: 'completed',
      results: [inconclusiveResult],
      claims_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);

    const card = screen.getByRole('button', { name: /CLAIM-3/i });
    fireEvent.click(card);

    expect(screen.getByText('No reliable sources retrieved for this specific prediction.')).toBeTruthy();
  });

  // 15. Multiple claims rendered in order
  it('renders multiple claims with stable claim_ids in order', () => {
    const report = makeReport({
      report_status: 'partial',
      results: [supportedResult, contradictedResult, inconclusiveResult, notCheckedResult],
      claims_checked: 3,
      claims_not_checked: 1,
    });
    render(<ClaimVerificationPanel report={report} />);

    const claimIds = screen.getAllByText(/CLAIM-\d/);
    expect(claimIds.length).toBe(4);
    // Order must be CLAIM-1, CLAIM-2, CLAIM-3, CLAIM-4
    expect(claimIds[0].textContent).toContain('CLAIM-1');
    expect(claimIds[1].textContent).toContain('CLAIM-2');
    expect(claimIds[2].textContent).toContain('CLAIM-3');
    expect(claimIds[3].textContent).toContain('CLAIM-4');
  });
});
