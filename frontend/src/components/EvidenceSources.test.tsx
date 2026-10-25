import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EvidenceSources } from './EvidenceSources';
import type { EvidenceItem } from '../types/api';

describe('EvidenceSources Component', () => {
  it('renders safe source links with target=_blank and rel=noopener noreferrer', () => {
    const mockSources: EvidenceItem[] = [
      {
        title: 'Macroeconomic Outlook 2026',
        url: 'https://imf.example.org/reports/inflation-2026',
        snippet: 'Global monetary tightening has dampened demand-pull inflation.',
        domain: 'imf.example.org',
        source: 'mock',
      },
    ];

    render(<EvidenceSources sources={mockSources} />);

    expect(screen.getByText(/Grounded Evidence & Citations/i)).toBeInTheDocument();
    expect(screen.getByText(/Macroeconomic Outlook 2026/i)).toBeInTheDocument();
    expect(screen.getByText(/Global monetary tightening has dampened demand-pull inflation./i)).toBeInTheDocument();
    expect(screen.getAllByText(/imf.example.org/i).length).toBeGreaterThan(0);

    const link = screen.getByRole('link', { name: /Macroeconomic Outlook 2026/i });
    expect(link).toHaveAttribute('href', 'https://imf.example.org/reports/inflation-2026');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('renders empty notice when no sources are provided', () => {
    render(<EvidenceSources sources={[]} />);
    expect(
      screen.getByText(/No external web evidence was retrieved or cited for this query/i),
    ).toBeInTheDocument();
  });

  it('safely renders untrusted snippet content without executing script tags', () => {
    const maliciousSources: EvidenceItem[] = [
      {
        title: 'Untrusted Page <script>alert("xss")</script>',
        url: 'https://safe-domain.org/article',
        snippet: 'Attempted attack: <img src="x" onerror="alert(1)"> in snippet',
        domain: 'safe-domain.org',
      },
    ];

    const { container } = render(<EvidenceSources sources={maliciousSources} />);

    // Script tags should be rendered as plain text nodes, not executable HTML elements
    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('img[onerror]')).toBeNull();
    expect(screen.getByText(/Untrusted Page <script>alert\("xss"\)<\/script>/i)).toBeInTheDocument();
  });
});
