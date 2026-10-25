import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { submitQuery, checkHealth, ApiError, getApiBaseUrl } from './api';
import type { QueryResponse, HealthResponse } from '../types/api';

describe('API Service', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('getApiBaseUrl strips trailing slashes', () => {
    const url = getApiBaseUrl();
    expect(url).not.toMatch(/\/$/);
  });

  describe('submitQuery', () => {
    it('sends POST request with JSON body and returns QueryResponse', async () => {
      const mockResponse: QueryResponse = {
        query: 'What causes inflation?',
        status: 'completed',
        rounds_completed: 1,
        final_answer: 'Inflation is driven by demand-pull and cost-push factors.',
        subtasks: ['Analyze monetary supply', 'Analyze supply chains'],
        debate_rounds: [],
        trace: [],
        sources: [],
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockResponse,
      } as unknown as Response);

      const result = await submitQuery({
        query: 'What causes inflation?',
        max_rounds: 2,
        llm_provider: 'mock',
      });

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/query'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify({
            query: 'What causes inflation?',
            max_rounds: 2,
            llm_provider: 'mock',
          }),
        }),
      );
      expect(result).toEqual(mockResponse);
    });

    it('throws ApiError on HTTP 400 with backend error details', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          error: 'validation_error',
          message: 'query must not be blank',
          details: { field: 'query' },
        }),
      } as unknown as Response);

      await expect(submitQuery({ query: '   ' })).rejects.toThrow(ApiError);

      try {
        await submitQuery({ query: '   ' });
      } catch (err: unknown) {
        const apiErr = err as ApiError;
        expect(apiErr.status).toBe(400);
        expect(apiErr.code).toBe('validation_error');
        expect(apiErr.message).toBe('query must not be blank');
        expect(apiErr.details).toEqual({ field: 'query' });
      }
    });

    it('throws ApiError with network_error code when fetch fails completely', async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

      await expect(submitQuery({ query: 'Valid query' })).rejects.toThrow(
        /Unable to connect to VADER backend/,
      );

      try {
        await submitQuery({ query: 'Valid query' });
      } catch (err: unknown) {
        const apiErr = err as ApiError;
        expect(apiErr.code).toBe('network_error');
        expect(apiErr.status).toBe(0);
      }
    });

    it('handles malformed non-JSON server responses', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => {
          throw new Error('Unexpected token < in JSON at position 0');
        },
      } as unknown as Response);

      await expect(submitQuery({ query: 'Test' })).rejects.toThrow(
        /Invalid response format from server/,
      );
    });
  });

  describe('checkHealth', () => {
    it('returns HealthResponse on 200', async () => {
      const mockHealth: HealthResponse = {
        status: 'ok',
        version: '0.1.0',
        environment: 'test',
        timestamp: '2026-10-16T00:00:00Z',
        llm_provider: 'mock',
        llm_key_present: true,
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockHealth,
      } as unknown as Response);

      const res = await checkHealth();
      expect(res).toEqual(mockHealth);
    });

    it('throws ApiError when health check fails', async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('Connection refused'));

      await expect(checkHealth()).rejects.toThrow(/Backend unreachable/);
    });
  });
});
