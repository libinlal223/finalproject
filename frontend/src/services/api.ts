import type { HealthResponse, QueryRequest, QueryResponse, ErrorDetail } from '../types/api';

const DEFAULT_API_BASE = 'http://localhost:8000';

export function getApiBaseUrl(): string {
  const url = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE;
  return url.replace(/\/+$/, '');
}

export class ApiError extends Error {
  public readonly code: string;
  public readonly status: number;
  public readonly details?: Record<string, unknown> | null;

  constructor(message: string, status: number, code: string = 'api_error', details?: Record<string, unknown> | null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

/**
 * Submit a query to the VADER multi-agent debate orchestrator.
 */
export async function submitQuery(
  request: QueryRequest,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/query`;

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(request),
      signal,
    });
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('Request was aborted.', 0, 'aborted');
    }
    const msg = err instanceof Error ? err.message : 'Network request failed';
    throw new ApiError(
      `Unable to connect to VADER backend at ${baseUrl}. Ensure backend is running. (${msg})`,
      0,
      'network_error',
    );
  }

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      `Invalid response format from server (status ${response.status}). Expected JSON.`,
      response.status,
      'invalid_json',
    );
  }

  if (!response.ok) {
    const errorPayload = data as Partial<ErrorDetail>;
    const errorMessage = errorPayload.message || `Request failed with status ${response.status}`;
    const errorCode = errorPayload.error || `http_${response.status}`;
    throw new ApiError(errorMessage, response.status, errorCode, errorPayload.details);
  }

  return data as QueryResponse;
}

/**
 * Query the backend health check endpoint.
 */
export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/health`;

  try {
    const res = await fetch(endpoint, {
      headers: { Accept: 'application/json' },
      signal,
    });
    if (!res.ok) {
      throw new ApiError(`Health check returned status ${res.status}`, res.status);
    }
    return (await res.json()) as HealthResponse;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    const msg = err instanceof Error ? err.message : 'Connection failed';
    throw new ApiError(`Backend unreachable at ${baseUrl}: ${msg}`, 0, 'network_error');
  }
}
