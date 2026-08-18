export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface PipelineResponse {
  query: string;
  transcription?: string;
  answer: string;
  evidence_text?: string;
  citations: string[];
  confidence: 'high' | 'medium' | 'low';
  confidence_tier?: 'high' | 'medium' | 'low';
  grounded: boolean;
  status: string;
  response_mode?: 'extractive' | 'llm_generated' | 'refusal';
  total_rag_core_ms: number;
  stt_ms: number;
  timings: Record<string, number>;
  retrieval_result?: {
    chunks: Array<{
      chunk: { chunk_id: string; text: string; language?: string; parent_text?: string };
      score: number;
      rank: number;
    }>;
  };
}

export interface StreamEvent {
  type: 'transcript' | 'token' | 'citations' | 'timings' | 'meta' | 'done' | 'error';
  data: Record<string, unknown>;
}

/** POST /query — text query */
export async function queryText(
  query: string,
  languageCode = 'hi-IN',
  topK = 6,
): Promise<PipelineResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, language_code: languageCode, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status}`);
  return res.json();
}

/** POST /voice — audio file */
export async function queryVoice(
  audioBlob: Blob,
  languageCode = 'hi-IN',
): Promise<PipelineResponse> {
  const form = new FormData();
  form.append('audio', audioBlob, 'recording.wav');
  form.append('language_code', languageCode);
  form.append('top_k', '6');
  const res = await fetch(`${API_URL}/voice`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Voice query failed: ${res.status}`);
  return res.json();
}

/** GET /health */
export async function checkHealth(): Promise<{ status: string; chunk_count: number }> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/** SSE stream for /stream endpoint */
export function openStream(
  query: string,
  languageCode = 'hi-IN',
  onEvent: (event: StreamEvent) => void,
): EventSource {
  const url = `${API_URL}/stream?query=${encodeURIComponent(query)}&language_code=${languageCode}`;
  const es = new EventSource(url);

  const handleEvent = (type: string) => (e: MessageEvent) => {
    try {
      onEvent({ type: type as StreamEvent['type'], data: JSON.parse(e.data) });
    } catch {
      onEvent({ type: type as StreamEvent['type'], data: { raw: e.data } });
    }
  };

  es.addEventListener('transcript', handleEvent('transcript'));
  es.addEventListener('token', handleEvent('token'));
  es.addEventListener('citations', handleEvent('citations'));
  es.addEventListener('timings', handleEvent('timings'));
  es.addEventListener('meta', handleEvent('meta'));
  es.addEventListener('done', handleEvent('done'));
  es.addEventListener('error', handleEvent('error'));

  return es;
}
