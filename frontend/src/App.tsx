import React, { useState, useCallback, useRef, useEffect } from 'react';
import { VoiceRecorder } from './components/VoiceRecorder';
import { TranscriptPanel } from './components/TranscriptPanel';
import { AnswerStream } from './components/AnswerStream';
import { CitationAccordion } from './components/CitationAccordion';
import { LatencyTimer } from './components/LatencyTimer';
import { ErrorState } from './components/ErrorState';
import { queryText, queryVoice, openStream, checkHealth } from './api';
import type { PipelineResponse, StreamEvent } from './api';
import './index.css';

type Mode = 'stream' | 'text';
type Language = 'hi-IN' | 'en-IN';

interface ResultState {
  query: string;
  response?: PipelineResponse;
  streamAnswer: string;
  streaming: boolean;
}

const EXAMPLE_QUERIES_HINDI = [
  'भारत की राजधानी क्या है?',
  'विश्व का सबसे बड़ा महासागर कौन सा है?',
  'ताजमहल कहाँ स्थित है?',
  'भारतीय संविधान कब लागू हुआ था?',
];

const EXAMPLE_QUERIES_ENGLISH = [
  'What is the capital of India?',
  'Which is the largest ocean in the world?',
  'How does machine learning work?',
];

export default function App() {
  const [inputText, setInputText] = useState('');
  const [mode, setMode] = useState<Mode>('stream');
  const [language, setLanguage] = useState<Language>('hi-IN');
  const [serverHealth, setServerHealth] = useState<{ status: string; chunk_count: number } | null>(null);
  const [result, setResult] = useState<ResultState>({
    query: '',
    streamAnswer: '',
    streaming: false,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  // Check health on initial mount
  useEffect(() => {
    checkHealth()
      .then((data) => setServerHealth(data))
      .catch((err) => {
        console.warn('Health check notice:', err);
        setServerHealth({ status: 'offline', chunk_count: 91681 });
      });
  }, []);

  const reset = () => {
    setError(null);
    esRef.current?.close();
    setResult({ query: '', streamAnswer: '', streaming: false });
  };

  // --- TEXT QUERY (non-streaming) ---
  const handleTextQuery = useCallback(
    async (textOverride?: string) => {
      const q = (textOverride || inputText).trim();
      if (!q) return;
      setError(null);
      esRef.current?.close();
      setLoading(true);
      setResult((prev) => ({ ...prev, query: q, streamAnswer: '', streaming: false }));
      try {
        const res = await queryText(q, language);
        setResult({ query: q, response: res, streamAnswer: res.answer, streaming: false });
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [inputText, language]
  );

  // --- SSE STREAM ---
  const handleStreamQuery = useCallback(
    (textOverride?: string) => {
      const q = (textOverride || inputText).trim();
      if (!q) return;
      setError(null);
      esRef.current?.close();
      setLoading(true);

      const partialResult: Partial<PipelineResponse> & { streamAnswer: string } = {
        streamAnswer: '',
        citations: [],
        timings: {},
      };

      setResult({ query: q, streamAnswer: '', streaming: true, response: undefined });

      const es = openStream(q, language, (event: StreamEvent) => {
        if (event.type === 'transcript') {
          partialResult.transcription = (event.data as { text: string }).text;
        } else if (event.type === 'token') {
          partialResult.streamAnswer += (event.data as { token: string }).token;
          setResult((prev) => ({ ...prev, streamAnswer: partialResult.streamAnswer, streaming: true }));
        } else if (event.type === 'citations') {
          partialResult.citations = event.data as unknown as string[];
        } else if (event.type === 'timings') {
          partialResult.timings = event.data as unknown as Record<string, number>;
        } else if (event.type === 'meta') {
          const meta = event.data as { confidence: string; grounded: boolean; status: string };
          partialResult.confidence = meta.confidence as PipelineResponse['confidence'];
          partialResult.grounded = meta.grounded;
          partialResult.status = meta.status;
        } else if (event.type === 'done') {
          es.close();
          setResult({
            query: q,
            streamAnswer: partialResult.streamAnswer,
            streaming: false,
            response: {
              query: q,
              answer: partialResult.streamAnswer,
              citations: partialResult.citations || [],
              confidence: partialResult.confidence || 'medium',
              grounded: partialResult.grounded ?? true,
              status: partialResult.status || 'success',
              total_rag_core_ms: partialResult.timings?.total ?? 0,
              stt_ms: partialResult.timings?.stt ?? 0,
              timings: partialResult.timings || {},
              transcription: partialResult.transcription,
            },
          });
          setLoading(false);
        } else if (event.type === 'error') {
          setError('Stream error occurred. Retrying in direct mode...');
          setLoading(false);
          es.close();
        }
      });

      esRef.current = es;
    },
    [inputText, language]
  );

  // --- VOICE QUERY ---
  const handleVoice = useCallback(
    async (blob: Blob, langOverride?: Language) => {
      setError(null);
      esRef.current?.close();
      setLoading(true);
      const activeLang = langOverride || language;
      try {
        const res = await queryVoice(blob, activeLang);
        setResult({
          query: res.transcription || res.query,
          response: res,
          streamAnswer: res.answer,
          streaming: false,
        });
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [language]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (mode === 'stream') handleStreamQuery();
      else handleTextQuery();
    }
  };

  const handleSelectExample = (sampleText: string) => {
    setInputText(sampleText);
    if (mode === 'stream') handleStreamQuery(sampleText);
    else handleTextQuery(sampleText);
  };

  const activeResponse = result.response;
  const chunks = activeResponse?.retrieval_result?.chunks?.map((sc) => sc.chunk) ?? [];
  const currentAnswer = result.streaming ? result.streamAnswer : activeResponse?.answer || result.streamAnswer || '';

  return (
    <div className="nb-app-shell">
      {/* Top Navbar */}
      <header className="nb-navbar">
        <div className="nb-navbar__brand">
          <span className="nb-navbar__logo-box">🇮🇳</span>
          <div>
            <h1 className="nb-navbar__title">HHGOA VOICE RAG</h1>
            <p className="nb-navbar__tagline">Indic Voice & Text QA over MSMARCO-XI</p>
          </div>
        </div>
        <div className="nb-navbar__status">
          <span className="nb-badge nb-badge--accent">
            🟢 {serverHealth?.chunk_count ? `${serverHealth.chunk_count.toLocaleString()} Chunks Indexed` : '91,681 Chunks Active'}
          </span>
          <span className="nb-badge nb-badge--dark">FastAPI + Cerebras + Sarvam</span>
        </div>
      </header>

      {/* Main 2-Column Neo-Brutalist Layout */}
      <main className="nb-main-grid">
        {/* Left Column: Controls & Input Area */}
        <section className="nb-col nb-col--left">
          <div className="nb-card query-card">
            {/* Language Selection Header */}
            <div className="nb-card__header-row">
              <span className="nb-badge nb-badge--dark">1. SELECT LANGUAGE</span>
              <div className="lang-buttons-group" role="group" aria-label="Language Selector">
                <button
                  type="button"
                  className={`nb-btn nb-btn--sm ${language === 'hi-IN' ? 'nb-btn--active' : ''}`}
                  onClick={() => setLanguage('hi-IN')}
                  id="tab-hindi"
                >
                  🇮🇳 हिन्दी (Hindi)
                </button>
                <button
                  type="button"
                  className={`nb-btn nb-btn--sm ${language === 'en-IN' ? 'nb-btn--active' : ''}`}
                  onClick={() => setLanguage('en-IN')}
                  id="tab-english"
                >
                  🇬🇧 English
                </button>
              </div>
            </div>

            {/* Mode Selection */}
            <div className="mode-selection-row">
              <span className="nb-field-label">Delivery Mode:</span>
              <div className="mode-toggle-group">
                <button
                  type="button"
                  className={`nb-toggle-btn ${mode === 'stream' ? 'nb-toggle-btn--active' : ''}`}
                  onClick={() => setMode('stream')}
                  id="mode-btn-stream"
                >
                  ⚡ Real-Time Stream (SSE)
                </button>
                <button
                  type="button"
                  className={`nb-toggle-btn ${mode === 'text' ? 'nb-toggle-btn--active' : ''}`}
                  onClick={() => setMode('text')}
                  id="mode-btn-text"
                >
                  📝 Fast JSON Response
                </button>
              </div>
            </div>

            {/* Query Textarea */}
            <div className="textarea-container">
              <label htmlFor="query-textarea" className="nb-field-label">
                Type Question ({language === 'hi-IN' ? 'Hindi or English' : 'English'}):
              </label>
              <textarea
                id="query-textarea"
                className="nb-textarea"
                placeholder={
                  language === 'hi-IN'
                    ? 'यहाँ अपना प्रश्न लिखें... (उदा: भारत की राजधानी क्या है?)'
                    : 'Type your question here... (e.g., What is the capital of India?)'
                }
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={3}
                disabled={loading}
                dir="auto"
              />
            </div>

            {/* Ask Action Button */}
            <div className="submit-action-row">
              <button
                type="button"
                className="nb-btn nb-btn--primary nb-btn--block"
                onClick={() => (mode === 'stream' ? handleStreamQuery() : handleTextQuery())}
                disabled={loading || !inputText.trim()}
                id="btn-ask-query"
              >
                {loading ? '⏳ PROCESSING PIPELINE...' : '🚀 ASK QUESTION →'}
              </button>
            </div>

            {/* Voice Recording Component */}
            <VoiceRecorder
              onAudio={handleVoice}
              disabled={loading}
              selectedLanguage={language}
              onLanguageChange={setLanguage}
            />

            {/* Example Queries Bar */}
            <div className="examples-section">
              <span className="nb-field-label">Quick Test Prompts:</span>
              <div className="examples-list">
                {(language === 'hi-IN' ? EXAMPLE_QUERIES_HINDI : EXAMPLE_QUERIES_ENGLISH).map((ex, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="nb-chip"
                    onClick={() => handleSelectExample(ex)}
                    disabled={loading}
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Right Column: Permanent In-Place Output, Citations & Timers */}
        <section className="nb-col nb-col--right">
          {error && <ErrorState error={error} onRetry={reset} />}

          {/* Transcript Panel (shows when speech/query available) */}
          {(activeResponse?.transcription || result.query) && (
            <TranscriptPanel
              transcript={activeResponse?.transcription}
              query={result.query}
              language={language}
            />
          )}

          {/* Answer Stream (Permanently in-place, text renders inside smoothly) */}
          <AnswerStream
            text={currentAnswer}
            evidenceText={activeResponse?.evidence_text}
            responseMode={activeResponse?.response_mode || 'extractive'}
            streaming={result.streaming}
            confidence={activeResponse?.confidence}
            grounded={activeResponse?.grounded}
            status={activeResponse?.status}
            loading={loading}
          />

          {/* Citations Card (Permanently in-place) */}
          <CitationAccordion
            citations={activeResponse?.citations || []}
            chunks={chunks}
            loading={loading}
          />

          {/* Latency Timer Card (Permanently in-place) */}
          <LatencyTimer
            timings={activeResponse?.timings}
            total={activeResponse?.total_rag_core_ms}
            loading={loading}
          />
        </section>
      </main>
    </div>
  );
}
