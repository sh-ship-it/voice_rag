import { useEffect, useRef } from 'react';

interface Props {
  text: string;
  evidenceText?: string;
  streaming?: boolean;
  confidence?: 'high' | 'medium' | 'low';
  grounded?: boolean;
  status?: string;
  responseMode?: 'extractive' | 'llm_generated' | 'refusal' | string;
  loading?: boolean;
}

export function AnswerStream({
  text,
  evidenceText,
  streaming,
  confidence,
  grounded,
  status,
  responseMode = 'extractive',
  loading,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (streaming && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [text, streaming]);

  const isRefusal = status === 'low_confidence_fallback' || confidence === 'low' || grounded === false;
  const isIdle = !text && !loading;

  return (
    <div className={`nb-card answer-box ${isRefusal ? 'answer-box--refusal' : 'answer-box--success'}`}>
      <div className="answer-box__header">
        <div className="answer-box__badges">
          <span className="nb-badge nb-badge--dark">
            💬 {streaming ? 'EXTRACTING EVIDENCE...' : loading ? 'RETRIEVING EVIDENCE...' : isIdle ? 'OUTPUT DISPLAY' : 'EXTRACTED ANSWER'}
          </span>
          {!isIdle && (
            <span className="nb-badge nb-badge--accent">
              📌 {responseMode === 'refusal' || isRefusal ? 'REFUSAL' : 'EXTRACTIVE (ZERO-LLM)'}
            </span>
          )}
          {confidence && !isIdle && (
            <span
              className={`nb-badge ${
                confidence === 'high'
                  ? 'nb-badge--success'
                  : confidence === 'medium'
                  ? 'nb-badge--warning'
                  : 'nb-badge--danger'
              }`}
            >
              {confidence === 'high' ? '✅ HIGH CONFIDENCE' : confidence === 'medium' ? '⚠️ MEDIUM CONFIDENCE' : '❌ LOW CONFIDENCE'}
            </span>
          )}
          {grounded !== undefined && !isIdle && (
            <span className={`nb-badge ${grounded ? 'nb-badge--success' : 'nb-badge--danger'}`}>
              {grounded ? '🛡️ 100% GROUNDED' : '⚠️ UNGROUNDED'}
            </span>
          )}
        </div>
      </div>

      <div className="answer-box__content" dir="auto">
        {isIdle ? (
          <p className="answer-box__text answer-box__text--idle">
            Awaiting question... Type a query or press the microphone on the left to extract the exact grounded sentence from 91,681 passages.
          </p>
        ) : loading && !text ? (
          <p className="answer-box__text answer-box__text--loading">
            <span className="inline-spinner"></span> Searching parallel FAISS + BM25 and extracting exact evidence...
          </p>
        ) : (
          <>
            <div className="answer-box__section">
              <span className="answer-box__section-title">👉 Extracted Answer:</span>
              <p className="answer-box__text">
                {text}
                {streaming && <span className="nb-cursor" aria-hidden="true">█</span>}
              </p>
            </div>

            {evidenceText && !streaming && (
              <div className="evidence-box">
                <span className="evidence-box__title">📖 Supporting Evidence Passage (Verbatim Corpus):</span>
                <p className="evidence-box__text" dir="auto">
                  "{evidenceText}"
                </p>
              </div>
            )}
          </>
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}
