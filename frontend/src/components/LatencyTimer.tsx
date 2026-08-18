interface Props {
  timings?: Record<string, number>;
  total?: number;
  loading?: boolean;
}

const STAGE_LABELS: Record<string, string> = {
  stt: '🎙️ STT (Sarvam)',
  guardrail: '🛡️ Safety Guard',
  embed: '🔢 E5 Embedding',
  retrieve: '🔍 FAISS + BM25',
  gate: '🚦 Conf. Gate',
  generation: '🤖 LLM Gen (Cerebras)',
};

const STAGE_ORDER = ['stt', 'guardrail', 'embed', 'retrieve', 'gate', 'generation'];

export function LatencyTimer({ timings, total, loading }: Props) {
  const hasTimings = timings && Object.keys(timings).length > 0;
  const totalMs = total ?? (hasTimings ? timings['total'] ?? 0 : 0);

  return (
    <div className="nb-card latency-card">
      <div className="latency-card__header">
        <span className="nb-badge nb-badge--dark">⏱️ LATENCY METRICS</span>
        <span className="nb-badge nb-badge--accent latency-card__total-badge">
          {loading ? '⏱️ MEASURING...' : `TOTAL: ${totalMs.toFixed(1)} ms`}
        </span>
      </div>

      <div className="latency-breakdown">
        {STAGE_ORDER.map((stage) => {
          const ms = hasTimings && timings[stage] !== undefined ? timings[stage] : 0;
          const pct = totalMs > 0 ? Math.min(100, Math.max(3, (ms / totalMs) * 100)) : 0;
          return (
            <div key={stage} className="latency-row">
              <span className="latency-row__label">{STAGE_LABELS[stage] || stage}</span>
              <div className="latency-row__bar-track">
                <div
                  className={`latency-row__bar-fill ${loading ? 'latency-row__bar-fill--pulse' : ''}`}
                  style={{ width: loading ? '60%' : `${pct}%` }}
                />
              </div>
              <span className="latency-row__ms">{hasTimings ? `${ms.toFixed(1)} ms` : '--'}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
