interface Props {
  transcript?: string;
  query?: string;
  language?: string;
}

export function TranscriptPanel({ transcript, query, language }: Props) {
  if (!transcript && !query) return null;

  return (
    <div className="nb-card transcript-box">
      <div className="transcript-box__header">
        <span className="nb-badge nb-badge--accent">
          {transcript ? '🎙️ TRANSCRIPT' : '❓ QUERY'}
        </span>
        {language && (
          <span className="nb-badge nb-badge--light">
            {language === 'hi-IN' ? '🇮🇳 Hindi' : '🇬🇧 English'}
          </span>
        )}
      </div>
      <div className="transcript-box__body">
        <p className="transcript-box__text" dir="auto">
          {transcript || query}
        </p>
      </div>
    </div>
  );
}
