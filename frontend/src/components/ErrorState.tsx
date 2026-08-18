interface Props {
  error: string;
  onRetry?: () => void;
}

const ERROR_MESSAGES: Record<string, string> = {
  'Failed to fetch': 'Backend connection error. Please make sure FastAPI backend is running on port 8000.',
  'Network Error': 'Network connection error. Please check your internet connection.',
  '401': 'Authentication failed. Please verify your API key configuration.',
  '429': 'Rate limit reached on API. Please wait a brief moment and retry.',
  '500': 'Server encountered an internal error.',
  '503': 'Backend service initializing. Please try again shortly.',
};

function friendlyMessage(raw: string): string {
  for (const [key, msg] of Object.entries(ERROR_MESSAGES)) {
    if (raw.includes(key)) return msg;
  }
  return raw;
}

export function ErrorState({ error, onRetry }: Props) {
  return (
    <div className="nb-card error-alert" role="alert">
      <div className="error-alert__header">
        <span className="nb-badge nb-badge--danger">⚠️ PIPELINE ERROR</span>
      </div>
      <div className="error-alert__body">
        <p className="error-alert__msg">{friendlyMessage(error)}</p>
        {onRetry && (
          <button type="button" className="nb-btn nb-btn--danger" onClick={onRetry}>
            🔄 Retry Request
          </button>
        )}
      </div>
    </div>
  );
}
