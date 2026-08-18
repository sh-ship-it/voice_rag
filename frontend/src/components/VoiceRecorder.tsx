import { useRef, useState, useEffect } from 'react';

interface Props {
  onAudio: (blob: Blob, lang: 'hi-IN' | 'en-IN') => void;
  disabled?: boolean;
  selectedLanguage: 'hi-IN' | 'en-IN';
  onLanguageChange: (lang: 'hi-IN' | 'en-IN') => void;
}

type RecordingState = 'idle' | 'recording' | 'processing';

export function VoiceRecorder({ onAudio, disabled, selectedLanguage, onLanguageChange }: Props) {
  const [state, setState] = useState<RecordingState>('idle');
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<BlobPart[]>([]);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (state === 'recording') {
      setRecordingSeconds(0);
      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [state]);

  const getMimeType = (): string => {
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) return 'audio/webm;codecs=opus';
    if (MediaRecorder.isTypeSupported('audio/webm')) return 'audio/webm';
    if (MediaRecorder.isTypeSupported('audio/mp4')) return 'audio/mp4';
    return '';
  };

  const startRecording = async (lang: 'hi-IN' | 'en-IN') => {
    try {
      onLanguageChange(lang);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks.current = [];
      const mimeType = getMimeType();
      const options: MediaRecorderOptions = mimeType ? { mimeType } : {};
      const mr = new MediaRecorder(stream, options);

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunks.current.push(e.data);
        }
      };

      mr.onstop = () => {
        const finalType = mr.mimeType || 'audio/webm';
        const blob = new Blob(chunks.current, { type: finalType });
        stream.getTracks().forEach((t) => t.stop());
        setState('processing');
        onAudio(blob, lang);
      };

      mr.start(250);
      mediaRecorder.current = mr;
      setState('recording');
    } catch (err) {
      console.error('Microphone access error:', err);
      alert('Microphone permission denied or microphone not found.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
      mediaRecorder.current.stop();
    }
  };

  const isRecording = state === 'recording';
  const isProcessing = state === 'processing';

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="voice-control-box">
      <div className="voice-control-box__header">
        <span className="nb-badge nb-badge--dark">🎙️ Voice Input (Sarvam AI STT)</span>
        {isRecording && (
          <span className="nb-badge nb-badge--danger recording-live-indicator">
            <span className="pulse-dot"></span> REC ({formatTime(recordingSeconds)})
          </span>
        )}
      </div>

      <div className="voice-buttons-row">
        {/* Hindi Recording Button */}
        <button
          type="button"
          className={`nb-btn voice-btn-action ${selectedLanguage === 'hi-IN' ? 'voice-btn-action--active' : ''} ${
            isRecording && selectedLanguage === 'hi-IN' ? 'voice-btn-action--recording' : ''
          }`}
          onClick={() => {
            if (isRecording) {
              stopRecording();
            } else {
              startRecording('hi-IN');
            }
          }}
          disabled={disabled || (isRecording && selectedLanguage !== 'hi-IN') || isProcessing}
          id="btn-voice-hindi"
          aria-label="Record Hindi voice query"
        >
          {isProcessing && selectedLanguage === 'hi-IN' ? (
            '⏳ Processing...'
          ) : isRecording && selectedLanguage === 'hi-IN' ? (
            '⏹️ Stop Recording (Hindi)'
          ) : (
            '🎤 Record Hindi (हिन्दी)'
          )}
        </button>

        {/* English Recording Button */}
        <button
          type="button"
          className={`nb-btn voice-btn-action ${selectedLanguage === 'en-IN' ? 'voice-btn-action--active' : ''} ${
            isRecording && selectedLanguage === 'en-IN' ? 'voice-btn-action--recording' : ''
          }`}
          onClick={() => {
            if (isRecording) {
              stopRecording();
            } else {
              startRecording('en-IN');
            }
          }}
          disabled={disabled || (isRecording && selectedLanguage !== 'en-IN') || isProcessing}
          id="btn-voice-english"
          aria-label="Record English voice query"
        >
          {isProcessing && selectedLanguage === 'en-IN' ? (
            '⏳ Processing...'
          ) : isRecording && selectedLanguage === 'en-IN' ? (
            '⏹️ Stop Recording (English)'
          ) : (
            '🎤 Record English (EN)'
          )}
        </button>
      </div>

      <div className="voice-status-bar">
        <span className="voice-status-bar__hint">
          {isRecording
            ? `Speaking in ${selectedLanguage === 'hi-IN' ? 'Hindi (हिन्दी)' : 'English'}... Click stop when finished.`
            : isProcessing
            ? 'Transcribing speech with Sarvam saaras:v3...'
            : 'Click a button above to record speech in Hindi or English.'}
        </span>
      </div>
    </div>
  );
}
