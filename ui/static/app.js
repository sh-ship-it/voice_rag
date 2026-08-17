/**
 * Voice RAG System Frontend Application Logic
 * Audio recording, Web Audio API waveform visualizer, latency telemetry rendering, and citation cards.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const micBtn = document.getElementById('micBtn');
  const micPrompt = document.getElementById('micPrompt');
  const recordingTimer = document.getElementById('recordingTimer');
  const queryForm = document.getElementById('queryForm');
  const queryInput = document.getElementById('queryInput');
  const langSelect = document.getElementById('langSelect');
  const sendBtn = document.getElementById('sendBtn');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const canvasCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;

  // Latency & Result Elements
  const emptyState = document.getElementById('emptyState');
  const responseView = document.getElementById('responseView');
  const coreLatencyValue = document.getElementById('coreLatencyValue');
  const coreLatencyStatus = document.getElementById('coreLatencyStatus');
  const targetPill = document.getElementById('targetPill');

  const valStt = document.getElementById('valStt');
  const valGuard = document.getElementById('valGuard');
  const valEmbed = document.getElementById('valEmbed');
  const valRetrieve = document.getElementById('valRetrieve');
  const valGate = document.getElementById('valGate');
  const valGen = document.getElementById('valGen');

  const transcriptRow = document.getElementById('transcriptRow');
  const transcriptContent = document.getElementById('transcriptContent');
  const answerBody = document.getElementById('answerBody');
  const tagStatus = document.getElementById('tagStatus');
  const tagGrounded = document.getElementById('tagGrounded');
  const tagConfidence = document.getElementById('tagConfidence');
  const citationsSection = document.getElementById('citationsSection');
  const citationsChips = document.getElementById('citationsChips');
  const contextSection = document.getElementById('contextSection');
  const toggleContextBtn = document.getElementById('toggleContextBtn');
  const contextBody = document.getElementById('contextBody');
  const contextList = document.getElementById('contextList');
  const chunkCountBadge = document.getElementById('chunkCountBadge');

  // Audio Recording State
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let audioContext = null;
  let analyser = null;
  let dataArray = null;
  let animFrameId = null;
  let timerInterval = null;
  let recordingSeconds = 0;

  // ---------------------------------------------------------------------------
  // Canvas Idle & Live Waveform Rendering
  // ---------------------------------------------------------------------------
  function drawIdleWaveform() {
    if (!canvasCtx || !waveformCanvas) return;
    const width = waveformCanvas.width;
    const height = waveformCanvas.height;

    canvasCtx.clearRect(0, 0, width, height);
    canvasCtx.beginPath();
    canvasCtx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
    canvasCtx.lineWidth = 2;

    const midY = height / 2;
    for (let x = 0; x < width; x += 6) {
      canvasCtx.moveTo(x, midY - 2);
      canvasCtx.lineTo(x, midY + 2);
    }
    canvasCtx.stroke();
  }

  function drawLiveWaveform() {
    if (!canvasCtx || !analyser || !dataArray) return;
    animFrameId = requestAnimationFrame(drawLiveWaveform);

    analyser.getByteFrequencyData(dataArray);
    const width = waveformCanvas.width;
    const height = waveformCanvas.height;

    canvasCtx.clearRect(0, 0, width, height);

    const barCount = 48;
    const barWidth = (width / barCount) - 3;
    const midY = height / 2;

    for (let i = 0; i < barCount; i++) {
      const idx = Math.floor((i / barCount) * dataArray.length);
      const val = dataArray[idx] / 255.0;
      const barHeight = Math.max(4, val * (height * 0.85));

      const gradient = canvasCtx.createLinearGradient(0, midY - barHeight / 2, 0, midY + barHeight / 2);
      gradient.addColorStop(0, '#6366f1');
      gradient.addColorStop(1, '#06b6d4');

      canvasCtx.fillStyle = gradient;
      canvasCtx.fillRect(i * (barWidth + 3), midY - barHeight / 2, barWidth, barHeight);
    }
  }

  drawIdleWaveform();

  // ---------------------------------------------------------------------------
  // Audio Recording (MediaRecorder API)
  // ---------------------------------------------------------------------------
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 128;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      dataArray = new Uint8Array(analyser.frequencyBinCount);
      drawLiveWaveform();

      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (animFrameId) cancelAnimationFrame(animFrameId);
        stream.getTracks().forEach(track => track.stop());
        if (audioContext && audioContext.state !== 'closed') audioContext.close();

        drawIdleWaveform();
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        await handleVoiceSubmit(audioBlob);
      };

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add('recording');
      micPrompt.textContent = 'Listening... Click again to process';
      recordingTimer.style.display = 'block';

      recordingSeconds = 0;
      recordingTimer.textContent = '00:00';
      timerInterval = setInterval(() => {
        recordingSeconds++;
        const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, '0');
        const secs = String(recordingSeconds % 60).padStart(2, '0');
        recordingTimer.textContent = `${mins}:${secs}`;
      }, 1000);

    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Microphone access denied or not available. You can still test with text queries!');
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
      micBtn.classList.remove('recording');
      micPrompt.textContent = 'Processing voice query...';
      clearInterval(timerInterval);
      recordingTimer.style.display = 'none';
    }
  }

  if (micBtn) {
    micBtn.addEventListener('click', () => {
      if (!isRecording) {
        startRecording();
      } else {
        stopRecording();
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Submission Handlers (Voice & Text)
  // ---------------------------------------------------------------------------
  async function handleVoiceSubmit(audioBlob) {
    setLoadingState(true);
    const lang = langSelect.value || 'hi-IN';

    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');
    formData.append('language_code', lang);

    try {
      const resp = await fetch('/api/query/voice', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        throw new Error(`Server returned status ${resp.status}`);
      }

      const data = await resp.json();
      renderResponse(data);
    } catch (err) {
      console.error('Voice query failed:', err);
      renderError(err.message);
    } finally {
      setLoadingState(false);
      micPrompt.textContent = 'Click microphone to speak in Hindi or English';
    }
  }

  async function handleTextSubmit(queryText) {
    if (!queryText.trim()) return;
    setLoadingState(true);
    const lang = langSelect.value || 'hi-IN';

    try {
      const resp = await fetch('/api/query/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText.trim(),
          language_code: lang,
        }),
      });

      if (!resp.ok) {
        throw new Error(`Server returned status ${resp.status}`);
      }

      const data = await resp.json();
      renderResponse(data);
    } catch (err) {
      console.error('Text query failed:', err);
      renderError(err.message);
    } finally {
      setLoadingState(false);
    }
  }

  if (queryForm) {
    queryForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const val = queryInput.value;
      if (val) {
        handleTextSubmit(val);
      }
    });
  }

  // Sample Query Chips
  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      const l = chip.getAttribute('data-lang');
      if (langSelect && l) langSelect.value = l;
      if (queryInput) queryInput.value = q;
      handleTextSubmit(q);
    });
  });

  // ---------------------------------------------------------------------------
  // Response & Telemetry Rendering
  // ---------------------------------------------------------------------------
  function setLoadingState(loading) {
    if (sendBtn) sendBtn.disabled = loading;
    if (loading) {
      coreLatencyValue.innerHTML = `<span style="font-size: 1.5rem; opacity: 0.6;">Calculating...</span>`;
      targetPill.className = 'target-pill pill-idle';
      targetPill.textContent = 'PROCESSING';
      coreLatencyStatus.textContent = 'Executing end-to-end pipeline...';
    }
  }

  function renderResponse(data) {
    if (emptyState) emptyState.style.display = 'none';
    if (responseView) responseView.style.display = 'flex';

    // 1. Headline Core RAG Latency (<200ms target)
    const ragCoreMs = data.total_rag_core_ms || 0.0;
    coreLatencyValue.innerHTML = `${ragCoreMs.toFixed(1)}<span class="unit">ms</span>`;

    if (ragCoreMs <= 200.0) {
      targetPill.className = 'target-pill pill-met';
      targetPill.textContent = '✅ <=200ms TARGET MET';
      coreLatencyStatus.textContent = 'Meets ultra-low latency target';
    } else {
      targetPill.className = 'target-pill pill-exceeded';
      targetPill.textContent = '⚠️ >200ms';
      coreLatencyStatus.textContent = 'Exceeds target budget';
    }

    // 2. Granular Telemetry Grid
    const t = data.timings || {};
    if (valStt) valStt.textContent = `${(data.stt_ms || t.stt || 0).toFixed(1)} ms`;
    if (valGuard) valGuard.textContent = `${(t.guardrail || 0).toFixed(2)} ms`;
    if (valEmbed) valEmbed.textContent = `${(t.embed || 0).toFixed(1)} ms`;
    if (valRetrieve) valRetrieve.textContent = `${(t.retrieve || 0).toFixed(1)} ms`;
    if (valGate) valGate.textContent = `${(t.gate || 0).toFixed(2)} ms`;
    if (valGen) valGen.textContent = `${(t.generation || 0).toFixed(1)} ms`;

    // 3. Transcript
    if (data.transcription && data.transcription.trim()) {
      transcriptRow.style.display = 'flex';
      transcriptContent.textContent = `"${data.transcription}"`;
    } else {
      transcriptRow.style.display = 'none';
    }

    // 4. Status & Grounding Tags
    if (data.status === 'guardrail_blocked') {
      tagStatus.className = 'tag tag-blocked';
      tagStatus.textContent = 'BLOCKED (GUARDRAIL)';
    } else if (data.status === 'low_confidence_fallback') {
      tagStatus.className = 'tag tag-fallback';
      tagStatus.textContent = 'LOW CONFIDENCE';
    } else {
      tagStatus.className = 'tag tag-success';
      tagStatus.textContent = 'SUCCESS';
    }

    if (data.grounded) {
      tagGrounded.className = 'tag tag-grounded';
      tagGrounded.textContent = 'GROUNDED IN CORPUS';
    } else {
      tagGrounded.className = 'tag tag-fallback';
      tagGrounded.textContent = 'UNGROUNDED / FALLBACK';
    }

    tagConfidence.textContent = `${(data.confidence || 'MEDIUM').toUpperCase()} CONFIDENCE`;

    // 5. Answer Body
    answerBody.textContent = data.answer || '(No answer returned)';

    // 6. Citations
    if (data.citations && data.citations.length > 0) {
      citationsSection.style.display = 'flex';
      citationsChips.innerHTML = '';
      data.citations.forEach((cid) => {
        const chip = document.createElement('span');
        chip.className = 'citation-chip';
        chip.textContent = `# ${cid}`;
        chip.title = 'Click to highlight source chunk';
        chip.addEventListener('click', () => {
          highlightChunk(cid);
        });
        citationsChips.appendChild(chip);
      });
    } else {
      citationsSection.style.display = 'none';
    }

    // 7. Context Chunks
    const chunks = (data.retrieval_result && data.retrieval_result.chunks) || [];
    if (chunks.length > 0) {
      contextSection.style.display = 'block';
      chunkCountBadge.textContent = chunks.length;
      contextList.innerHTML = '';

      chunks.forEach((sc, idx) => {
        const c = sc.chunk;
        const card = document.createElement('div');
        card.className = 'chunk-card';
        card.id = `chunkCard_${c.chunk_id}`;

        card.innerHTML = `
          <div class="chunk-header">
            <span class="chunk-id">[#${idx + 1}] ${c.chunk_id} <span style="color: var(--accent-cyan);">[${c.chunk_strategy || 'fixed'}]</span></span>
            <span class="chunk-score">RRF: ${sc.score.toFixed(5)}</span>
          </div>
          <div class="chunk-text">${escapeHtml(c.text)}</div>
        `;
        contextList.appendChild(card);
      });
    } else {
      contextSection.style.display = 'none';
    }
  }

  function highlightChunk(chunkId) {
    if (contextBody && contextBody.style.display === 'none') {
      contextBody.style.display = 'block';
      toggleContextBtn.classList.add('open');
    }
    document.querySelectorAll('.chunk-card').forEach((el) => el.classList.remove('highlighted'));
    const target = document.getElementById(`chunkCard_${chunkId}`);
    if (target) {
      target.classList.add('highlighted');
      target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function renderError(message) {
    if (emptyState) emptyState.style.display = 'none';
    if (responseView) responseView.style.display = 'flex';

    coreLatencyValue.innerHTML = `<span style="color: #ef4444;">Error</span>`;
    targetPill.className = 'target-pill pill-exceeded';
    targetPill.textContent = 'FAILED';
    coreLatencyStatus.textContent = 'Request failed';

    tagStatus.className = 'tag tag-blocked';
    tagStatus.textContent = 'ERROR';
    answerBody.textContent = `Error: ${message}`;
  }

  // Toggle Context Chunks
  if (toggleContextBtn && contextBody) {
    toggleContextBtn.addEventListener('click', () => {
      const isHidden = contextBody.style.display === 'none';
      contextBody.style.display = isHidden ? 'block' : 'none';
      toggleContextBtn.classList.toggle('open', isHidden);
    });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
});
