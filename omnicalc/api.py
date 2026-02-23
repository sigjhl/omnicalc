"""FastAPI routes for AgentiCalc."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import base64
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from .models import EventType, OrchestratorRequest, OrchestratorResponse, StreamEvent, UserLocale
from .orchestrator import OrchestratorAgent, create_orchestrator

logger = logging.getLogger(__name__)

# Configuration from environment
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")

# Paths
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]  # /.../agenticalc
PROJECTS_ROOT = HERE.parents[2]  # /.../Projects

# Global orchestrator instance
_orchestrator: Optional[OrchestratorAgent] = None
_asr_loaded: bool = False
_asr_transcribe_fn = None
_asr_backend_info = None
_asr_model_path: Optional[str] = None

from mcp.server.fastmcp import FastMCP
mcp_server = FastMCP("OmniCalc")

@mcp_server.tool()
async def execute_calc(calc_id: str, variables: dict) -> str:
    """Execute a calculation with extracted variables. Returns the calculated result or validation errors."""
    if not _orchestrator:
        return "Error: Orchestrator not initialized"
    import json
    res = await _orchestrator.tools.execute_calc(calc_id=calc_id, variables=variables)
    
    _orchestrator.last_mcp_calc_id = calc_id
    _orchestrator.last_mcp_result = res
    output_str = json.dumps(res.model_dump())
    return f"[TOOL_RESULT]\n{output_str}\n[END_TOOL_RESULT]"

@mcp_server.tool()
async def calc_info(calc_id: str) -> str:
    """Get the input schema for a clinical calculator. Returns field names, types, units, and constraints needed for execute_calc."""
    if not _orchestrator:
        return "Error: Orchestrator not initialized"
    import json
    res = await _orchestrator.tools.calc_info(calc_id=calc_id)
    output_str = json.dumps(res.model_dump())
    return f"[TOOL_RESULT]\n{output_str}\n[END_TOOL_RESULT]"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _orchestrator
    logger.info("Starting OmniCalc API...")
    logger.info(f"LM Studio URL: {LM_STUDIO_URL}")

    try:
        _orchestrator = await create_orchestrator(
            lm_studio_url=LM_STUDIO_URL,
            model=MODEL_NAME,
        )
        logger.info("LM Studio orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LM Studio orchestrator: {e}")

    yield

    # Cleanup
    if _orchestrator:
        await _orchestrator.llm.close()
        await _orchestrator.tools.close()


app = FastAPI(
    title="OmniCalc",
    description="Voice-first clinical scoring agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/mcp", mcp_server.sse_app())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files API
STATIC_DIR = HERE.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class HealthResponse(BaseModel):
    status: str
    lm_studio_available: bool
    loaded_model: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[str]
    selected_model: Optional[str] = None
    online_models: List[str] = []
    local_models: List[str] = []


class MedASRRequest(BaseModel):
    pcm16: str  # base64-encoded int16 PCM mono audio
    sample_rate: int


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and dependencies."""
    if not _orchestrator:
        return HealthResponse(
            status="degraded",
            lm_studio_available=False,
        )

    # Simplified health check since we only use LM Studio
    health_ok = await _orchestrator.llm.health_check()
    
    return HealthResponse(
        status="ok" if health_ok else "degraded",
        lm_studio_available=health_ok,
        loaded_model=_orchestrator.llm.model,
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available LLM models from LM Studio."""
    if not _orchestrator:
        return ModelsResponse(models=[], selected_model=None)

    models = await _orchestrator.llm.list_models()
    return ModelsResponse(
        models=models,
        selected_model=_orchestrator.llm.model,
        local_models=models,
    )


class ModelSelectRequest(BaseModel):
    model: str


@app.post("/models/select")
async def select_model(request: ModelSelectRequest):
    """Select an LLM model to use for future requests."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    models = await _orchestrator.llm.list_models()
    if request.model not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
        
    _orchestrator.llm.model = request.model

    return {"selected_model": request.model}


# ------------- MedASR helpers -------------


def _ensure_asr_loaded():
    """Lazy-load MedASR for server-side transcription."""
    global _asr_loaded, _asr_transcribe_fn, _asr_backend_info, _asr_model_path
    if _asr_loaded:
        return

    from omnicalc.asr import load_asr_model, create_transcriber  # reuse helpers
    from transformers import AutoProcessor

    backend = os.environ.get("MEDASR_BACKEND", "mlx")
    # Both backends default to the HF ID; MLX loader now handles it directly.
    model_path_to_use = os.environ.get("MEDASR_MODEL_PATH", "google/medasr")

    model, backend_info, model_path = load_asr_model(model_path_to_use, backend)
    print(f"DEBUG: Trying to load AutoProcessor from path: {model_path}")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
    
    _asr_transcribe_fn = create_transcriber(model, processor, backend_info)
    _asr_backend_info = backend_info
    _asr_model_path = model_path
    _asr_loaded = True


def _resample_to_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample int16 mono PCM to 16kHz using simple linear interpolation."""
    target_sr = 16000
    if sample_rate == target_sr:
        return pcm
    duration = len(pcm) / sample_rate
    target_len = int(duration * target_sr)
    if target_len <= 0:
        return np.array([], dtype=np.int16)
    x_old = np.linspace(0, duration, num=len(pcm), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    resampled = np.interp(x_new, x_old, pcm.astype(np.float32))
    return resampled.astype(np.int16)


def _transcribe_with_vad(pcm16: np.ndarray, sample_rate: int) -> str:
    """Run VAD-chunked transcription with pre-speech context (~1s)."""
    if pcm16.size == 0:
        return ""

    from ten_vad import TenVad

    vad = TenVad()
    pcm16 = _resample_to_16k(pcm16, sample_rate)
    SR = 16000
    FRAME_SIZE = 256
    PAUSE_THRESHOLD_SEC = 0.5
    MIN_DURATION_SEC = 0.5
    PRE_SPEECH_CONTEXT_FRAMES = int(0.5 * SR / FRAME_SIZE)
    PAUSE_FRAMES = int(PAUSE_THRESHOLD_SEC * SR / FRAME_SIZE)

    vad = TenVad()
    pre_speech_buffer = deque(maxlen=PRE_SPEECH_CONTEXT_FRAMES)
    audio_buffer = []
    silence_counter = 0
    is_speaking = False
    transcript_parts = []

    # Iterate over frames
    total_frames = len(pcm16) // FRAME_SIZE
    for i in range(total_frames):
        frame = pcm16[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        _, is_speech_flag = vad.process(frame)

        if is_speech_flag == 1:
            if not is_speaking:
                is_speaking = True
                audio_buffer.extend(pre_speech_buffer)
                pre_speech_buffer.clear()
            silence_counter = 0
            audio_buffer.append(frame)
        else:
            if is_speaking:
                silence_counter += 1
                audio_buffer.append(frame)
                if silence_counter >= PAUSE_FRAMES:
                    is_speaking = False
                    total_samples = len(audio_buffer) * FRAME_SIZE
                    duration = total_samples / SR
                    if duration >= MIN_DURATION_SEC:
                        text = _asr_transcribe_fn(audio_buffer)
                        if text:
                            transcript_parts.append(text)
                    audio_buffer = []
                    silence_counter = 0
            else:
                pre_speech_buffer.append(frame)

    # Final buffer
    if audio_buffer:
        text = _asr_transcribe_fn(audio_buffer)
        if text:
            transcript_parts.append(text)

    return " ".join(transcript_parts)


@app.websocket("/transcribe/medasr/stream")
async def transcribe_medasr_stream(ws: WebSocket):
    """
    Stream PCM int16 frames over WebSocket for live VAD chunking + transcription.
    Client should send binary frames (int16 mono). Server emits JSON:
      {"type": "partial", "text": "..."} on each VAD-segment transcript
      {"type": "done"} on close
      {"type": "error", "error": "..."} on failure
    """
    await ws.accept()
    try:
        _ensure_asr_loaded()
    except Exception as e:
        await ws.send_json({"type": "error", "error": str(e)})
        await ws.close()
        return

    from ten_vad import TenVad

    SR = 16000
    FRAME_SIZE = 256
    PAUSE_THRESHOLD_SEC = 0.5
    MIN_DURATION_SEC = 0.5
    PRE_SPEECH_CONTEXT_FRAMES = int(0.5 * SR / FRAME_SIZE)
    PAUSE_FRAMES = int(PAUSE_THRESHOLD_SEC * SR / FRAME_SIZE)

    vad = TenVad()
    pre_speech_buffer = deque(maxlen=PRE_SPEECH_CONTEXT_FRAMES)
    audio_buffer = []
    silence_counter = 0
    is_speaking = False

    async def flush_segment():
        nonlocal audio_buffer
        if not audio_buffer:
            return
        text = _asr_transcribe_fn(audio_buffer)
        if text:
            await ws.send_json({"type": "partial", "text": text})
        audio_buffer = []

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] != "websocket.receive":
                continue
            data = msg.get("bytes")
            if not data:
                continue
            pcm = np.frombuffer(data, dtype=np.int16)
            if pcm.size == 0:
                continue
            # Ensure 16k; assume client sends 16k. If not, skip for now.
            if pcm.size % FRAME_SIZE != 0:
                # Trim to nearest frame
                pcm = pcm[: (pcm.size // FRAME_SIZE) * FRAME_SIZE]
            total_frames = pcm.size // FRAME_SIZE
            for i in range(total_frames):
                frame = pcm[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
                _, is_speech_flag = vad.process(frame)
                if is_speech_flag == 1:
                    if not is_speaking:
                        is_speaking = True
                        audio_buffer.extend(pre_speech_buffer)
                        pre_speech_buffer.clear()
                    silence_counter = 0
                    audio_buffer.append(frame)
                else:
                    if is_speaking:
                        silence_counter += 1
                        audio_buffer.append(frame)
                        if silence_counter >= PAUSE_FRAMES:
                            is_speaking = False
                            total_samples = len(audio_buffer) * FRAME_SIZE
                            duration = total_samples / SR
                            if duration >= MIN_DURATION_SEC:
                                await flush_segment()
                            audio_buffer = []
                            silence_counter = 0
                    else:
                        pre_speech_buffer.append(frame)
    except Exception as e:
        await ws.send_json({"type": "error", "error": str(e)})
    finally:
        # Final segment
        if audio_buffer:
            try:
                await flush_segment()
            except Exception:
                pass
        if ws.application_state == WebSocketState.CONNECTED:
            try:
                await ws.send_json({"type": "done"})
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass


@app.post("/transcribe/medasr")
async def transcribe_medasr(request: MedASRRequest):
    """Transcribe PCM audio using MedASR + VAD chunking."""
    try:
        raw = base64.b64decode(request.pcm16)
        pcm = np.frombuffer(raw, dtype=np.int16)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {e}")

    try:
        _ensure_asr_loaded()
        transcript = _transcribe_with_vad(pcm, request.sample_rate)
        return {"transcript": transcript}
    except Exception as e:
        logger.exception("MedASR transcription error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/medasr/load")
async def load_medasr():
    """Preload MedASR weights so UI toggle is instant."""
    try:
        _ensure_asr_loaded()
        return {"status": "loaded", "backend": _asr_backend_info, "model_path": _asr_model_path}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("MedASR load error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orchestrate", response_model=OrchestratorResponse)
async def orchestrate(req: Request, request: OrchestratorRequest):
    """
    Main endpoint for clinical calculation.

    Accepts text input (transcript or typed) and returns calculation results
    or clarification requests.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    if not request.mcp_url:
        request.mcp_url = f"http://{req.headers.get('host', 'localhost:8002')}/mcp/sse"

    try:
        response = await _orchestrator.process(request)
        return response
    except Exception as e:
        logger.exception("Orchestration error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orchestrate/stream")
async def orchestrate_stream(req: Request, request: OrchestratorRequest):
    """
    Streaming endpoint for clinical calculation.

    Returns Server-Sent Events (SSE) for real-time UI updates.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    if not request.mcp_url:
        request.mcp_url = f"http://{req.headers.get('host', 'localhost:8002')}/mcp/sse"

    async def event_generator():
        try:
            async for event in _orchestrator.process_stream(request):
                yield {"data": event.model_dump_json()}
        except Exception as e:
            logger.exception("Streaming error")
            error_event = StreamEvent(
                type=EventType.ERROR,
                data={"error": str(e)},
            )
            yield {"data": error_event.model_dump_json()}
        finally:
            yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


class SessionRequest(BaseModel):
    session_id: str


@app.post("/session/clear")
async def clear_session(request: SessionRequest):
    """Clear a conversation session."""
    if _orchestrator:
        _orchestrator.clear_session(request.session_id)
    return {"status": "ok"}


import tempfile

@app.post("/capture/screen")
async def capture_screen():
    """Trigger native macOS screen capture (interactive region)."""
    if sys.platform != "darwin":
        raise HTTPException(status_code=400, detail="Native capture only supported on macOS.")
        
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Run screencapture interactively (-i). This blocks until user finishes drawing.
        process = await asyncio.create_subprocess_exec(
            "screencapture", "-i", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
             return {"status": "cancelled"}
             
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            
        return {
            "status": "success",
            "attachment": {
                "kind": "image",
                "mime_type": "image/png",
                "data": b64,
                "name": "Screenshot.png"
            }
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class TextInputRequest(BaseModel):
    """Simple text input for quick testing."""
    text: str
    calculator: Optional[str] = None
    model: Optional[str] = None


@app.post("/calculate")
async def quick_calculate(request: TextInputRequest):
    """
    Quick calculation endpoint for testing.

    Simpler interface than /orchestrate - just pass text and optionally a calculator.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        orch_request = OrchestratorRequest(
            input=request.text,
            calculator_hint=request.calculator,
            model=request.model,
        )
        response = await _orchestrator.process(orch_request)
        return response
    except Exception as e:
        logger.exception("Calculation error")
        raise HTTPException(status_code=500, detail=str(e))


DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OmniCalc Demo</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --accent: #22c55e;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --border: #1f2937;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 24px;
      font-family: "Inter", system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at 20% 20%, #1f2937, #0f172a 45%),
                  radial-gradient(circle at 80% 0%, #111827, #0f172a 40%),
                  #0f172a;
      color: var(--text);
      min-height: 100vh;
    }
    h1 { margin: 0 0 8px 0; letter-spacing: -0.5px; }
    .subhead { color: var(--muted); margin-bottom: 16px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }
    textarea {
      width: 100%;
      min-height: 140px;
      background: #0b1220;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      font-size: 14px;
      resize: vertical;
    }
    label { font-weight: 600; display: block; margin-bottom: 6px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    select, button {
      background: #0b1220;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
    }
    button.primary {
      background: linear-gradient(135deg, #22c55e, #16a34a);
      color: #051406;
      border: none;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary { cursor: pointer; }
    .status { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .pill {
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--border);
      background: #0b1220;
      color: var(--muted);
    }
    .pill.ok { color: #22c55e; border-color: #1d3f2b; background: #0d1a12; }
    .pill.bad { color: #f87171; border-color: #3f1d1d; background: #1a0d0d; }
    pre {
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      color: var(--muted);
    }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
    .small { font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="panel">
    <h1>OmniCalc Demo</h1>
    <div class="subhead">Enter or paste your transcript. Runs through the orchestrator & CalcSpec.</div>
    <div class="status" id="status-row"></div>

    <form id="calc-form">
      <div class="row" style="align-items: center; margin-bottom: 12px;">
        <div>
          <label for="model">LLM Model</label>
          <div class="row" style="align-items:center; gap:8px;">
            <select id="model"></select>
            <button type="button" class="secondary" id="refresh-models">Refresh</button>
          </div>
          <div class="small" id="model-note"></div>
        </div>
        <div>
          <label for="calc-hint">Calculator hint (optional)</label>
          <input id="calc-hint" style="width: 220px; background:#0b1220; color:var(--text); border:1px solid var(--border); border-radius:10px; padding:10px 12px; font-size:14px;" placeholder="e.g. meld_na or wells_dvt"/>
        </div>
      </div>

      <label for="input">Transcript / text</label>
      <textarea id="input" placeholder="Describe the clinical scenario..."></textarea>
      <div class="row" style="margin-top:8px; align-items:center; gap:10px;">
        <button type="button" class="secondary" id="voice-btn">🎙️ Start voice (Web Speech)</button>
        <button type="button" class="secondary" id="mode-toggle">Mode: Web Speech</button>
        <span class="small" id="voice-status">Mic idle.</span>
      </div>

      <div class="row" style="margin-top:10px; align-items:center; gap:10px;">
        <input id="file-input" type="file" accept="image/*,audio/*" multiple style="display:none;" />
        <button type="button" class="secondary" id="file-btn">➕ Add file</button>
        <button type="button" class="secondary" id="clear-files">Clear files</button>
        <span class="small" id="file-status">No files attached.</span>
      </div>

      <div class="row" style="margin-top:12px; gap:10px;">
        <button class="primary" type="submit">Run</button>
        <button type="button" class="secondary" id="clear-session">New session</button>
      </div>
    </form>

    <div class="grid" style="margin-top:16px;">
      <div>
        <label>Outputs</label>
        <pre id="outputs">–</pre>
      </div>
      <div>
        <label>Extracted variables</label>
        <pre id="variables">–</pre>
      </div>
    </div>
    <div style="margin-top:12px;">
      <label>Assistant message</label>
      <pre id="assistant">–</pre>
    </div>
    <div style="margin-top:12px;">
      <label>Errors</label>
      <pre id="errors">–</pre>
    </div>
  </div>

  <script>
    const statusRow = document.getElementById('status-row');
    const modelSelect = document.getElementById('model');
    const modelNote = document.getElementById('model-note');
    const inputEl = document.getElementById('input');
    const outputsEl = document.getElementById('outputs');
    const varsEl = document.getElementById('variables');
    const assistantEl = document.getElementById('assistant');
    const errorsEl = document.getElementById('errors');
    const calcHintEl = document.getElementById('calc-hint');
    const refreshBtn = document.getElementById('refresh-models');
    const clearBtn = document.getElementById('clear-session');
    const form = document.getElementById('calc-form');
    const voiceBtn = document.getElementById('voice-btn');
    const modeToggle = document.getElementById('mode-toggle');
    const voiceStatus = document.getElementById('voice-status');
    const fileInput = document.getElementById('file-input');
    const fileBtn = document.getElementById('file-btn');
    const clearFilesBtn = document.getElementById('clear-files');
    const fileStatus = document.getElementById('file-status');

    let sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
    let recognizer = null;
    let voiceActive = false;
    let finalText = '';
    let interimText = '';
    let speechMode = 'web'; // 'web' | 'medasr'
    let mediaStream = null;
    let audioCtx = null;
    let processor = null;
    let medasrBuffers = [];
    let medasrSampleRate = 16000;
    let medasrSocket = null;
    let attachments = [];
    const MAX_FILE_BYTES = 15 * 1024 * 1024;
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition || null;

    function pill(label, ok=true) {
      const cls = ok ? 'pill ok' : 'pill bad';
      return `<span class="${cls}">${label}</span>`;
    }

    async function fetchJSON(url, options) {
      const res = await fetch(url, options);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      return res.json();
    }

    async function refreshHealth() {
      try {
        const health = await fetchJSON('/health');
        const pills = [
          pill(health.lm_studio_available ? 'LM Studio: OK' : 'LM Studio: missing', health.lm_studio_available),
          pill(health.gemini_available ? 'Gemini: OK' : 'Gemini: missing', health.gemini_available),
          pill(health.calcspec_available ? 'CalcSpec: OK' : 'CalcSpec: missing', health.calcspec_available),
          pill('Status: ' + health.status, health.status === 'ok')
        ];
        statusRow.innerHTML = pills.join('');
      } catch (e) {
        statusRow.innerHTML = pill('Health check failed', false);
      }
    }

    async function refreshModels() {
      modelSelect.innerHTML = '<option>Loading...</option>';
      try {
        const data = await fetchJSON('/models');
        modelSelect.innerHTML = '';
        if (!data.models || data.models.length === 0) {
          modelSelect.innerHTML = '<option value="">No models found</option>';
          modelNote.textContent = 'Load a local model in LM Studio or set GEMINI_API_KEY for online models.';
          return;
        }
        for (const m of data.models) {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          if (data.selected_model === m) opt.selected = true;
          modelSelect.appendChild(opt);
        }
        modelSelect.dataset.onlineModels = JSON.stringify(data.online_models || []);
        const onlineSet = new Set(data.online_models || []);
        if (onlineSet.has(modelSelect.value)) {
          modelNote.textContent = 'Online Gemini (google-genai) — multimodal enabled.';
        } else {
          modelNote.textContent = 'Local model from LM Studio.';
        }
      } catch (e) {
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        modelNote.textContent = e.message;
      }
    }

    function renderResponse(data) {
      outputsEl.textContent = data.result && data.result.outputs ? JSON.stringify(data.result.outputs, null, 2) : '–';
      varsEl.textContent = data.variables && data.variables.length ? JSON.stringify(data.variables, null, 2) : '–';
      assistantEl.textContent = data.assistant_message || data.clarification_question || '–';
      errorsEl.textContent = data.errors && data.errors.length ? data.errors.join('\\n') : '–';
    }

    function downsample(buffer, inRate, outRate) {
      if (outRate === inRate) return buffer;
      const ratio = inRate / outRate;
      const newLen = Math.round(buffer.length / ratio);
      const result = new Float32Array(newLen);
      let offsetResult = 0;
      let offsetBuffer = 0;
      while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
          accum += buffer[i];
          count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
      }
      return result;
    }

    function floatTo16BitPCM(floatBuf) {
      const output = new Int16Array(floatBuf.length);
      for (let i = 0; i < floatBuf.length; i++) {
        const s = Math.max(-1, Math.min(1, floatBuf[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return output;
    }

    function base64FromArrayBuffer(buffer) {
      let binary = '';
      const bytes = new Uint8Array(buffer);
      const len = bytes.byteLength;
      for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      return btoa(binary);
    }

    function updateFileStatus() {
      if (!attachments.length) {
        fileStatus.textContent = 'No files attached.';
        return;
      }
      const names = attachments.map(att => att.name || att.kind).join(', ');
      fileStatus.textContent = `${attachments.length} file(s): ${names}`;
    }

    function readFileAsDataURL(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }

    async function addFiles(fileList) {
      for (const file of fileList) {
        if (file.size > MAX_FILE_BYTES) {
          alert(`File too large (${file.name}). Max 15 MB.`);
          continue;
        }
        const kind = file.type.startsWith('image/') ? 'image' : (file.type.startsWith('audio/') ? 'audio' : null);
        if (!kind) {
          alert(`Unsupported file type for ${file.name}.`);
          continue;
        }
        const dataUrl = await readFileAsDataURL(file);
        const parts = String(dataUrl).split(',');
        if (parts.length < 2) continue;
        attachments.push({
          name: file.name,
          kind,
          mime_type: file.type || 'application/octet-stream',
          data: parts[1],
        });
      }
      updateFileStatus();
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = inputEl.value.trim();
      if (!text && attachments.length === 0) {
        alert('Please enter text or attach a file.');
        return;
      }
      // Fresh session on each Run (future: add checkbox to continue session)
      sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
      const body = {
        input: text || '',
        session_id: sessionId,
        calculator_hint: calcHintEl.value || undefined,
        model: modelSelect.value || undefined,
        attachments: attachments.length ? attachments : undefined
      };
      outputsEl.textContent = 'Running...';
      varsEl.textContent = 'Running...';
      assistantEl.textContent = 'Running...';
      errorsEl.textContent = '–';
      try {
        const data = await fetchJSON('/orchestrate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        renderResponse(data);
      } catch (err) {
        errorsEl.textContent = err.message;
      }
    });

    refreshBtn.addEventListener('click', refreshModels);
    modelSelect.addEventListener('change', () => {
      const online = JSON.parse(modelSelect.dataset.onlineModels || '[]');
      if (online.includes(modelSelect.value)) {
        modelNote.textContent = 'Online Gemini (google-genai) — multimodal enabled.';
      } else {
        modelNote.textContent = 'Local model from LM Studio.';
      }
    });
    fileBtn.addEventListener('click', () => fileInput.click());
    clearFilesBtn.addEventListener('click', () => {
      attachments = [];
      updateFileStatus();
    });
    fileInput.addEventListener('change', async () => {
      if (fileInput.files && fileInput.files.length) {
        await addFiles(Array.from(fileInput.files));
      }
      fileInput.value = '';
    });
    inputEl.addEventListener('paste', async (e) => {
      const items = (e.clipboardData && e.clipboardData.items) ? Array.from(e.clipboardData.items) : [];
      const files = [];
      for (const item of items) {
        if (item.type && item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) files.push(file);
        }
      }
      if (files.length) {
        e.preventDefault();
        await addFiles(files);
      }
    });
    clearBtn.addEventListener('click', async () => {
      sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
      await fetch('/session/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      outputsEl.textContent = varsEl.textContent = assistantEl.textContent = '–';
      errorsEl.textContent = '–';
      inputEl.value = '';
      attachments = [];
      updateFileStatus();
    });

    function initVoice() {
      if (Recognition) {
        recognizer = new Recognition();
        recognizer.lang = 'en-US';
        recognizer.continuous = true;
        recognizer.interimResults = true;

        recognizer.onstart = () => {
          voiceStatus.textContent = 'Listening...';
          voiceBtn.textContent = '■ Stop voice (Web)';
          const existing = inputEl.value.trim();
          finalText = existing ? existing + ' ' : '';
          interimText = '';
        };
        recognizer.onresult = (event) => {
          interimText = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const res = event.results[i];
            if (res.isFinal) {
              finalText += res[0].transcript + ' ';
            } else {
              interimText += res[0].transcript;
            }
          }
          inputEl.value = (finalText + interimText).trim();
        };
        recognizer.onerror = (e) => {
          voiceStatus.textContent = 'Mic error: ' + e.error;
          voiceActive = false;
          voiceBtn.textContent = '🎙️ Start voice (Web Speech)';
        };
        recognizer.onend = () => {
          interimText = '';
          voiceStatus.textContent = voiceActive ? 'Restarting...' : 'Mic stopped.';
          if (voiceActive && speechMode === 'web') {
            try { recognizer.start(); } catch (err) { voiceStatus.textContent = 'Mic error: ' + err.message; voiceActive = false; voiceBtn.textContent = '🎙️ Start voice (Web Speech)'; }
          } else {
            voiceBtn.textContent = '🎙️ Start voice (Web Speech)';
          }
        };
      } else {
        voiceBtn.disabled = true;
        voiceStatus.textContent = 'Speech recognition not supported in this browser.';
      }
    }

    voiceBtn.addEventListener('click', () => {
      if (speechMode === 'web') {
        if (!Recognition) return;
        if (!voiceActive) {
          voiceActive = true;
          voiceStatus.textContent = 'Starting mic...';
          try { recognizer.start(); } catch (err) { voiceStatus.textContent = 'Cannot start mic: ' + err.message; voiceActive = false; }
        } else {
          voiceActive = false;
          voiceStatus.textContent = 'Stopping...';
          try { recognizer.stop(); } catch (err) { voiceStatus.textContent = 'Mic error: ' + err.message; }
        }
      } else {
        if (!voiceActive) {
          startMedasrRecording();
        } else {
          stopMedasrRecording();
        }
      }
    });

    modeToggle.addEventListener('click', () => {
      speechMode = speechMode === 'web' ? 'medasr' : 'web';
      voiceBtn.textContent = speechMode === 'web' ? '🎙️ Start voice (Web Speech)' : '🎙️ Start voice (MedASR)';
      modeToggle.textContent = 'Mode: ' + (speechMode === 'web' ? 'Web Speech' : 'MedASR');
      voiceStatus.textContent = 'Mic idle.';
      voiceActive = false;
      if (recognizer) { try { recognizer.stop(); } catch(e) {} }
      if (processor) { processor.disconnect(); processor = null; }
      if (audioCtx) { audioCtx.close(); audioCtx = null; }
      if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
      medasrBuffers = [];
      interimText = '';

      if (speechMode === 'medasr') {
        voiceStatus.textContent = 'Loading MedASR...';
        fetchJSON('/transcribe/medasr/load', { method: 'POST' })
          .then(() => voiceStatus.textContent = 'MedASR ready. Mic idle.')
          .catch(err => {
            voiceStatus.textContent = 'MedASR load failed: ' + err.message;
            speechMode = 'web';
            voiceBtn.textContent = '🎙️ Start voice (Web Speech)';
            modeToggle.textContent = 'Mode: Web Speech';
          });
      }
    });

    async function startMedasrRecording() {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (e) {
        voiceStatus.textContent = 'Mic blocked: ' + e.message;
        return;
      }
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      medasrSampleRate = audioCtx.sampleRate;
      const source = audioCtx.createMediaStreamSource(mediaStream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);

      medasrSocket = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/transcribe/medasr/stream');
      medasrSocket.binaryType = 'arraybuffer';
      medasrSocket.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'partial') {
            const existing = inputEl.value.trim();
            const append = data.text || '';
            inputEl.value = append ? (existing ? existing + ' ' + append : append) : existing;
          }
          if (data.type === 'error') {
            voiceStatus.textContent = 'MedASR error: ' + data.error;
          }
        } catch (err) {
          console.error('MedASR message error', err);
        }
      };

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Downsample to 16k on the fly
        const down = downsample(inputData, medasrSampleRate, 16000);
        const pcm16 = floatTo16BitPCM(down);
        if (medasrSocket && medasrSocket.readyState === WebSocket.OPEN) {
          medasrSocket.send(pcm16.buffer);
        }
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
      voiceActive = true;
      voiceStatus.textContent = `Recording (MedASR stream @ 16k)...`;
      voiceBtn.textContent = '■ Stop voice (MedASR)';
    }

    async function stopMedasrRecording() {
      voiceActive = false;
      voiceStatus.textContent = 'Stopping...';
      if (processor) { processor.disconnect(); processor = null; }
      if (audioCtx) { audioCtx.close(); audioCtx = null; }
      if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
      if (medasrSocket && medasrSocket.readyState === WebSocket.OPEN) {
        medasrSocket.close();
      }
      medasrSocket = null;
      voiceBtn.textContent = '🎙️ Start voice (MedASR)';
      voiceStatus.textContent = 'Mic idle.';
    }

    (async function init() {
      await refreshHealth();
      await refreshModels();
      initVoice();
      updateFileStatus();
    })();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def demo_page():
    """Serve a simple in-browser demo UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=DEMO_HTML)


# WebSocket endpoint for real-time interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional communication.

    Protocol:
    - Client sends: {"type": "input", "text": "...", "session_id": "..."}
    - Server sends: StreamEvent objects as JSON
    """
    await websocket.accept()

    if not _orchestrator:
        await websocket.send_json({"type": "error", "error": "Orchestrator not initialized"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "input":
                text = data.get("text", "")
                session_id = data.get("session_id")
                calculator = data.get("calculator")

                orch_req = OrchestratorRequest(
                    input=text or "",
                    session_id=session_id,
                    calculator_hint=calculator,
                )

                async for event in _orchestrator.process_stream(orch_req):
                    await websocket.send_json(event.model_dump())

                await websocket.send_json({"type": "done"})

            elif msg_type == "clear_session":
                session_id = data.get("session_id")
                if session_id:
                    _orchestrator.clear_session(session_id)
                await websocket.send_json({"type": "session_cleared"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


def run_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the AgentiCalc server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
