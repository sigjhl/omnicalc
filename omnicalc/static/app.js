/**
 * OmniCalc Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Elements ---
    const DOM = {
        modelSelect: document.getElementById('modelSelect'),
        refreshModelsBtn: document.getElementById('refreshModelsBtn'),
        textInput: document.getElementById('textInput'),
        recordBtn: document.getElementById('recordBtn'),
        captureBtn: document.getElementById('captureBtn'),
        sendBtn: document.getElementById('sendBtn'),
        newSessionBtn: document.getElementById('newSessionBtn'),
        variablesOutput: document.getElementById('variablesOutput'),
        resultOutput: document.getElementById('resultOutput'),
        agentMessage: document.getElementById('agentMessage'),
        errorBox: document.getElementById('errorBox'),

        statusBar: document.getElementById('statusBar'),
        statusText: document.getElementById('statusText'),

        attachmentsArea: document.getElementById('attachmentsArea'),
        attachmentName: document.getElementById('attachmentName'),
        clearAttachmentBtn: document.getElementById('clearAttachmentBtn'),

        settingsBtn: document.getElementById('settingsBtn'),
        settingsModal: document.getElementById('settingsModal'),
        closeSettingsBtn: document.getElementById('closeSettingsBtn'),
        saveSettingsBtn: document.getElementById('saveSettingsBtn'),

        hotkeyInputs: {
            record: document.getElementById('hotkeyRecordInput'),
            capture: document.getElementById('hotkeyCaptureInput'),
            send: document.getElementById('hotkeySendInput')
        },
        hotkeyLabels: {
            record: document.getElementById('hotkeyLabelRecord'),
            capture: document.getElementById('hotkeyLabelCapture'),
            send: document.getElementById('hotkeyLabelSend')
        },

        captureOverlay: document.getElementById('captureOverlay'),
        captureCanvas: document.getElementById('captureCanvas')
    };

    // --- Application State ---
    const State = {
        sessionId: crypto.randomUUID(),
        isRecording: false,
        attachment: null, // {data: base64, type: string}
        hotkeys: {
            record: getStoredHotkey('record', 'Control+r'), // Using standard Ctrl modifier as default
            capture: getStoredHotkey('capture', 'Control+Shift+X'),
            send: getStoredHotkey('send', 'Control+Enter')
        },
        medasr: {
            socket: null,
            audioCtx: null,
            processor: null,
            mediaStream: null
        },
        isProcessing: false
    };

    // --- Initialization ---
    init();

    async function init() {
        updateHotkeyLabels();
        setupEventListeners();
        await refreshModels();
        setStatus('System Ready');
    }

    // --- Core API Interactions ---

    async function apiRequest(endpoint, options = {}) {
        try {
            const res = await fetch(endpoint, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...(options.headers || {})
                }
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
            return data;
        } catch (err) {
            console.error(`API Error (${endpoint}):`, err);
            throw err;
        }
    }

    async function refreshModels() {
        DOM.modelSelect.disabled = true;
        DOM.modelSelect.innerHTML = '<option>Loading...</option>';
        try {
            const data = await apiRequest('/models');
            DOM.modelSelect.innerHTML = '';
            if (!data.models || data.models.length === 0) {
                DOM.modelSelect.innerHTML = '<option value="">No models found</option>';
            } else {
                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = opt.textContent = m;
                    if (data.selected_model === m) opt.selected = true;
                    DOM.modelSelect.appendChild(opt);
                });
            }
        } catch (e) {
            DOM.modelSelect.innerHTML = '<option value="">Error loading models</option>';
            showError("Failed to fetch models: " + e.message);
        } finally {
            DOM.modelSelect.disabled = false;
        }
    }

    async function submitRequest() {
        if (State.isProcessing) return;

        const text = DOM.textInput.value.trim();
        if (!text && !State.attachment) {
            showError('Please enter text or capture an image.');
            return;
        }

        // Fresh session on each explicit run
        State.sessionId = crypto.randomUUID();

        const payload = {
            input: text,
            session_id: State.sessionId,
            model: DOM.modelSelect.value || undefined
        };

        if (State.attachment) {
            payload.attachments = [State.attachment];
        }

        setProcessingState(true);
        setStatus('Processing via Orchestrator...', 'active');
        clearOutputs();
        hideError();

        try {
            const res = await fetch('/orchestrate/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || errData.error || res.statusText);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');

                // Keep the last partial line in the buffer
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6).trim();
                        if (dataStr === '[DONE]') {
                            continue;
                        }
                        try {
                            const event = JSON.parse(dataStr);
                            handleStreamEvent(event);
                        } catch (e) {
                            console.error('Failed to parse SSE JSON', e, dataStr);
                        }
                    }
                }
            }

            // Process any remaining buffer if it didn't end with a newline
            if (buffer.startsWith('data: ')) {
                const dataStr = buffer.substring(6).trim();
                if (dataStr !== '[DONE]' && dataStr.length > 0) {
                    try {
                        const event = JSON.parse(dataStr);
                        handleStreamEvent(event);
                    } catch (e) {
                        console.error('Failed to parse leftover SSE JSON', e, dataStr);
                    }
                }
            }

            setStatus('Ready');
        } catch (err) {
            showError(err.message);
            setStatus('Error processing request', 'error');
        } finally {
            setProcessingState(false);
        }
    }

    function handleStreamEvent(event) {
        // Event types match python enums, e.g., 'extracting_variables', 'calculation_complete', 'assistant_message', 'validation_error', 'error'
        const type = event.type.toLowerCase();
        const data = event.data;

        if (type === 'calculator_selected') {
            setStatus(`Selected calculator: ${data.calc_id}`, 'active');
            DOM.resultOutput.textContent = `Loading ${data.calc_id}...`;
            DOM.resultOutput.classList.remove('empty');
        } else if (type === 'extracting_variables') {
            // Render variables immediately
            const varsObj = data.variables || {};
            // Convert to array format expected by the previous render logic
            const varsArr = Object.entries(varsObj).map(([k, v]) => {
                let val = v;
                let unit = '';
                if (v && typeof v === 'object' && v.value !== undefined) {
                    val = v.value;
                    unit = v.unit || '';
                }
                return { key: k, value: val, unit: unit };
            });
            renderPartial({ variables: varsArr });
        } else if (type === 'calculation_complete') {
            renderPartial({ result: { success: true, outputs: data.outputs } });
        } else if (type === 'validation_error') {
            renderPartial({ result: { success: false, errors: data.errors } });
        } else if (type === 'assistant_message') {
            renderPartial({ assistant_message: data.content });
        } else if (type === 'error') {
            showError(data.error);
        }
    }

    function renderPartial(data) {
        // Variables
        if (data.variables !== undefined) {
            if (data.variables && data.variables.length > 0) {
                const formattedVars = data.variables.map(v => {
                    let text = `${v.key}: ${v.value}`;
                    if (v.unit) text += ` ${v.unit}`;
                    return text;
                }).join('\n');
                DOM.variablesOutput.textContent = formattedVars;
                DOM.variablesOutput.classList.remove('empty');
            } else {
                DOM.variablesOutput.textContent = 'No variables extracted.';
                DOM.variablesOutput.classList.add('empty');
            }
        }

        // Result
        if (data.result !== undefined) {
            if (data.result && data.result.success) {
                const outputs = data.result.outputs || {};
                let formattedText = Object.entries(outputs)
                    .filter(([k, v]) => k !== 'source' && k !== 'confidence')
                    .map(([k, v]) => {
                        return `${k}: ${v}`;
                    }).join('\n');

                // Expose inputs used and unit conversions
                const audit_trace = data.result.audit_trace || {};
                const inputs_used = audit_trace.inputs_used || {};
                if (Object.keys(inputs_used).length > 0) {
                    formattedText += '\n\nVariables Used:\n' + Object.entries(inputs_used)
                        .map(([k, v]) => `${k}: ${v}`)
                        .join('\n');
                }

                DOM.resultOutput.textContent = formattedText || 'No outputs.';
                DOM.resultOutput.classList.remove('empty');
            } else if (data.result && !data.result.success) {
                DOM.resultOutput.textContent = 'Calculation failed:\n' + (data.result.errors ? data.result.errors.join('\n') : 'Unknown error');
                DOM.resultOutput.classList.remove('empty');
            } else {
                DOM.resultOutput.textContent = '–';
                DOM.resultOutput.classList.add('empty');
            }
        }

        // Agent Message
        if (data.assistant_message !== undefined) {
            const msg = data.assistant_message;
            if (msg) {
                DOM.agentMessage.textContent = msg;
                DOM.agentMessage.classList.remove('empty');
            } else {
                DOM.agentMessage.textContent = 'No message';
                DOM.agentMessage.classList.add('empty');
            }
        }

        if (data.errors && data.errors.length > 0) {
            showError(data.errors.join('\\n'));
        }
    }

    function clearOutputs() {
        DOM.variablesOutput.textContent = 'Waiting for input...';
        DOM.variablesOutput.classList.add('empty');
        DOM.resultOutput.textContent = 'Waiting for input...';
        DOM.resultOutput.classList.add('empty');
        DOM.agentMessage.textContent = 'No message';
        DOM.agentMessage.classList.add('empty');
    }

    // --- Audio Math Helpers ---
    function floatTo16BitPCM(floatBuf) {
        const output = new Int16Array(floatBuf.length);
        for (let i = 0; i < floatBuf.length; i++) {
            const s = Math.max(-1, Math.min(1, floatBuf[i]));
            output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return output;
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

    // --- MedASR Integration (WebSockets & TEN-VAD backend) ---

    async function toggleRecording() {
        if (State.isProcessing) return; // Prevent record while sending

        if (State.isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    async function loadMedASR() {
        try {
            setStatus('Loading MedASR backend...', 'active');
            await apiRequest('/transcribe/medasr/load', { method: 'POST' });
            return true;
        } catch (e) {
            showError("Failed to load MedASR: " + e.message);
            setStatus('MedASR unavailable', 'error');
            return false;
        }
    }

    async function startRecording() {
        // Pre-load backend model if needed; api.py ensures it lazily but good to explicit wait
        const loaded = await loadMedASR();
        if (!loaded) return;

        try {
            State.medasr.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
            showError("Mic access denied or unavailable: " + e.message);
            return;
        }

        const AudioContext = window.AudioContext || window.webkitAudioContext;
        State.medasr.audioCtx = new AudioContext();
        const sampleRate = State.medasr.audioCtx.sampleRate;

        const source = State.medasr.audioCtx.createMediaStreamSource(State.medasr.mediaStream);
        State.medasr.processor = State.medasr.audioCtx.createScriptProcessor(4096, 1, 1);

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/transcribe/medasr/stream`;
        State.medasr.socket = new WebSocket(wsUrl);
        State.medasr.socket.binaryType = 'arraybuffer';

        State.medasr.socket.onopen = () => {
            source.connect(State.medasr.processor);
            State.medasr.processor.connect(State.medasr.audioCtx.destination);

            State.isRecording = true;
            DOM.recordBtn.classList.remove('inactive');
            DOM.recordBtn.classList.add('active');
            setStatus('Recording (TEN-VAD Active)...', 'recording');

            // Append space to existing text if needed
            if (DOM.textInput.value && !DOM.textInput.value.endsWith(' ')) {
                DOM.textInput.value += ' ';
            }
        };

        State.medasr.socket.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data);
                if (data.type === 'partial') {
                    // Received a chunk from TEN-VAD
                    const currentPrefix = DOM.textInput.value;
                    const append = data.text || '';
                    if (append) {
                        DOM.textInput.value = currentPrefix + (currentPrefix.endsWith(' ') || !currentPrefix ? '' : ' ') + append + ' ';
                        // Auto-scroll to bottom
                        DOM.textInput.scrollTop = DOM.textInput.scrollHeight;
                    }
                } else if (data.type === 'error') {
                    showError("ASR Error: " + data.error);
                    stopRecording();
                }
            } catch (err) {
                console.error("WebSocket message parse error", err);
            }
        };

        State.medasr.socket.onerror = (e) => {
            console.error('WebSocket Error', e);
            showError("ASR Connection error.");
            stopRecording();
        };

        State.medasr.processor.onaudioprocess = (e) => {
            const floatData = e.inputBuffer.getChannelData(0);
            const downsampled = downsample(floatData, sampleRate, 16000); // Server needs 16k for TEN-VAD
            const pcm16 = floatTo16BitPCM(downsampled);

            if (State.medasr.socket && State.medasr.socket.readyState === WebSocket.OPEN) {
                State.medasr.socket.send(pcm16.buffer);
            }
        };
    }

    function stopRecording() {
        if (!State.isRecording) return;

        State.isRecording = false;
        DOM.recordBtn.classList.remove('active');
        DOM.recordBtn.classList.add('inactive');
        setStatus('System Ready');

        if (State.medasr.processor) { State.medasr.processor.disconnect(); State.medasr.processor = null; }
        if (State.medasr.audioCtx) { State.medasr.audioCtx.close(); State.medasr.audioCtx = null; }
        if (State.medasr.mediaStream) { State.medasr.mediaStream.getTracks().forEach(t => t.stop()); State.medasr.mediaStream = null; }
        if (State.medasr.socket) {
            if (State.medasr.socket.readyState === WebSocket.OPEN) {
                State.medasr.socket.close();
            }
            State.medasr.socket = null;
        }
    }

    // --- Screen Region Capture ---

    async function startScreenCapture() {
        if (State.isProcessing) return;

        try {
            setStatus('Waiting for screen capture selection...', 'recording');

            // Call the new native endpoint
            const data = await apiRequest('/capture/screen', { method: 'POST' });

            if (data.status === 'success' && data.attachment) {
                State.attachment = data.attachment;
                DOM.attachmentName.textContent = State.attachment.name;
                DOM.attachmentsArea.style.display = 'flex'; // Use flex to match chip layout
                setStatus('System Ready');
            } else {
                // Cancelled or no selection
                setStatus('Capture cancelled');
                setTimeout(() => setStatus('System Ready'), 2000);
            }
        } catch (err) {
            console.error("Capture error", err);
            showError("Screen capture failed: " + err.message);
            setStatus('Capture failed', 'error');
        }
    }


    // --- UI State Helpers ---

    function setStatus(text, cname = '') {
        DOM.statusText.textContent = text;
        DOM.statusBar.className = 'status-bar ' + cname;
        DOM.statusBar.classList.remove('hidden');
    }

    function showError(msg) {
        DOM.errorBox.textContent = msg;
        DOM.errorBox.classList.remove('hidden');
    }

    function hideError() {
        DOM.errorBox.textContent = '';
        DOM.errorBox.classList.add('hidden');
    }

    function setProcessingState(isProc) {
        State.isProcessing = isProc;
        DOM.sendBtn.disabled = isProc;
        DOM.recordBtn.disabled = isProc;
        DOM.sendBtn.querySelector('.btn-label').textContent = isProc ? 'Sending...' : 'Send';

        if (isProc) {
            DOM.textInput.style.opacity = '0.7';
        } else {
            DOM.textInput.style.opacity = '1';
        }
    }


    // --- Hotkeys & Settings ---

    function getStoredHotkey(action, defaultKey) {
        return localStorage.getItem(`omnicalc_hk_${action}`) || defaultKey;
    }

    function setStoredHotkey(action, keyCombo) {
        localStorage.setItem(`omnicalc_hk_${action}`, keyCombo);
        State.hotkeys[action] = keyCombo;
        updateHotkeyLabels();
    }

    function updateHotkeyLabels() {
        DOM.hotkeyLabels.record.textContent = formatHotkey(State.hotkeys.record);
        DOM.hotkeyLabels.capture.textContent = formatHotkey(State.hotkeys.capture);
        DOM.hotkeyLabels.send.textContent = formatHotkey(State.hotkeys.send);

        DOM.hotkeyInputs.record.value = State.hotkeys.record;
        DOM.hotkeyInputs.capture.value = State.hotkeys.capture;
        DOM.hotkeyInputs.send.value = State.hotkeys.send;
    }

    function formatHotkey(comboStr) {
        // e.g. Control+Shift+X -> Ctrl+Shift+X
        let s = comboStr.replace('Control', 'Ctrl');
        // Mac specific replacements could map Ctrl to ⌘ visually if desired, but we keep it literal
        return s;
    }

    // Hotkey listener
    window.addEventListener('keydown', (e) => {
        // Don't trigger if typing in text input (except for designated send hotkey which is ok)
        // or if modal is open
        if (!DOM.settingsModal.classList.contains('hidden')) return;

        // Build generic combo string
        const keys = [];
        if (e.ctrlKey || e.metaKey) keys.push('Control'); // Map Meta (Cmd) and Ctrl identically for ease
        if (e.altKey) keys.push('Alt');
        if (e.shiftKey) keys.push('Shift');

        // Don't append if it's just modifier key
        if (!['Control', 'Alt', 'Shift', 'Meta'].includes(e.key)) {
            let k = e.key;
            if (k === ' ') k = 'Space';
            keys.push(k.length === 1 ? k.toUpperCase() : k); // e.g., 'x' -> 'X', 'Enter' -> 'Enter'
        }

        const combo = keys.join('+');
        if (keys.length === 0 || keys.length === 1 && ['Control', 'Alt', 'Shift'].includes(combo)) return;

        let matched = false;

        // Compare ignoring case just in case
        if (combo.toLowerCase() === State.hotkeys.send.toLowerCase()) {
            e.preventDefault();
            submitRequest();
            matched = true;
        } else if (combo.toLowerCase() === State.hotkeys.record.toLowerCase() && document.activeElement !== DOM.textInput) {
            e.preventDefault();
            toggleRecording();
            matched = true;
        } else if (combo.toLowerCase() === State.hotkeys.capture.toLowerCase() && document.activeElement !== DOM.textInput) {
            e.preventDefault();
            startScreenCapture();
            matched = true;
        }
    });

    // Modal behavior & setting recorder
    let activeInputForHotkey = null;

    function hotkeyInputHandler(e) {
        e.preventDefault();
        e.stopPropagation();

        const keys = [];
        // Map Command/Win to Control to simplify mapping strings on all OS
        if (e.ctrlKey || e.metaKey) keys.push('Control');
        if (e.altKey) keys.push('Alt');
        if (e.shiftKey) keys.push('Shift');

        if (!['Control', 'Alt', 'Shift', 'Meta', 'Tab', 'Escape'].includes(e.key)) {
            let k = e.key;
            if (k === ' ') k = 'Space';
            keys.push(k.length === 1 ? k.toUpperCase() : k);
        }

        if (keys.length > 0) {
            e.target.value = keys.join('+');
        }
    }

    Object.values(DOM.hotkeyInputs).forEach(inputEl => {
        inputEl.addEventListener('keydown', hotkeyInputHandler);
        inputEl.addEventListener('focus', (e) => {
            e.target.select();
            e.target.style.borderColor = '#3b82f6';
        });
        inputEl.addEventListener('blur', (e) => {
            e.target.style.borderColor = '';
        });
    });

    // --- Wiring Events ---
    function setupEventListeners() {
        DOM.recordBtn.addEventListener('click', toggleRecording);
        DOM.captureBtn.addEventListener('click', startScreenCapture);
        DOM.sendBtn.addEventListener('click', submitRequest);
        DOM.refreshModelsBtn.addEventListener('click', refreshModels);

        DOM.clearAttachmentBtn.addEventListener('click', () => {
            State.attachment = null;
            DOM.attachmentsArea.style.display = 'none';
        });

        DOM.newSessionBtn.addEventListener('click', () => {
            State.sessionId = crypto.randomUUID();
            DOM.textInput.value = '';
            State.attachment = null;
            DOM.attachmentsArea.style.display = 'none';
            clearOutputs();
            hideError();
            setStatus('New session started');
        });

        DOM.settingsBtn.addEventListener('click', () => {
            DOM.settingsModal.classList.remove('hidden');
        });

        const closeModals = () => {
            DOM.settingsModal.classList.add('hidden');
        };

        DOM.closeSettingsBtn.addEventListener('click', closeModals);
        DOM.settingsModal.addEventListener('click', (e) => {
            if (e.target === DOM.settingsModal) closeModals();
        });

        DOM.saveSettingsBtn.addEventListener('click', () => {
            setStoredHotkey('record', DOM.hotkeyInputs.record.value);
            setStoredHotkey('capture', DOM.hotkeyInputs.capture.value);
            setStoredHotkey('send', DOM.hotkeyInputs.send.value);
            closeModals();
        });
    }

});
