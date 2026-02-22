/**
 * OmniCalc Frontend Logic - Mobile-First Chat UI
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

        chatContainer: document.getElementById('chatContainer'),
        welcomeMessage: document.getElementById('welcomeMessage'),

        statusIndicator: document.getElementById('statusIndicator'),

        attachmentsArea: document.getElementById('attachmentsArea'),
        attachmentName: document.getElementById('attachmentName'),
        clearAttachmentBtn: document.getElementById('clearAttachmentBtn'),

        settingsBtn: document.getElementById('settingsBtn'),
        settingsModal: document.getElementById('settingsModal'),
        closeSettingsBtn: document.getElementById('closeSettingsBtn'),
        saveSettingsBtn: document.getElementById('saveSettingsBtn'),
        hotkeySelect: document.getElementById('hotkeySelect'),
        asrHotkeyInput: document.getElementById('asrHotkeyInput'),
        captureHotkeyInput: document.getElementById('captureHotkeyInput'),
        clearAsrHotkeyBtn: document.getElementById('clearAsrHotkeyBtn'),
        clearCaptureHotkeyBtn: document.getElementById('clearCaptureHotkeyBtn'),

        captureOverlay: document.getElementById('captureOverlay'),
        captureCanvas: document.getElementById('captureCanvas')
    };

    // --- Application State ---
    const State = {
        sessionId: crypto.randomUUID(),
        isRecording: false,
        attachment: null, // {data: base64, type: string}
        medasr: {
            socket: null,
            audioCtx: null,
            processor: null,
            mediaStream: null
        },
        isProcessing: false,

        // Chat state tracking to append chunks to the same bubble
        currentAssistantBubble: null,
        typingIndicator: null,
        sendHotkey: localStorage.getItem('omnicalc_send_hotkey') || 'enter',
        asrHotkey: localStorage.getItem('omnicalc_asr_hotkey') || 'Cmd+R',
        captureHotkey: localStorage.getItem('omnicalc_capture_hotkey') || 'Cmd+Shift+X'
    };

    // --- Initialization ---
    init();

    async function init() {
        setupEventListeners();
        await refreshModels();
        setStatus('ready', 'System Ready');
        adjustTextareaHeight();

        try {
            const savedModel = localStorage.getItem('omnicalc_model');
            if (savedModel) {
                // If we want to restore saved model...
            }
            if (DOM.hotkeySelect) DOM.hotkeySelect.value = State.sendHotkey;
            if (DOM.asrHotkeyInput) DOM.asrHotkeyInput.value = State.asrHotkey;
            if (DOM.captureHotkeyInput) DOM.captureHotkeyInput.value = State.captureHotkey;
        } catch (e) {
            console.error('Storage access error', e);
        }
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
            showErrorToast("Failed to fetch models: " + e.message);
        } finally {
            DOM.modelSelect.disabled = false;
        }
    }

    async function submitRequest() {
        if (State.isProcessing) return;

        if (State.isRecording) {
            stopRecording();
        }

        const text = DOM.textInput.value.trim();
        if (!text && !State.attachment) {
            return; // don't send empty
        }

        // Generate persistent session ID if not set
        if (!State.sessionId) {
            State.sessionId = crypto.randomUUID();
        }

        const payload = {
            input: text,
            session_id: State.sessionId,
            model: DOM.modelSelect.value || undefined
        };

        if (State.attachment) {
            payload.attachments = [State.attachment];
        }

        // Hide welcome message
        if (DOM.welcomeMessage) {
            DOM.welcomeMessage.style.display = 'none';
        }

        // Render User Message
        appendUserMessage(text, State.attachment);

        // Clear input
        DOM.textInput.value = '';
        State.attachment = null;
        DOM.attachmentsArea.style.display = 'none';
        adjustTextareaHeight();

        setProcessingState(true);
        setStatus('processing', 'Processing...');

        // Prepare Assistant Bubble
        startAssistantResponse();

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

                buffer = lines.pop(); // Keep last partial line

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6).trim();
                        if (dataStr === '[DONE]') continue;
                        try {
                            const event = JSON.parse(dataStr);
                            handleStreamEvent(event);
                        } catch (e) {
                            console.error('Failed to parse SSE JSON', e, dataStr);
                        }
                    }
                }
            }

            if (buffer.startsWith('data: ')) {
                const dataStr = buffer.substring(6).trim();
                if (dataStr !== '[DONE]' && dataStr.length > 0) {
                    try {
                        const event = JSON.parse(dataStr);
                        handleStreamEvent(event);
                    } catch (e) { }
                }
            }

            // Cleanup typing indicator if it's still there
            removeTypingIndicator();
            setStatus('ready', 'System Ready');
        } catch (err) {
            removeTypingIndicator();
            appendErrorMessage(err.message);
            setStatus('error', 'Error');
        } finally {
            setProcessingState(false);
            State.currentAssistantBubble = null;
        }
    }

    // --- Chat DOM Helpers ---

    function scrollToBottom() {
        DOM.chatContainer.scrollTop = DOM.chatContainer.scrollHeight;
    }

    function appendUserMessage(text, attachment) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';

        let contentHtml = '';
        if (attachment) {
            contentHtml += `<div style="font-size:0.8rem; margin-bottom:4px; opacity:0.8;">&#128206; ${attachment.name}</div>`;
        }
        if (text) {
            // Very simple escape
            const safeText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            contentHtml += `<div>${safeText.replace(/\\n/g, '<br/>')}</div>`;
        } else {
            contentHtml += `<div><em>[Image attached]</em></div>`;
        }
        bubble.innerHTML = contentHtml;
        msgDiv.appendChild(bubble);
        DOM.chatContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    function startAssistantResponse() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.id = 'typingIndicatorWrapper';
        bubble.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;

        msgDiv.appendChild(bubble);
        DOM.chatContainer.appendChild(msgDiv);
        scrollToBottom();

        State.currentAssistantBubble = bubble;
        State.typingIndicator = bubble.querySelector('.typing-indicator');
    }

    function removeTypingIndicator() {
        if (State.typingIndicator && State.typingIndicator.parentNode) {
            const bubble = State.typingIndicator.parentNode;
            bubble.removeChild(State.typingIndicator);
            State.typingIndicator = null;

            // If the bubble is now empty (ignoring whitespace), remove the whole message div
            if (bubble.innerHTML.trim() === '') {
                const msgDiv = bubble.closest('.message');
                if (msgDiv && msgDiv.parentNode) {
                    msgDiv.parentNode.removeChild(msgDiv);
                }
                if (State.currentAssistantBubble === bubble) {
                    State.currentAssistantBubble = null;
                }
            }
        }
    }

    function appendAssistantText(text) {
        removeTypingIndicator();
        if (!State.currentAssistantBubble) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message assistant';
            State.currentAssistantBubble = document.createElement('div');
            State.currentAssistantBubble.className = 'msg-bubble';
            msgDiv.appendChild(State.currentAssistantBubble);
            DOM.chatContainer.appendChild(msgDiv);
        }

        const safeText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");

        State.currentAssistantBubble.innerHTML += safeText.replace(/\\n/g, '<br/>');

        let currentText = State.currentAssistantBubble.innerText.trim();
        if (/^Done\.?$/i.test(currentText) || currentText === '.') {
            const msgDiv = State.currentAssistantBubble.closest('.message');
            if (msgDiv) msgDiv.style.display = 'none';
        } else {
            const msgDiv = State.currentAssistantBubble.closest('.message');
            if (msgDiv) {
                msgDiv.style.display = '';

                // Dynamically upgrade to a Card layout if it's currently a simple msg-bubble
                if (State.currentAssistantBubble.classList.contains('msg-bubble')) {
                    const originalHTML = State.currentAssistantBubble.innerHTML;

                    const card = document.createElement('div');
                    card.className = 'card';
                    card.style.margin = '0';
                    card.innerHTML = `
                        <div class="card-title">
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                            OmniCalc
                        </div>
                        <div class="card-content" style="padding-top: 8px; color: var(--text-primary); line-height: 1.5;">
                        </div>
                    `;

                    const cardContent = card.querySelector('.card-content');
                    cardContent.innerHTML = originalHTML;

                    // Replace the inner bubble with the new card
                    msgDiv.innerHTML = '';
                    msgDiv.appendChild(card);

                    // Retarget the pointer so subsequent appends go inside the card
                    State.currentAssistantBubble = cardContent;
                }
            }
        }

        scrollToBottom();
    }

    function appendSystemCard(cardHtml) {
        removeTypingIndicator();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system';
        msgDiv.innerHTML = cardHtml;
        DOM.chatContainer.appendChild(msgDiv);
        scrollToBottom();

        // Re-inject typing indicator to the bottom if processing
        if (State.isProcessing) {
            startAssistantResponse();
        }
    }

    function appendErrorMessage(text) {
        removeTypingIndicator();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system';
        msgDiv.innerHTML = `<div class="message error-text">${text}</div>`;
        DOM.chatContainer.appendChild(msgDiv);
        // Assuming closeModal() is defined elsewhere or should be added.
        // If not defined, this will cause an error.
        // For now, adding it as requested.
        closeModal(); // This line was added based on the instruction.
    }

    // Chat Container Event Delegation
    DOM.chatContainer.addEventListener('click', (e) => {
        const btn = e.target.closest('.copy-btn');
        if (!btn) return;

        try {
            const type = btn.dataset.copyType;
            const data = JSON.parse(decodeURIComponent(btn.dataset.copyData));
            let text = "";

            if (type === 'vars') {
                text = "Variables:\n";
                text += "--------------------------------------\n";
                data.forEach(v => {
                    const name = v.label || formatTitleCase(v.key);
                    const padding = " ".repeat(Math.max(0, 20 - name.length));
                    text += `${name}${padding} | ${v.value} ${v.unit || ''}\n`;
                });
                text += "--------------------------------------\n";
            } else if (type === 'result') {
                text = "Calculation Result:\n";
                text += "--------------------------------------\n";
                Object.entries(data).forEach(([k, v]) => {
                    const name = formatTitleCase(k);
                    const padding = " ".repeat(Math.max(0, 20 - name.length));
                    text += `${name}${padding} | ${v}\n`;
                });
                text += "--------------------------------------\n";
            }

            navigator.clipboard.writeText(text).then(() => {
                const originalHtml = btn.innerHTML;
                btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
                btn.style.color = "var(--accent-success)";
                setTimeout(() => {
                    btn.innerHTML = originalHtml;
                    btn.style.color = "var(--text-secondary)";
                }, 2000);
            });
        } catch (err) {
            console.error("Failed to copy", err);
        }
    });

    // --- Stream Handlers ---

    function handleStreamEvent(event) {
        const type = event.type.toLowerCase();
        const data = event.data;
        console.log("STREAM EVENT:", type, data);

        if (['extracting_variables', 'calculation_complete', 'validation_error', 'error'].includes(type)) {
            const loader = document.getElementById('loading-indicator');
            if (loader) {
                const msg = loader.closest('.message.system');
                if (msg && msg.parentNode) msg.parentNode.removeChild(msg);
            }
        }

        if (type === 'calculator_selected') {
            const calcId = data.calc_id || 'calculator';
            const card = buildLoadingCard(calcId);
            appendSystemCard(card);
        } else if (type === 'extracting_variables') {
            const varsArr = dictToArr(data.variables || {});
            const card = buildVariablesCard(data.calc_id, varsArr);
            appendSystemCard(card);
        } else if (type === 'calculation_complete') {
            const card = buildResultCard(data.calc_id, data.outputs, data.audit_trace);
            appendSystemCard(card);
        } else if (type === 'validation_error') {
            const errs = data.errors || ["Unknown error"];
            const card = buildErrorCard(errs);
            appendSystemCard(card);
        } else if (type === 'assistant_message') {
            appendAssistantText(data.content);
        } else if (type === 'error') {
            appendErrorMessage(data.error);
        }
    }

    function dictToArr(varsObj) {
        return Object.entries(varsObj).map(([k, v]) => {
            let val = v; let unit = ''; let label = '';
            if (v && typeof v === 'object' && v.value !== undefined) {
                val = v.value; unit = v.unit || ''; label = v.label || '';
            }
            return { key: k, value: val, unit: unit, label: label };
        });
    }

    function buildLoadingCard(calcId) {
        return `
            <div class="card" id="loading-indicator" style="opacity: 0.8; padding: 12px 16px;">
                <div class="card-title" style="margin-bottom: 0; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <svg class="spin" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
                        Preparing ${calcId}...
                    </div>
                </div>
            </div>
        `;
    }

    function formatTitleCase(str) {
        return str.replace(/_/g, ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    function buildVariablesCard(calcId, vars) {
        let rows = '';
        if (vars.length === 0) {
            rows = `<div class="var-row"><span class="var-name">No variables extracted</span></div>`;
        } else {
            rows = vars.map(v => `
                <div class="var-row">
                    <span class="var-name">${v.label || formatTitleCase(v.key)}</span>
                    <span class="var-value">${v.value} <span style="font-size:0.8em; opacity:0.6">${v.unit || ''}</span></span>
                </div>
            `).join('');
        }

        const dataStr = encodeURIComponent(JSON.stringify(vars));

        return `
            <div class="card">
                <div class="card-title" style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                        Executing ${calcId}
                    </div>
                    <button class="circle-btn copy-btn" data-copy-type="vars" data-copy-data="${dataStr}" style="width:24px; height:24px; background:var(--bg-panel); border:1px solid var(--border-color); color:var(--text-secondary); cursor:pointer;" title="Copy to Clipboard">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    </button>
                </div>
                <div class="var-grid">${rows}</div>
            </div>
        `;
    }

    function buildResultCard(calcId, outputs, audit_trace) {
        outputs = outputs || {};
        let rows = Object.entries(outputs)
            .filter(([k, v]) => k !== 'source' && k !== 'confidence')
            .map(([k, v]) => `
                <div class="var-row">
                    <span class="var-name" style="color:var(--text-primary)">${formatTitleCase(k)}</span>
                    <span class="var-value" style="color:var(--accent-success)">${v}</span>
                </div>
            `).join('');

        let auditLog = '';
        // Removed audit log per user request

        const safeOutputs = {};
        Object.keys(outputs).forEach(k => {
            if (k !== 'source' && k !== 'confidence') safeOutputs[k] = outputs[k];
        });
        const dataStr = encodeURIComponent(JSON.stringify(safeOutputs));

        return `
            <div class="card success">
                <div class="card-title" style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                        Calculation Result
                    </div>
                    <button class="circle-btn copy-btn" data-copy-type="result" data-copy-data="${dataStr}" style="width:24px; height:24px; background:var(--bg-panel); border:1px solid var(--border-color); color:var(--text-secondary); cursor:pointer;" title="Copy to Clipboard">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    </button>
                </div>
                <div class="var-grid">${rows}</div>
                ${auditLog}
            </div>
        `;
    }

    function buildErrorCard(errors) {
        let txt = errors.join('<br/>');
        return `
            <div class="card error">
                <div class="card-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    Validation Failed
                </div>
                <div style="font-size:0.9rem; color:#fca5a5; font-family:var(--font-mono);">${txt}</div>
            </div>
        `;
    }

    function showErrorToast(msg) {
        // Appends a fast system error to the chat
        appendErrorMessage(msg);
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

    // --- MedASR Integration ---


    async function toggleRecording() {
        if (State.isProcessing) return;

        if (State.isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    async function loadMedASR() {
        try {
            setStatus('processing', 'Loading MedASR backend...');
            await apiRequest('/transcribe/medasr/load', { method: 'POST' });
            return true;
        } catch (e) {
            showErrorToast("Failed to load MedASR: " + e.message);
            setStatus('error', 'MedASR unavailable');
            return false;
        }
    }

    async function startRecording() {
        const loaded = await loadMedASR();
        if (!loaded) return;

        State.isRecording = true;
        DOM.recordBtn.classList.remove('inactive');
        DOM.recordBtn.classList.add('active');

        if (DOM.textInput.value && !DOM.textInput.value.endsWith(' ')) {
            DOM.textInput.value += ' ';
        }

        try {
            State.medasr.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
            showErrorToast("Mic access denied or unavailable: " + e.message);
            stopRecording();
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
            setStatus('recording', 'Recording (VAD)...');

            if (DOM.textInput.value && !DOM.textInput.value.endsWith(' ')) {
                DOM.textInput.value += ' ';
            }
        };

        State.medasr.socket.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data);
                if (data.type === 'partial') {
                    const currentPrefix = DOM.textInput.value;
                    const append = data.text || '';
                    if (append) {
                        DOM.textInput.value = currentPrefix + (currentPrefix.endsWith(' ') || !currentPrefix ? '' : ' ') + append + ' ';
                        adjustTextareaHeight();
                    }
                } else if (data.type === 'error') {
                    showErrorToast("ASR Error: " + data.error);
                    stopRecording();
                }
            } catch (err) { }
        };

        State.medasr.socket.onerror = (e) => {
            showErrorToast("ASR Connection error.");
            stopRecording();
        };

        State.medasr.processor.onaudioprocess = (e) => {
            const floatData = e.inputBuffer.getChannelData(0);
            const downsampled = downsample(floatData, sampleRate, 16000);
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
        setStatus('ready', 'System Ready');

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
            setStatus('recording', 'Waiting for screen selection...');
            const data = await apiRequest('/capture/screen', { method: 'POST' });

            if (data.status === 'success' && data.attachment) {
                State.attachment = data.attachment;
                DOM.attachmentName.textContent = State.attachment.name;
                DOM.attachmentsArea.style.display = 'flex';
                setStatus('ready', 'System Ready');
            } else {
                setStatus('ready', 'Capture cancelled');
                setTimeout(() => setStatus('ready', 'System Ready'), 2000);
            }
        } catch (err) {
            showErrorToast("Screen capture failed: " + err.message);
            setStatus('error', 'Capture failed');
            setTimeout(() => setStatus('ready', 'System Ready'), 3000);
        }
    }


    // --- UI State Helpers ---

    /* cls: 'ready', 'recording', 'processing', 'error' */
    function setStatus(cls, title) {
        DOM.statusIndicator.className = 'status-indicator ' + cls;
        DOM.statusIndicator.title = title;
        // Optionally update a small text element if we decide to re-add statusText
    }

    function setProcessingState(isProc) {
        State.isProcessing = isProc;
        DOM.sendBtn.disabled = isProc;
        DOM.recordBtn.disabled = isProc;

        if (isProc) {
            DOM.textInput.style.opacity = '0.7';
            DOM.sendBtn.classList.remove('primary');
            DOM.sendBtn.classList.add('inactive');
        } else {
            DOM.textInput.style.opacity = '1';
            DOM.sendBtn.classList.add('primary');
            DOM.sendBtn.classList.remove('inactive');
        }
    }

    function adjustTextareaHeight() {
        DOM.textInput.style.height = 'auto';
        DOM.textInput.style.height = Math.min(DOM.textInput.scrollHeight, 120) + 'px';
    }

    // --- Wiring Events ---

    function setupEventListeners() {
        DOM.recordBtn.addEventListener('click', toggleRecording);
        DOM.captureBtn.addEventListener('click', startScreenCapture);
        DOM.sendBtn.addEventListener('click', submitRequest);
        DOM.refreshModelsBtn.addEventListener('click', refreshModels);

        DOM.textInput.addEventListener('input', adjustTextareaHeight);
        DOM.textInput.addEventListener('keydown', (e) => {
            if (State.sendHotkey === 'enter') {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    submitRequest();
                }
            } else if (State.sendHotkey === 'cmd_enter') {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    submitRequest();
                }
            }
        });

        DOM.clearAttachmentBtn.addEventListener('click', () => {
            State.attachment = null;
            DOM.attachmentsArea.style.display = 'none';
        });

        DOM.newSessionBtn.addEventListener('click', () => {
            State.sessionId = crypto.randomUUID();
            DOM.chatContainer.innerHTML = '';
            // Re-inject welcome message
            if (DOM.welcomeMessage) {
                DOM.welcomeMessage.style.display = 'flex';
                DOM.chatContainer.appendChild(DOM.welcomeMessage);
            }
            DOM.textInput.value = '';
            State.attachment = null;
            DOM.attachmentsArea.style.display = 'none';
            adjustTextareaHeight();
            setStatus('ready', 'New session started');
        });

        DOM.settingsBtn.addEventListener('click', () => DOM.settingsModal.classList.remove('hidden'));
        const closeModals = () => DOM.settingsModal.classList.add('hidden');

        DOM.closeSettingsBtn.addEventListener('click', closeModals);
        DOM.settingsModal.addEventListener('click', (e) => {
            if (e.target === DOM.settingsModal) closeModals();
        });

        DOM.saveSettingsBtn.addEventListener('click', closeModals);

        DOM.hotkeySelect.addEventListener('change', (e) => {
            State.sendHotkey = e.target.value;
            try { localStorage.setItem('omnicalc_send_hotkey', State.sendHotkey); } catch (err) { }
        });

        function formatHotkey(e) {
            const keys = [];
            if (e.ctrlKey) keys.push('Ctrl');
            if (e.metaKey) keys.push('Cmd');
            if (e.altKey) keys.push('Alt');
            if (e.shiftKey) keys.push('Shift');
            if (e.key !== 'Control' && e.key !== 'Meta' && e.key !== 'Alt' && e.key !== 'Shift') {
                let keyName = e.key;
                if (keyName === ' ') keyName = 'Space';
                keys.push(keyName.length === 1 ? keyName.toUpperCase() : keyName);
            }
            return keys.join('+');
        }

        function bindHotkeyInput(inputEl, clearBtn, stateKey, storageKey) {
            if (!inputEl) return;

            let originalValue = '';

            inputEl.addEventListener('focus', () => {
                originalValue = inputEl.value;
                inputEl.value = 'Press new key...';
                inputEl.style.color = 'var(--text-accent, #3b82f6)';
                inputEl.style.borderColor = 'var(--text-accent, #3b82f6)';
            });

            inputEl.addEventListener('blur', () => {
                if (inputEl.value === 'Press new key...') {
                    inputEl.value = originalValue;
                }
                inputEl.style.color = '';
                inputEl.style.borderColor = '';
            });

            inputEl.addEventListener('keydown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (e.key === 'Escape') {
                    inputEl.blur();
                    return;
                }
                const hotkeyStr = formatHotkey(e);
                // Do not register if it's just a lonely modifier key
                if (hotkeyStr && !['Ctrl', 'Cmd', 'Alt', 'Shift'].includes(hotkeyStr)) {
                    inputEl.value = hotkeyStr;
                    State[stateKey] = hotkeyStr;
                    try { localStorage.setItem(storageKey, hotkeyStr); } catch (err) { }
                    setTimeout(() => inputEl.blur(), 100);
                }
            });

            if (clearBtn) {
                clearBtn.addEventListener('click', () => {
                    inputEl.value = 'None';
                    State[stateKey] = 'None';
                    try { localStorage.setItem(storageKey, 'None'); } catch (err) { }
                });
            }
        }

        bindHotkeyInput(DOM.asrHotkeyInput, DOM.clearAsrHotkeyBtn, 'asrHotkey', 'omnicalc_asr_hotkey');
        bindHotkeyInput(DOM.captureHotkeyInput, DOM.clearCaptureHotkeyBtn, 'captureHotkey', 'omnicalc_capture_hotkey');

        // Fast Keyboard Shortcuts Mapping
        window.addEventListener('keydown', (e) => {
            if (!DOM.settingsModal || !DOM.settingsModal.classList.contains('hidden')) return;

            const pressed = formatHotkey(e);

            // Screen Capture Hotkey
            if (pressed === State.captureHotkey && State.captureHotkey !== 'None') {
                e.preventDefault();
                startScreenCapture();
                return;
            }

            // ASR Hotkey
            if (pressed === State.asrHotkey && State.asrHotkey !== 'None') {
                e.preventDefault();
                toggleRecording();
                return;
            }

            // Secondary enter mapped just in case fallback
            const isCmd = e.ctrlKey || e.metaKey;
            if (isCmd && e.key === 'Enter') {
                e.preventDefault();
                submitRequest();
            }
        });
    }

});
