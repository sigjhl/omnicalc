/**
 * OmniCalc Frontend Logic - Mobile-First Chat UI
 */

document.addEventListener('DOMContentLoaded', () => {
    // Keep this explicit so switching to direct-audio LLM input is a one-line change later.
    const AUDIO_ATTACHMENT_MODE = 'medasr_transcript'; // 'medasr_transcript' | 'raw_audio'
    const FORWARD_RAW_AUDIO_WITH_TRANSCRIPT = false;

    const ATTACHMENT_LIMITS = {
        text: 1 * 1024 * 1024,
        image: 15 * 1024 * 1024,
        audio: 25 * 1024 * 1024,
    };
    const TEXT_EXTENSIONS = new Set(['.txt', '.md', '.csv', '.tsv', '.json', '.yaml', '.yml', '.log']);
    const AUDIO_EXTENSIONS = new Set(['.wav', '.mp3', '.m4a', '.aac', '.ogg', '.oga', '.webm', '.flac', '.opus', '.mp4', '.mpeg', '.mpga']);
    const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff', '.heic', '.heif']);
    const TEXT_MIME_TYPES = new Set(['application/json', 'application/xml', 'application/x-yaml']);

    // --- DOM Elements ---
    const DOM = {
        modelSelect: document.getElementById('modelSelect'),
        refreshModelsBtn: document.getElementById('refreshModelsBtn'),
        textInput: document.getElementById('textInput'),
        recordBtn: document.getElementById('recordBtn'),
        captureBtn: document.getElementById('captureBtn'),
        screenshotBtn: document.getElementById('screenshotBtn'),
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
        themeSelect: document.getElementById('themeSelect'),
        hotkeySelect: document.getElementById('hotkeySelect'),
        asrHotkeyInput: document.getElementById('asrHotkeyInput'),
        captureHotkeyInput: document.getElementById('captureHotkeyInput'),
        captureHotkeyLabel: document.getElementById('captureHotkeyLabel'),
        clearAsrHotkeyBtn: document.getElementById('clearAsrHotkeyBtn'),
        clearCaptureHotkeyBtn: document.getElementById('clearCaptureHotkeyBtn'),

        attachPickerModal: document.getElementById('attachPickerModal'),
        pickFileBtn: document.getElementById('pickFileBtn'),
        pickImageBtn: document.getElementById('pickImageBtn'),
        pickCameraBtn: document.getElementById('pickCameraBtn'),
        cancelAttachPickerBtn: document.getElementById('cancelAttachPickerBtn'),
        attachFileInput: document.getElementById('attachFileInput'),
        attachImageInput: document.getElementById('attachImageInput'),
        attachCameraInput: document.getElementById('attachCameraInput'),
        dragDropOverlay: document.getElementById('dragDropOverlay')
    };

    // --- Application State ---
    const State = {
        sessionId: crypto.randomUUID(),
        isRecording: false,
        attachment: null, // {name, kind, mime_type, data, transcript?}
        isDesktop: detectIsDesktopBrowser(),
        medasr: {
            socket: null,
            audioCtx: null,
            processor: null,
            mediaStream: null,
            loaded: false,
        },
        dragDropDepth: 0,
        isProcessing: false,

        // Chat state tracking to append chunks to the same bubble
        currentAssistantBubble: null,
        typingIndicator: null,
        generatingStatusMsg: null,
        pendingExecutionCardMsg: null,
        pendingExecutionKey: null,
        lastCompletedExecutionKey: null,
        lastResultKey: null,
        sendHotkey: localStorage.getItem('omnicalc_send_hotkey') || 'enter',
        asrHotkey: localStorage.getItem('omnicalc_asr_hotkey') || 'Cmd+R',
        captureHotkey: localStorage.getItem('omnicalc_capture_hotkey') || 'Cmd+Shift+X',
        theme: localStorage.getItem('omnicalc_theme') || 'system'
    };

    // --- Initialization ---
    init();

    async function init() {
        applyTheme(State.theme);
        applyCaptureModeUI();
        setupEventListeners();
        await refreshModels();
        setStatus('ready', 'System Ready');
        adjustTextareaHeight();

        try {
            const savedModel = localStorage.getItem('omnicalc_model');
            if (savedModel) {
                // If we want to restore saved model...
            }
            if (DOM.themeSelect) DOM.themeSelect.value = State.theme;
            if (DOM.hotkeySelect) DOM.hotkeySelect.value = State.sendHotkey;
            if (DOM.asrHotkeyInput) DOM.asrHotkeyInput.value = State.asrHotkey;
            if (DOM.captureHotkeyInput) DOM.captureHotkeyInput.value = State.captureHotkey;
        } catch (e) {
            console.error('Storage access error', e);
        }
    }

    function detectIsDesktopBrowser() {
        const ua = navigator.userAgent || '';
        const isTouchMac = /Macintosh/i.test(ua) && (navigator.maxTouchPoints || 0) > 1;
        const isMobileUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile/i.test(ua);
        const isMobile = isMobileUA || isTouchMac || (navigator.userAgentData && navigator.userAgentData.mobile);
        return !isMobile;
    }

    function applyCaptureModeUI() {
        if (DOM.captureBtn) {
            DOM.captureBtn.title = State.isDesktop ? 'Add Attachment' : `Add Attachment (${State.captureHotkey})`;
        }
        if (DOM.screenshotBtn) {
            DOM.screenshotBtn.title = `Capture Screen (${State.captureHotkey})`;
            DOM.screenshotBtn.classList.toggle('hidden', !State.isDesktop);
        }
        if (DOM.captureHotkeyLabel) {
            DOM.captureHotkeyLabel.textContent = State.isDesktop ? 'Screenshot Hotkey' : 'Attachment Picker Hotkey';
        }
        if (DOM.pickImageBtn) {
            DOM.pickImageBtn.classList.toggle('hidden', State.isDesktop);
        }
        if (DOM.pickCameraBtn) {
            DOM.pickCameraBtn.classList.toggle('hidden', State.isDesktop);
        }
        if (!State.isDesktop && DOM.dragDropOverlay) {
            DOM.dragDropOverlay.classList.add('hidden');
        }
    }

    // --- Theme Logic ---
    function applyTheme(themeValue) {
        document.body.classList.remove('theme-light', 'theme-dark');

        let activeTheme = themeValue;
        if (themeValue === 'system') {
            activeTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
        }

        if (activeTheme === 'light') {
            document.body.classList.add('theme-light');
        } else {
            document.body.classList.add('theme-dark');
        }
    }

    // Listen for system theme changes if set to system
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', e => {
        if (State.theme === 'system') {
            applyTheme('system');
        }
    });

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
                const savedModel = localStorage.getItem('omnicalc_model');
                let foundSelection = false;

                let modelToSelect = savedModel;
                if (!modelToSelect || !data.models.includes(modelToSelect)) {
                    const gemmaModel = data.models.find(m => m.toLowerCase().includes('medgemma'));
                    if (gemmaModel) {
                        modelToSelect = gemmaModel;
                    } else if (data.selected_model && data.models.includes(data.selected_model)) {
                        modelToSelect = data.selected_model;
                    } else {
                        modelToSelect = data.models[0];
                    }
                    localStorage.setItem('omnicalc_model', modelToSelect);
                }

                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = opt.textContent = m;
                    if (m === modelToSelect) {
                        opt.selected = true;
                        foundSelection = true;
                    }
                    DOM.modelSelect.appendChild(opt);
                });

                if (!foundSelection && data.models.length > 0 && !Array.from(DOM.modelSelect.options).some(o => o.selected)) {
                    const defaultOpt = Array.from(DOM.modelSelect.options).find(o => o.value === data.selected_model) || DOM.modelSelect.options[0];
                    defaultOpt.selected = true;
                    localStorage.setItem('omnicalc_model', defaultOpt.value);
                }
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
            const preparedAttachments = buildRequestAttachments(State.attachment);
            if (preparedAttachments.length) {
                payload.attachments = preparedAttachments;
            }
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
        clearGeneratingStatusCard();
        State.pendingExecutionKey = null;
        State.lastCompletedExecutionKey = null;
        State.lastResultKey = null;
        removePendingExecutionCard();

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
            clearGeneratingStatusCard();
            setStatus('ready', 'System Ready');
        } catch (err) {
            removeTypingIndicator();
            clearGeneratingStatusCard();
            appendErrorMessage(err.message);
            setStatus('error', 'Error');
        } finally {
            setProcessingState(false);
            clearGeneratingStatusCard();
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
        bubble.className = 'card user-card';
        bubble.style.margin = '0';

        let contentHtml = '';
        if (attachment) {
            contentHtml += `<div style="font-size:0.8rem; margin-bottom:4px; opacity:0.8;">&#128206; ${attachment.name}</div>`;
        }
        if (text) {
            const safeText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            contentHtml += `<div style="padding-top: 8px; line-height: 1.5;">${safeText.replace(/\\n/g, '<br/>')}</div>`;
        } else {
            const attachmentLabel = attachment?.kind === 'audio'
                ? '[Audio attached]'
                : attachment?.kind === 'text'
                    ? '[Text attached]'
                    : '[Image attached]';
            contentHtml += `<div style="padding-top: 8px;"><em>${attachmentLabel}</em></div>`;
        }

        const encodedText = encodeURIComponent(text || '');
        bubble.innerHTML = `
            <div class="card-title" style="color: var(--accent-primary); display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                    User Query
                </div>
                <div style="display: flex; gap: 4px;">
                    <button class="circle-btn copy-btn copy-user-btn" data-copy-type="user-text" data-copy-data="${encodedText}" style="width:24px; height:24px; background:var(--bg-panel); border:1px solid var(--border-color); color:var(--text-secondary); cursor:pointer;" title="Copy Message">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    </button>
                    <button class="circle-btn edit-user-btn" data-raw-text="${encodedText}" style="width:24px; height:24px; background:var(--bg-panel); border:1px solid var(--border-color); color:var(--text-secondary); cursor:pointer;" title="Edit Message">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                    </button>
                </div>
            </div>
            ${contentHtml}
        `;

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

        return msgDiv;
    }

    function removePendingExecutionCard() {
        clearGeneratingStatusCard();
        State.pendingExecutionCardMsg = null;
        State.pendingExecutionKey = null;
    }

    function setGeneratingStatusCard(cardHtml) {
        if (State.generatingStatusMsg && State.generatingStatusMsg.parentNode) {
            State.generatingStatusMsg.innerHTML = cardHtml;
            scrollToBottom();
            return State.generatingStatusMsg;
        }
        State.generatingStatusMsg = appendSystemCard(cardHtml);
        return State.generatingStatusMsg;
    }

    function clearGeneratingStatusCard() {
        if (State.generatingStatusMsg && State.generatingStatusMsg.parentNode) {
            State.generatingStatusMsg.parentNode.removeChild(State.generatingStatusMsg);
        }
        State.generatingStatusMsg = null;
    }

    function stableSerialize(value) {
        if (value === null || value === undefined) return String(value);
        if (typeof value !== 'object') return JSON.stringify(value);
        if (Array.isArray(value)) {
            return `[${value.map(stableSerialize).join(',')}]`;
        }
        const keys = Object.keys(value).sort();
        return `{${keys.map(k => `${JSON.stringify(k)}:${stableSerialize(value[k])}`).join(',')}}`;
    }

    function makeExecKey(calcId, variables) {
        return `${calcId || ''}|${stableSerialize(variables || {})}`;
    }

    function makeResultKey(calcId, outputs) {
        return `${calcId || ''}|${stableSerialize(outputs || {})}`;
    }

    function appendErrorMessage(text) {
        removeTypingIndicator();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system';
        msgDiv.innerHTML = `
            <div class="card error">
                <div class="card-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    Failed
                </div>
                <div style="font-size:0.9rem; color:#fca5a5; font-family:var(--font-mono);">${text}</div>
            </div>
        `;
        DOM.chatContainer.appendChild(msgDiv);
    }

    // Chat Container Event Delegation
    DOM.chatContainer.addEventListener('click', (e) => {
        const btn = e.target.closest('.copy-btn');
        if (btn) {
            try {
                const type = btn.dataset.copyType;
                const dataRaw = btn.dataset.copyData;
                let text = "";

                if (type === 'user-text') {
                    text = decodeURIComponent(dataRaw);
                } else {
                    const data = JSON.parse(decodeURIComponent(dataRaw));
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
            return;
        }

        const editBtn = e.target.closest('.edit-user-btn');
        if (editBtn) {
            try {
                const rawText = decodeURIComponent(editBtn.dataset.rawText || '');
                DOM.textInput.value = rawText;
                adjustTextareaHeight();

                // Delete this user message and all subsequent messages
                const userMsgDiv = editBtn.closest('.message.user');
                if (userMsgDiv) {
                    let nextNode = userMsgDiv.nextSibling;
                    while (nextNode) {
                        const toRemove = nextNode;
                        nextNode = nextNode.nextSibling;
                        toRemove.remove();
                    }
                    userMsgDiv.remove();
                }

                // Generate a fresh session ID so history resets back to system
                State.sessionId = crypto.randomUUID();
                setStatus('ready', 'Editing previous message');
                DOM.textInput.focus();
            } catch (err) {
                console.error("Failed to edit", err);
            }
            return;
        }
    });


    // --- Stream Handlers ---

    function handleStreamEvent(event) {
        const type = event.type.toLowerCase();
        const data = event.data;
        console.log("STREAM EVENT:", type, data);

        if (type === 'calculator_selected') {
            const calcId = data.calc_id || 'calculator';
            const card = buildLoadingCard(calcId);
            setGeneratingStatusCard(card);
        } else if (type === 'extracting_variables') {
            const execKey = makeExecKey(data.calc_id, data.variables || {});
            if (execKey === State.pendingExecutionKey || execKey === State.lastCompletedExecutionKey) {
                return;
            }
            const varsArr = dictToArr(data.variables || {});
            const card = buildVariablesCard(data.calc_id, varsArr);
            // Promote execution details to a persistent card (not a transient status card).
            clearGeneratingStatusCard();
            State.pendingExecutionCardMsg = appendSystemCard(card);
            State.pendingExecutionKey = execKey;
        } else if (type === 'calculation_complete') {
            const resultKey = makeResultKey(data.calc_id, data.outputs || {});
            if (resultKey === State.lastResultKey) {
                clearGeneratingStatusCard();
                return;
            }
            // This execution succeeded; keep the card and clear pending marker.
            if (State.pendingExecutionKey) {
                State.lastCompletedExecutionKey = State.pendingExecutionKey;
            }
            State.lastResultKey = resultKey;
            clearGeneratingStatusCard();
            State.pendingExecutionCardMsg = null;
            State.pendingExecutionKey = null;
            const card = buildResultCard(data.calc_id, data.outputs, data.audit_trace);
            appendSystemCard(card);
        } else if (type === 'validation_error') {
            // Drop the pre-exec variables card when backend rejected the call.
            removePendingExecutionCard();
            const errs = data.errors || ["Unknown error"];
            const card = buildErrorCard(errs);
            appendSystemCard(card);
        } else if (type === 'assistant_message') {
            appendAssistantText(data.content);
        } else if (type === 'error') {
            // Remove dangling execution card if the request aborted before result.
            removePendingExecutionCard();
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

    function openAttachmentPicker() {
        if (State.isProcessing) return;
        if (!DOM.attachPickerModal) return;
        DOM.attachPickerModal.classList.remove('hidden');
    }

    function closeAttachmentPicker() {
        if (!DOM.attachPickerModal) return;
        DOM.attachPickerModal.classList.add('hidden');
    }

    function triggerAttachmentInput(inputEl) {
        closeAttachmentPicker();
        if (!inputEl) return;
        inputEl.click();
    }

    function hasFileDataTransfer(event) {
        const dt = event && event.dataTransfer;
        if (!dt) return false;
        if (dt.items && dt.items.length) {
            return Array.from(dt.items).some((item) => item && item.kind === 'file');
        }
        if (dt.types && dt.types.length) {
            return Array.from(dt.types).includes('Files');
        }
        return false;
    }

    function showDragDropOverlay() {
        if (!State.isDesktop || !DOM.dragDropOverlay) return;
        DOM.dragDropOverlay.classList.remove('hidden');
    }

    function hideDragDropOverlay() {
        if (!DOM.dragDropOverlay) return;
        DOM.dragDropOverlay.classList.add('hidden');
    }

    function resetDragDropOverlay() {
        State.dragDropDepth = 0;
        hideDragDropOverlay();
    }

    function extensionForMimeType(mimeType) {
        const mt = String(mimeType || '').toLowerCase();
        if (mt === 'image/png') return '.png';
        if (mt === 'image/jpeg') return '.jpg';
        if (mt === 'image/webp') return '.webp';
        if (mt === 'image/gif') return '.gif';
        if (mt === 'audio/wav' || mt === 'audio/x-wav') return '.wav';
        if (mt === 'audio/mpeg' || mt === 'audio/mp3') return '.mp3';
        if (mt === 'audio/mp4' || mt === 'audio/m4a') return '.m4a';
        if (mt === 'audio/ogg') return '.ogg';
        if (mt === 'audio/webm') return '.webm';
        if (mt.startsWith('text/')) return '.txt';
        return '';
    }

    function normalizeClipboardFile(file, fallbackBaseName) {
        if (!file) return null;
        const originalName = String(file.name || '').trim();
        if (originalName) return file;
        const mimeType = String(file.type || 'application/octet-stream');
        const ext = extensionForMimeType(mimeType);
        const base = fallbackBaseName || 'ClipboardFile';
        const fileName = `${base}${ext}`;
        try {
            return new File([file], fileName, { type: mimeType || undefined });
        } catch (e) {
            file.name = fileName;
            return file;
        }
    }

    function extractClipboardFiles(event) {
        const dt = event && event.clipboardData;
        if (!dt) return [];

        const output = [];
        const dtFiles = Array.from(dt.files || []);
        dtFiles.forEach((file) => {
            const normalized = normalizeClipboardFile(file, 'ClipboardFile');
            if (normalized) output.push(normalized);
        });
        if (output.length) return output;

        const items = Array.from(dt.items || []);
        items.forEach((item, idx) => {
            if (!item || item.kind !== 'file') return;
            const rawFile = item.getAsFile();
            if (!rawFile) return;
            const normalized = normalizeClipboardFile(rawFile, `PastedFile${idx + 1}`);
            if (normalized) output.push(normalized);
        });
        return output;
    }

    function isCaptureCancelError(err) {
        const name = err && err.name ? String(err.name) : '';
        return name === 'NotAllowedError' || name === 'AbortError' || name === 'NotFoundError';
    }

    function stopStreamTracks(stream) {
        if (!stream) return;
        const tracks = typeof stream.getTracks === 'function' ? stream.getTracks() : [];
        tracks.forEach((track) => {
            try { track.stop(); } catch (e) { }
        });
    }

    function dataUrlToBase64(dataUrl) {
        const parts = String(dataUrl || '').split(',');
        return parts.length > 1 ? parts[1] : '';
    }

    async function captureDisplayFrameDataUrl() {
        let stream = null;
        try {
            stream = await navigator.mediaDevices.getDisplayMedia({
                video: true,
                audio: false
            });
        } catch (err) {
            if (isCaptureCancelError(err)) {
                return null;
            }
            throw err;
        }

        try {
            const track = stream.getVideoTracks && stream.getVideoTracks()[0];
            if (!track) {
                throw new Error('No video track from display capture');
            }

            // Preferred path: grab a clean frame directly from the track.
            if (typeof window.ImageCapture === 'function') {
                try {
                    const imageCapture = new ImageCapture(track);
                    const bitmap = await imageCapture.grabFrame();
                    const canvas = document.createElement('canvas');
                    canvas.width = bitmap.width;
                    canvas.height = bitmap.height;
                    const ctx = canvas.getContext('2d');
                    if (!ctx) throw new Error('Canvas context unavailable');
                    ctx.drawImage(bitmap, 0, 0);
                    if (typeof bitmap.close === 'function') {
                        bitmap.close();
                    }
                    return canvas.toDataURL('image/png');
                } catch (e) {
                    // Fallback to video-frame path below.
                }
            }

            const video = document.createElement('video');
            video.srcObject = stream;
            video.muted = true;
            video.playsInline = true;

            await new Promise((resolve, reject) => {
                const onLoaded = () => {
                    cleanup();
                    resolve();
                };
                const onError = (event) => {
                    cleanup();
                    reject(event || new Error('Failed to load display video'));
                };
                const cleanup = () => {
                    video.removeEventListener('loadedmetadata', onLoaded);
                    video.removeEventListener('error', onError);
                };
                video.addEventListener('loadedmetadata', onLoaded);
                video.addEventListener('error', onError);
            });

            await video.play();
            await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

            const width = video.videoWidth || 0;
            const height = video.videoHeight || 0;
            if (width <= 0 || height <= 0) {
                throw new Error('Captured frame has invalid dimensions');
            }

            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Canvas context unavailable');
            ctx.drawImage(video, 0, 0, width, height);
            return canvas.toDataURL('image/png');
        } finally {
            stopStreamTracks(stream);
        }
    }

    async function cropImageDataUrl(dataUrl) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const viewportW = Math.max(360, window.innerWidth || document.documentElement.clientWidth || 0);
                const viewportH = Math.max(280, window.innerHeight || document.documentElement.clientHeight || 0);
                const panelW = Math.max(320, Math.min(Math.floor(viewportW * 0.96), 1120));
                const maxW = Math.max(280, panelW - 32);
                const maxH = Math.max(200, Math.floor(viewportH * 0.62));
                const scale = Math.min(1, maxW / img.width, maxH / img.height);
                const drawW = Math.max(1, Math.floor(img.width * scale));
                const drawH = Math.max(1, Math.floor(img.height * scale));
                const invScaleX = img.width / drawW;
                const invScaleY = img.height / drawH;

                const backdrop = document.createElement('div');
                backdrop.style.position = 'fixed';
                backdrop.style.inset = '0';
                backdrop.style.background = 'rgba(0,0,0,0.78)';
                backdrop.style.display = 'flex';
                backdrop.style.alignItems = 'center';
                backdrop.style.justifyContent = 'center';
                backdrop.style.zIndex = '2000';

                const panel = document.createElement('div');
                panel.style.width = `${panelW}px`;
                panel.style.maxWidth = '96vw';
                panel.style.maxHeight = '94vh';
                panel.style.background = 'var(--bg-surface)';
                panel.style.border = '1px solid var(--border-color)';
                panel.style.borderRadius = '16px';
                panel.style.padding = '12px';
                panel.style.display = 'flex';
                panel.style.flexDirection = 'column';
                panel.style.gap = '10px';
                panel.style.overflow = 'hidden';

                const title = document.createElement('div');
                title.textContent = 'Drag to crop, then attach';
                title.style.fontSize = '0.92rem';
                title.style.color = 'var(--text-secondary)';

                const canvasWrap = document.createElement('div');
                canvasWrap.style.display = 'flex';
                canvasWrap.style.justifyContent = 'center';
                canvasWrap.style.overflow = 'auto';
                canvasWrap.style.maxWidth = '100%';
                canvasWrap.style.maxHeight = 'calc(94vh - 120px)';

                const canvas = document.createElement('canvas');
                canvas.width = drawW;
                canvas.height = drawH;
                canvas.style.width = `${drawW}px`;
                canvas.style.height = `${drawH}px`;
                canvas.style.maxWidth = '100%';
                canvas.style.maxHeight = '100%';
                canvas.style.cursor = 'crosshair';
                canvas.style.border = '1px solid var(--border-color)';
                canvas.style.borderRadius = '10px';
                canvas.style.background = '#000';

                const controls = document.createElement('div');
                controls.style.display = 'flex';
                controls.style.justifyContent = 'flex-end';
                controls.style.gap = '8px';

                const fullBtn = document.createElement('button');
                fullBtn.type = 'button';
                fullBtn.textContent = 'Use Full';
                fullBtn.style.border = '1px solid var(--border-color)';
                fullBtn.style.background = 'var(--bg-panel)';
                fullBtn.style.color = 'var(--text-primary)';
                fullBtn.style.borderRadius = '10px';
                fullBtn.style.padding = '8px 12px';
                fullBtn.style.cursor = 'pointer';

                const cancelBtn = document.createElement('button');
                cancelBtn.type = 'button';
                cancelBtn.textContent = 'Cancel';
                cancelBtn.style.border = '1px solid var(--border-color)';
                cancelBtn.style.background = 'var(--bg-panel)';
                cancelBtn.style.color = 'var(--text-primary)';
                cancelBtn.style.borderRadius = '10px';
                cancelBtn.style.padding = '8px 12px';
                cancelBtn.style.cursor = 'pointer';

                const attachBtn = document.createElement('button');
                attachBtn.type = 'button';
                attachBtn.textContent = 'Attach';
                attachBtn.style.border = 'none';
                attachBtn.style.background = 'var(--accent-primary)';
                attachBtn.style.color = '#fff';
                attachBtn.style.borderRadius = '10px';
                attachBtn.style.padding = '8px 14px';
                attachBtn.style.fontWeight = '600';
                attachBtn.style.cursor = 'pointer';

                controls.appendChild(fullBtn);
                controls.appendChild(cancelBtn);
                controls.appendChild(attachBtn);
                canvasWrap.appendChild(canvas);
                panel.appendChild(title);
                panel.appendChild(canvasWrap);
                panel.appendChild(controls);
                backdrop.appendChild(panel);
                document.body.appendChild(backdrop);

                const ctx = canvas.getContext('2d');
                if (!ctx) {
                    cleanup();
                    resolve(null);
                    return;
                }

                const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
                const normalizeRect = (x0, y0, x1, y1) => {
                    const nx0 = clamp(Math.min(x0, x1), 0, drawW);
                    const ny0 = clamp(Math.min(y0, y1), 0, drawH);
                    const nx1 = clamp(Math.max(x0, x1), 0, drawW);
                    const ny1 = clamp(Math.max(y0, y1), 0, drawH);
                    return { x: nx0, y: ny0, w: nx1 - nx0, h: ny1 - ny0 };
                };
                const toCanvasPoint = (event) => {
                    const rect = canvas.getBoundingClientRect();
                    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
                    return {
                        x: clamp(x, 0, canvas.width),
                        y: clamp(y, 0, canvas.height),
                    };
                };

                let selection = { x: 0, y: 0, w: drawW, h: drawH };
                let dragging = false;
                let startPoint = null;
                let onKeyDown = null;

                const drawScene = () => {
                    ctx.clearRect(0, 0, drawW, drawH);
                    ctx.drawImage(img, 0, 0, drawW, drawH);

                    ctx.fillStyle = 'rgba(0, 0, 0, 0.42)';
                    ctx.fillRect(0, 0, drawW, drawH);
                    // Repaint the selected area from the scaled image (not raw image coords),
                    // so the viewport preview matches what the user expects.
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(selection.x, selection.y, selection.w, selection.h);
                    ctx.clip();
                    ctx.drawImage(img, 0, 0, drawW, drawH);
                    ctx.restore();

                    ctx.strokeStyle = '#3b82f6';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(selection.x, selection.y, selection.w, selection.h);
                };

                const onPointerDown = (event) => {
                    event.preventDefault();
                    startPoint = toCanvasPoint(event);
                    dragging = true;
                    selection = { x: startPoint.x, y: startPoint.y, w: 0, h: 0 };
                    drawScene();
                };

                const onPointerMove = (event) => {
                    if (!dragging || !startPoint) return;
                    const point = toCanvasPoint(event);
                    selection = normalizeRect(startPoint.x, startPoint.y, point.x, point.y);
                    drawScene();
                };

                const onPointerUp = () => {
                    if (!dragging) return;
                    dragging = false;
                    if (selection.w < 4 || selection.h < 4) {
                        selection = { x: 0, y: 0, w: drawW, h: drawH };
                    }
                    drawScene();
                };

                const cleanup = () => {
                    canvas.removeEventListener('pointerdown', onPointerDown);
                    window.removeEventListener('pointermove', onPointerMove);
                    window.removeEventListener('pointerup', onPointerUp);
                    if (onKeyDown) {
                        window.removeEventListener('keydown', onKeyDown);
                    }
                    if (backdrop.parentNode) {
                        backdrop.parentNode.removeChild(backdrop);
                    }
                };

                const finishWithSelection = () => {
                    const sel = (selection.w < 4 || selection.h < 4)
                        ? { x: 0, y: 0, w: drawW, h: drawH }
                        : selection;

                    const sx = Math.max(0, Math.floor(sel.x * invScaleX));
                    const sy = Math.max(0, Math.floor(sel.y * invScaleY));
                    const sw = Math.max(1, Math.floor(sel.w * invScaleX));
                    const sh = Math.max(1, Math.floor(sel.h * invScaleY));

                    const cropCanvas = document.createElement('canvas');
                    cropCanvas.width = sw;
                    cropCanvas.height = sh;
                    const cropCtx = cropCanvas.getContext('2d');
                    if (!cropCtx) {
                        cleanup();
                        resolve(null);
                        return;
                    }
                    cropCtx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
                    const out = cropCanvas.toDataURL('image/png');
                    cleanup();
                    resolve(out);
                };

                onKeyDown = (event) => {
                    if (event.key === 'Escape') {
                        event.preventDefault();
                        cleanup();
                        resolve(null);
                    } else if (event.key === 'Enter') {
                        event.preventDefault();
                        finishWithSelection();
                    }
                };

                canvas.addEventListener('pointerdown', onPointerDown);
                window.addEventListener('pointermove', onPointerMove);
                window.addEventListener('pointerup', onPointerUp);
                window.addEventListener('keydown', onKeyDown);

                fullBtn.addEventListener('click', () => {
                    selection = { x: 0, y: 0, w: drawW, h: drawH };
                    drawScene();
                });
                cancelBtn.addEventListener('click', () => {
                    cleanup();
                    resolve(null);
                });
                attachBtn.addEventListener('click', finishWithSelection);
                backdrop.addEventListener('click', (event) => {
                    if (event.target === backdrop) {
                        cleanup();
                        resolve(null);
                    }
                });

                drawScene();
                attachBtn.focus();
            };

            img.onerror = () => resolve(null);
            img.src = dataUrl;
        });
    }

    async function fallbackServerSideScreenCapture() {
        const data = await apiRequest('/capture/screen', { method: 'POST' });
        if (data && data.status === 'success' && data.attachment) {
            return data.attachment;
        }
        return null;
    }

    async function startScreenCapture() {
        if (State.isProcessing) return;
        try {
            const supportsDisplayCapture = !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);

            if (supportsDisplayCapture) {
                setStatus('processing', 'Choose a screen/window to capture...');
                const screenDataUrl = await captureDisplayFrameDataUrl();
                if (!screenDataUrl) {
                    setStatus('ready', 'Capture cancelled');
                    setTimeout(() => setStatus('ready', 'System Ready'), 1200);
                    return;
                }

                setStatus('processing', 'Adjust crop and attach...');
                const croppedDataUrl = await cropImageDataUrl(screenDataUrl);
                if (!croppedDataUrl) {
                    setStatus('ready', 'Capture cancelled');
                    setTimeout(() => setStatus('ready', 'System Ready'), 1200);
                    return;
                }

                const base64 = dataUrlToBase64(croppedDataUrl);
                if (!base64) {
                    throw new Error('Failed to encode cropped screenshot');
                }
                State.attachment = {
                    name: 'Screenshot.png',
                    kind: 'image',
                    mime_type: 'image/png',
                    data: base64,
                };
            } else {
                setStatus('processing', 'Waiting for screen capture selection...');
                const fallbackAttachment = await fallbackServerSideScreenCapture();
                if (!fallbackAttachment) {
                    setStatus('ready', 'Capture cancelled');
                    setTimeout(() => setStatus('ready', 'System Ready'), 1200);
                    return;
                }
                State.attachment = fallbackAttachment;
            }

            DOM.attachmentName.textContent = State.attachment.name || 'Screenshot.png';
            DOM.attachmentsArea.style.display = 'flex';
            setStatus('ready', 'Screenshot ready');
            setTimeout(() => setStatus('ready', 'System Ready'), 1200);
        } catch (err) {
            if (isCaptureCancelError(err)) {
                setStatus('ready', 'Capture cancelled');
                setTimeout(() => setStatus('ready', 'System Ready'), 1200);
                return;
            }
            console.error('Capture error', err);
            showErrorToast('Screen capture failed: ' + (err.message || String(err)));
            setStatus('error', 'Capture failed');
            setTimeout(() => setStatus('ready', 'System Ready'), 2000);
        }
    }

    function handleCaptureAction() {
        if (State.isDesktop) {
            triggerAttachmentInput(DOM.attachFileInput);
            return;
        }
        openAttachmentPicker();
    }

    function getFileExtension(fileName) {
        const idx = String(fileName || '').lastIndexOf('.');
        if (idx < 0) return '';
        return fileName.slice(idx).toLowerCase();
    }

    function classifyAttachment(file) {
        const extension = getFileExtension(file.name);
        const mimeType = String(file.type || '').toLowerCase();

        if (TEXT_EXTENSIONS.has(extension) || mimeType.startsWith('text/') || TEXT_MIME_TYPES.has(mimeType)) {
            return 'text';
        }
        if (AUDIO_EXTENSIONS.has(extension) || mimeType.startsWith('audio/')) {
            return 'audio';
        }
        if (IMAGE_EXTENSIONS.has(extension) || mimeType.startsWith('image/')) {
            return 'image';
        }
        return null;
    }

    function getAttachmentLimit(kind) {
        return ATTACHMENT_LIMITS[kind] || ATTACHMENT_LIMITS.image;
    }

    function readFileAsDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error || new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    function arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        const chunk = 0x8000;
        let binary = '';
        for (let i = 0; i < bytes.length; i += chunk) {
            binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
        }
        return btoa(binary);
    }

    function textToBase64Utf8(text) {
        const bytes = new TextEncoder().encode(text);
        return arrayBufferToBase64(bytes.buffer);
    }

    function decodeAudioToMono(audioBuffer) {
        const channels = audioBuffer.numberOfChannels || 1;
        if (channels <= 1) {
            return new Float32Array(audioBuffer.getChannelData(0));
        }
        const mono = new Float32Array(audioBuffer.length);
        for (let c = 0; c < channels; c++) {
            const channelData = audioBuffer.getChannelData(c);
            for (let i = 0; i < channelData.length; i++) {
                mono[i] += channelData[i];
            }
        }
        for (let i = 0; i < mono.length; i++) {
            mono[i] = mono[i] / channels;
        }
        return mono;
    }

    function sanitizeAttachmentForPayload(attachment) {
        if (!attachment) return null;
        return {
            name: attachment.name,
            kind: attachment.kind,
            mime_type: attachment.mime_type,
            data: attachment.data,
        };
    }

    function buildRequestAttachments(attachment) {
        if (!attachment) return [];
        if (attachment.kind !== 'audio') {
            const basic = sanitizeAttachmentForPayload(attachment);
            return basic ? [basic] : [];
        }

        if (AUDIO_ATTACHMENT_MODE === 'raw_audio') {
            const rawOnly = sanitizeAttachmentForPayload(attachment);
            return rawOnly ? [rawOnly] : [];
        }

        const transcript = String(attachment.transcript || '').trim();
        if (!transcript) {
            const fallbackRaw = sanitizeAttachmentForPayload(attachment);
            return fallbackRaw ? [fallbackRaw] : [];
        }

        const transcriptText = `[Audio transcript from ${attachment.name || 'Audio'}]\n${transcript}`;
        const transcriptAttachment = {
            name: `${attachment.name || 'audio'}_transcript.txt`,
            kind: 'text',
            mime_type: 'text/plain',
            data: textToBase64Utf8(transcriptText),
        };

        if (FORWARD_RAW_AUDIO_WITH_TRANSCRIPT) {
            const raw = sanitizeAttachmentForPayload(attachment);
            return raw ? [transcriptAttachment, raw] : [transcriptAttachment];
        }
        return [transcriptAttachment];
    }

    async function transcribeAudioAttachment(file) {
        const asrReady = await loadMedASR();
        if (!asrReady) {
            throw new Error('MedASR is unavailable');
        }

        const AudioContextCls = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCls) {
            throw new Error('Audio decoding is not supported in this browser');
        }

        const arrayBuffer = await file.arrayBuffer();
        const decodeCtx = new AudioContextCls();
        try {
            const decoded = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
            const mono = decodeAudioToMono(decoded);
            const downsampled = downsample(mono, decoded.sampleRate, 16000);
            const pcm16 = floatTo16BitPCM(downsampled);
            const pcmB64 = arrayBufferToBase64(pcm16.buffer);

            const response = await apiRequest('/transcribe/medasr', {
                method: 'POST',
                body: JSON.stringify({
                    pcm16: pcmB64,
                    sample_rate: 16000,
                }),
            });
            return String(response.transcript || '').trim();
        } finally {
            try {
                await decodeCtx.close();
            } catch (e) { }
        }
    }

    async function attachFile(file) {
        if (!file) return;
        if (State.isProcessing) {
            showErrorToast('Please wait for the current request to finish before attaching a file.');
            return;
        }
        const kind = classifyAttachment(file);
        if (!kind) {
            showErrorToast(`Unsupported file type for ${file.name}. Allowed: text, audio, image.`);
            return;
        }

        const maxBytes = getAttachmentLimit(kind);
        if (file.size > maxBytes) {
            const maxMb = (maxBytes / (1024 * 1024)).toFixed(0);
            showErrorToast(`File too large (${file.name}). Max ${maxMb} MB for ${kind}.`);
            return;
        }

        try {
            const mimeType = file.type || (
                kind === 'image' ? 'image/png'
                    : kind === 'audio' ? 'audio/wav'
                        : 'text/plain'
            );

            if (kind === 'text') {
                setStatus('processing', 'Reading text attachment...');
                const rawText = await file.text();
                const text = String(rawText || '');
                if (!text.trim()) {
                    throw new Error('Text file is empty');
                }
                State.attachment = {
                    name: file.name || 'Attachment.txt',
                    kind: 'text',
                    mime_type: mimeType,
                    data: textToBase64Utf8(text),
                };
            } else if (kind === 'audio') {
                setStatus('processing', 'Reading audio attachment...');
                const dataUrl = await readFileAsDataURL(file);
                const parts = String(dataUrl).split(',');
                if (parts.length < 2) {
                    throw new Error('Malformed audio payload');
                }

                setStatus('processing', 'Transcribing audio with MedASR...');
                const transcript = await transcribeAudioAttachment(file);
                if (!transcript) {
                    throw new Error('No speech detected in audio');
                }
                State.attachment = {
                    name: file.name || 'Audio',
                    kind: 'audio',
                    mime_type: mimeType,
                    data: parts[1],
                    transcript,
                };
            } else {
                setStatus('processing', 'Reading image attachment...');
                const dataUrl = await readFileAsDataURL(file);
                const parts = String(dataUrl).split(',');
                if (parts.length < 2) {
                    throw new Error('Malformed image payload');
                }
                State.attachment = {
                    name: file.name || 'Image',
                    kind: 'image',
                    mime_type: mimeType,
                    data: parts[1],
                };
            }

            DOM.attachmentName.textContent = State.attachment.name;
            DOM.attachmentsArea.style.display = 'flex';
            setStatus('ready', 'Attachment ready');
            setTimeout(() => setStatus('ready', 'System Ready'), 1200);
        } catch (err) {
            showErrorToast("Attachment failed: " + (err.message || String(err)));
            setStatus('error', 'Attachment failed');
            setTimeout(() => setStatus('ready', 'System Ready'), 2000);
        }
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
        if (State.medasr.loaded) {
            return true;
        }
        try {
            setStatus('processing', 'Loading MedASR backend...');
            await apiRequest('/transcribe/medasr/load', { method: 'POST' });
            State.medasr.loaded = true;
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
        DOM.captureBtn.disabled = isProc;
        if (DOM.screenshotBtn) DOM.screenshotBtn.disabled = isProc;
        if (isProc) closeAttachmentPicker();

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
        DOM.modelSelect.addEventListener('change', () => {
            if (DOM.modelSelect.value) {
                localStorage.setItem('omnicalc_model', DOM.modelSelect.value);
            }
        });

        DOM.recordBtn.addEventListener('click', toggleRecording);
        DOM.captureBtn.addEventListener('click', handleCaptureAction);
        if (DOM.screenshotBtn) {
            DOM.screenshotBtn.addEventListener('click', startScreenCapture);
        }
        DOM.sendBtn.addEventListener('click', submitRequest);
        DOM.refreshModelsBtn.addEventListener('click', refreshModels);

        if (DOM.attachPickerModal) {
            DOM.attachPickerModal.addEventListener('click', (e) => {
                if (e.target === DOM.attachPickerModal) {
                    closeAttachmentPicker();
                }
            });
        }
        if (DOM.cancelAttachPickerBtn) {
            DOM.cancelAttachPickerBtn.addEventListener('click', closeAttachmentPicker);
        }
        if (DOM.pickFileBtn) {
            DOM.pickFileBtn.addEventListener('click', () => triggerAttachmentInput(DOM.attachFileInput));
        }
        if (DOM.pickImageBtn) {
            DOM.pickImageBtn.addEventListener('click', () => triggerAttachmentInput(DOM.attachImageInput));
        }
        if (DOM.pickCameraBtn) {
            DOM.pickCameraBtn.addEventListener('click', () => triggerAttachmentInput(DOM.attachCameraInput));
        }

        [DOM.attachFileInput, DOM.attachImageInput, DOM.attachCameraInput].forEach((inputEl) => {
            if (!inputEl) return;
            inputEl.addEventListener('change', async () => {
                const file = inputEl.files && inputEl.files[0] ? inputEl.files[0] : null;
                if (file) {
                    await attachFile(file);
                }
                inputEl.value = '';
            });
        });

        if (State.isDesktop) {
            window.addEventListener('dragenter', (e) => {
                if (!hasFileDataTransfer(e)) return;
                e.preventDefault();
                State.dragDropDepth += 1;
                showDragDropOverlay();
            });

            window.addEventListener('dragover', (e) => {
                if (!hasFileDataTransfer(e)) return;
                e.preventDefault();
                if (e.dataTransfer) {
                    e.dataTransfer.dropEffect = 'copy';
                }
                showDragDropOverlay();
            });

            window.addEventListener('dragleave', (e) => {
                if (!hasFileDataTransfer(e)) return;
                e.preventDefault();
                State.dragDropDepth = Math.max(0, State.dragDropDepth - 1);
                if (State.dragDropDepth === 0) {
                    hideDragDropOverlay();
                }
            });

            window.addEventListener('drop', async (e) => {
                if (!hasFileDataTransfer(e)) return;
                e.preventDefault();
                resetDragDropOverlay();
                closeAttachmentPicker();
                const files = Array.from((e.dataTransfer && e.dataTransfer.files) || []);
                if (!files.length) return;
                if (files.length > 1) {
                    showErrorToast('Multiple files dropped. Attaching the first file only.');
                }
                await attachFile(files[0]);
            });

            window.addEventListener('blur', () => {
                resetDragDropOverlay();
            });
        }

        DOM.textInput.addEventListener('input', adjustTextareaHeight);
        DOM.textInput.addEventListener('paste', async (e) => {
            const clipboardFiles = extractClipboardFiles(e);
            if (!clipboardFiles.length) {
                return; // normal text paste path
            }

            e.preventDefault();
            closeAttachmentPicker();

            if (clipboardFiles.length > 1) {
                showErrorToast('Multiple files pasted. Attaching the first file only.');
            }
            await attachFile(clipboardFiles[0]);
        });
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
            clearGeneratingStatusCard();
            State.pendingExecutionCardMsg = null;
            State.pendingExecutionKey = null;
            State.lastCompletedExecutionKey = null;
            State.lastResultKey = null;
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

        DOM.settingsBtn.addEventListener('click', () => {
            if (DOM.themeSelect) DOM.themeSelect.value = State.theme;
            if (DOM.hotkeySelect) DOM.hotkeySelect.value = State.sendHotkey;
            DOM.settingsModal.classList.remove('hidden');
        });
        const closeModals = () => DOM.settingsModal.classList.add('hidden');

        DOM.closeSettingsBtn.addEventListener('click', closeModals);
        DOM.settingsModal.addEventListener('click', (e) => {
            if (e.target === DOM.settingsModal) closeModals();
        });

        DOM.saveSettingsBtn.addEventListener('click', () => {
            if (DOM.themeSelect) {
                State.theme = DOM.themeSelect.value;
                localStorage.setItem('omnicalc_theme', State.theme);
                applyTheme(State.theme);
            }
            if (DOM.hotkeySelect) {
                State.sendHotkey = DOM.hotkeySelect.value;
                localStorage.setItem('omnicalc_send_hotkey', State.sendHotkey);
            }
            closeModals();
            DOM.textInput.focus();
        });

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
                    if (stateKey === 'captureHotkey') applyCaptureModeUI();
                    setTimeout(() => inputEl.blur(), 100);
                }
            });

            if (clearBtn) {
                clearBtn.addEventListener('click', () => {
                    inputEl.value = 'None';
                    State[stateKey] = 'None';
                    try { localStorage.setItem(storageKey, 'None'); } catch (err) { }
                    if (stateKey === 'captureHotkey') applyCaptureModeUI();
                });
            }
        }

        bindHotkeyInput(DOM.asrHotkeyInput, DOM.clearAsrHotkeyBtn, 'asrHotkey', 'omnicalc_asr_hotkey');
        bindHotkeyInput(DOM.captureHotkeyInput, DOM.clearCaptureHotkeyBtn, 'captureHotkey', 'omnicalc_capture_hotkey');

        // Fast Keyboard Shortcuts Mapping
        window.addEventListener('keydown', (e) => {
            if (!DOM.settingsModal || !DOM.settingsModal.classList.contains('hidden')) return;

            const pressed = formatHotkey(e);

            if (e.key === 'Escape' && DOM.attachPickerModal && !DOM.attachPickerModal.classList.contains('hidden')) {
                closeAttachmentPicker();
                return;
            }

            // Screenshot hotkey on desktop, attachment picker hotkey on mobile.
            if (pressed === State.captureHotkey && State.captureHotkey !== 'None') {
                e.preventDefault();
                if (State.isDesktop) {
                    startScreenCapture();
                } else {
                    handleCaptureAction();
                }
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
