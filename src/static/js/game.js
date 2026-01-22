document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let sessionId = null;
    let isTyping = false;
    let typingInterval = null;
    let fullText = "";
    let currentPhase = "";
    let isLoading = false; // Global Lock
    let isUserScrolledUp = false; // Smart Auto-Scroll flag
    let evaluationLog = []; // 평가 기록

    // Reference materials storage (initialized at briefing)
    let referenceMaterials = {
        legalContext: '대기 중...',
        sentencingInfo: '대기 중...'
    }; 

    // --- Elements ---
    const bgmPlayer = document.getElementById('bgm-player');
    const btnBgmToggle = document.getElementById('btn-bgm-toggle');
    const startScreen = document.getElementById('start-screen');
    const btnStart = document.getElementById('btn-start-game');
    
    // Main Stage
    const characterContainer = document.getElementById('character-container');
    const characterImg = document.getElementById('character-img');
    const characterPlaceholder = document.getElementById('character-placeholder');
    
    // Dialogue
    const dialogueSection = document.getElementById('dialogue-section'); // New Reference
    const dialogueBox = document.getElementById('dialogue-box'); // Scroll Container
    const speakerNameTag = document.getElementById('speaker-name-tag');
    const dialogueContent = document.getElementById('dialogue-content');
    const referenceText = document.getElementById('reference-text');
    const userInputContainer = document.getElementById('user-input-container');
    const userQueryInput = document.getElementById('user-query-input');
    const btnSendQuery = document.getElementById('btn-send-query');

    // Controls
    const btnNext = document.getElementById('btn-next');
    const actionButtonsArea = document.getElementById('action-buttons-area');
    const btnFinalJudgment = document.getElementById('btn-final-judgment');
    const btnStop = document.getElementById('btn-stop');
    const legalAdvisorShort = document.getElementById('legal-advisor-short');

    // Overlays
    const overlays = {
        'case-info': document.getElementById('overlay-case-info'),
        'history': document.getElementById('overlay-history'),
        'reference': document.getElementById('overlay-reference')
    };

    const menuBtns = {
        'case-info': document.getElementById('btn-case-info'),
        'history': document.getElementById('btn-history'),
        'eval-history': document.getElementById('btn-eval-history'),
        'reference': document.getElementById('btn-reference')
    };

    const globalLoader = document.getElementById('global-loader');
    const interactionLoader = document.getElementById('interaction-loader');

    // Judgment Form Elements
    const judgmentFormOverlay = document.getElementById('overlay-judgment-form');
    const judgmentForm = document.getElementById('judgment-form');
    const prisonYearsInput = document.getElementById('prison-years');
    const prisonMonthsInput = document.getElementById('prison-months');
    const suspensionYearsInput = document.getElementById('suspension-years');
    const suspensionMonthsInput = document.getElementById('suspension-months');
    const judgmentReasoningTextarea = document.getElementById('judgment-reasoning');


    // --- Event Listeners ---
    btnStart.addEventListener('click', startGame);

    btnBgmToggle.addEventListener('click', () => {
        if (bgmPlayer.paused) {
            bgmPlayer.play().then(() => {
                btnBgmToggle.classList.add('active');
                btnBgmToggle.textContent = '🔊';
            }).catch(e => console.log("BGM Play error:", e));
        } else {
            bgmPlayer.pause();
            btnBgmToggle.classList.remove('active');
            btnBgmToggle.textContent = '🎵';
        }
    });
    
    // Dialogue Click Interaction (Replaces btnNext)
    dialogueSection.addEventListener('click', (e) => {
        // ... existing click logic ...
        // Prevent interaction if loading or clicking inside input/buttons
        if (isLoading) return;
        if (e.target.closest('#user-input-container') || e.target.tagName === 'BUTTON') return;

        if (isTyping) {
            finishTyping();
        } else {
            // Block 'next' if we are in a state that requires specific input
            // 1. Judgment Phase: User MUST use the input box
            if (currentPhase === 'judgment' || currentPhase === 'user_judge') return;
            
            // 2. Choices Active: User MUST click a choice button
            // (We check if choice buttons exist in the DOM or state)
            if (actionButtonsArea.children.length > 0) return;

            sendAction('next');
        }
    });
    
    // Smart Auto-Scroll Detection
    dialogueBox.addEventListener('scroll', () => {
        // Check if user is at the bottom (with small tolerance)
        const isAtBottom = (dialogueBox.scrollTop + dialogueBox.clientHeight) >= (dialogueBox.scrollHeight - 10);
        isUserScrolledUp = !isAtBottom;
    });
    
    // User Input (Query)
    btnSendQuery.addEventListener('click', sendUserQuery);
    userQueryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendUserQuery();
    });
    // Stop event propagation from input to dialogue box (prevent double click effect)
    userInputContainer.addEventListener('click', (e) => e.stopPropagation());

    btnFinalJudgment.addEventListener('click', () => {
        if (confirm("모든 변론을 종결하고 최종 판결을 내리시겠습니까?")) {
            // Show judgment form overlay instead of directly entering judgment phase
            showJudgmentForm();
        }
    });

    // Judgment Form Submit
    judgmentForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitJudgment();
    });

    btnStop.addEventListener('click', () => {
        if(confirm("재판을 중단하시겠습니까? (세션 종료)")) {
            returnToStart();
        }
    });

    // Overlay Logic
    Object.keys(menuBtns).forEach(key => {
        if (key === 'eval-history' || key === 'reference') return;

        menuBtns[key].addEventListener('click', () => toggleOverlay(key));
    });

    menuBtns['eval-history'].addEventListener('click', () => {
        showEvaluationHistoryOverlay();
    });

    menuBtns['reference'].addEventListener('click', () => {
        showReferenceOverlay();
    });
    
    document.querySelectorAll('.close-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const overlay = e.target.closest('.overlay');
            overlay.classList.add('hidden');

            // Reset judgment form when closing
            if (overlay.id === 'overlay-judgment-form') {
                resetJudgmentForm();
            }
        });
    });

    // --- Core Logic ---

    async function startGame() {
        if (isLoading) return; // Concurrency check
        showLoading(true, 'global');
        try {
            // Get selected case from sessionStorage
            const selectedCaseStr = sessionStorage.getItem('selectedCase');
            let requestBody = {};

            if (selectedCaseStr) {
                const selectedCase = JSON.parse(selectedCaseStr);
                requestBody = {
                    case_summary: selectedCase.facts || selectedCase.description,
                    case_number: selectedCase.case_number,
                    case_id: selectedCase.id
                };
                // Clear the stored case
                sessionStorage.removeItem('selectedCase');
            }

            // Updated fetch with headers
            const response = await fetch('/api/init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) throw new Error('Failed to start game');
            const data = await response.json();

            sessionId = data.session_id;
            startScreen.classList.add('hidden');

            // Start BGM immediately after user interaction (Start button click)
            if (bgmPlayer && bgmPlayer.paused) {
                bgmPlayer.play().then(() => {
                    btnBgmToggle.classList.add('active');
                    btnBgmToggle.textContent = '🔊';
                }).catch(e => console.log("BGM Play error on start:", e));
            }

            updateUI(data);

        } catch (error) {
            console.error(error);
            alert('게임 시작 중 오류가 발생했습니다.');
        } finally {
            showLoading(false);
        }
    }

    async function sendAction(actionType, payload = {}) {
        if (!sessionId || isLoading) return; // Concurrency check
        
        if (isTyping && actionType === 'next') {
            finishTyping();
            return;
        }

        showLoading(true, 'interaction');
        try {
            const response = await fetch('/api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    action_type: actionType,
                    payload: payload
                })
            });
            
            if (!response.ok) throw new Error('Action failed');
            const data = await response.json();
            updateUI(data);

        } catch (error) {
            console.error(error);
            alert('통신 중 오류가 발생했습니다.');
        } finally {
            showLoading(false);
        }
    }

    function sendUserQuery() {
        if (isLoading) return; // Concurrency check
        const text = userQueryInput.value.trim();
        if (!text) return;
        
        userQueryInput.value = '';
        console.log("userQueryInput: "+text);
        console.log("currentPhase: "+currentPhase);
        // Determine action type based on phase
        if (currentPhase === 'judgment') {
            sendAction('judgment', { user_text: text });
        } else {
            // General input (interjection or question) during debate
            sendAction('next', { user_input: text });
        }
    }

    function updateUI(state) {
        console.log("State Update:", state);
        currentPhase = state.current_phase || "";

        // BGM Control: Play when game is active (from Briefing onwards)
        /*
        if (bgmPlayer && (currentPhase === 'briefing' || currentPhase === 'debate' || currentPhase === 'user_judge')) {
            if (bgmPlayer.paused) {
                bgmPlayer.play().then(() => {
                    btnBgmToggle.classList.add('active');
                    btnBgmToggle.textContent = '🔊';
                }).catch(e => console.log("BGM Autoplay prevented:", e));
            }
        }
        */
        // (new)=== 공방 평가 팝업 표시 ===
        if (state.evaluations_log) {
            evaluationLog = state.evaluations_log;      // 평가 기록 누적
        }

        if (currentPhase === 'user_judge' && state.round_summary) {
            showEvaluationOverlay(state.round_summary);
        }

        // === 판결 분석 결과 표시 ===
        if (currentPhase === 'result') {
            showAnalysisResult(state);
        }

        // 1. Character & Position
        const speaker = state.speaker || "system";
        updateCharacterPosition(speaker, state.emotion);

        // 2. Dialogue
        speakerNameTag.textContent = getSpeakerNameKR(speaker);
        
        // Set Name Tag Color Class
        speakerNameTag.className = ''; // reset
        if (speaker.includes('prosecutor')) speakerNameTag.classList.add('tag-prosecutor');
        else if (speaker.includes('defense')) speakerNameTag.classList.add('tag-defense');
        else if (speaker.includes('judge') || speaker.includes('user')) speakerNameTag.classList.add('tag-judge');

        startTyping(state.content || "");

        // 3. Info - Always update, even if empty
        referenceText.textContent = (state.references || []).join(', ');

        // Store reference materials on briefing phase (initial data)
        if (currentPhase === 'briefing' || currentPhase === 'debate') {
            if (state.legal_context !== undefined) {
                referenceMaterials.legalContext = state.legal_context || '정보가 제공되지 않았습니다.';
            }
            if (state.sentencing_info !== undefined) {
                referenceMaterials.sentencingInfo = state.sentencing_info || '정보가 제공되지 않았습니다.';
            }
        }

        // Case Info - always update
        if (state.case_info !== undefined) {
            document.getElementById('case-info-text').innerHTML = formatText(state.case_info) || '대기 중...';
        }

        // 5. History Update
        if (state.history && state.history.length > 0) {
            const historyList = document.getElementById('history-list');
            historyList.innerHTML = ''; // Clear
            
            // Show recent messages first (reverse loop or flex-direction: column-reverse?)
            // Usually history shows chronological top to bottom.
            state.history.forEach(msg => {
                const item = document.createElement('div');
                item.className = 'history-item';
                item.style.marginBottom = '15px';
                item.style.borderBottom = '1px solid #333';
                item.style.paddingBottom = '10px';
                
                const roleSpan = document.createElement('span');
                roleSpan.textContent = getSpeakerNameKR(msg.role);
                roleSpan.style.fontWeight = 'bold';
                roleSpan.style.color = getRoleColor(msg.role);
                roleSpan.style.display = 'block';
                roleSpan.style.marginBottom = '5px';
                
                const contentDiv = document.createElement('div');
                contentDiv.textContent = msg.content;
                contentDiv.style.color = '#ccc';
                
                item.appendChild(roleSpan);
                item.appendChild(contentDiv);
                historyList.appendChild(item);
            });
            
            // Scroll to bottom
            historyList.scrollTop = historyList.scrollHeight;
        }

        // 4. Controls & Input
        updateControls(state);
    }

    function getRoleColor(role) {
        if (role.includes('prosecutor')) return '#c0392b';
        if (role.includes('defense')) return '#3498db';
        if (role.includes('judge') || role.includes('user')) return '#f39c12';
        return '#aaa';
    }

    function updateCharacterPosition(speaker, emotion) {
        // Reset classes
        characterContainer.className = '';
        
        // Logic: Defense(Left), Prosecutor(Right), Judge/System(Center)
        if (speaker.includes('defense')) {
            characterContainer.classList.add('pos-left');
        } else if (speaker.includes('prosecutor')) {
            characterContainer.classList.add('pos-right');
        } else { // Judge, System, User_Judge
            characterContainer.classList.add('pos-center');
        }

        const desiredEmotion = emotion && emotion !== 'neutral' ? emotion : 'neutral';
        let initialImagePath = `/static/images/${speaker}_${desiredEmotion}.png`;

        // Function to attempt loading an image
        const loadImage = (path, isFallback = false) => {
            characterImg.onload = () => {
                characterImg.style.display = 'block';
                characterPlaceholder.style.display = 'none'; // Ensure placeholder is hidden
            };
            characterImg.onerror = () => {
                if (!isFallback) { // If original image failed, try neutral
                    console.warn(`Image for ${speaker}_${desiredEmotion}.png not found. Trying ${speaker}_neutral.png`);
                    loadImage(`/static/images/${speaker}_neutral.png`, true);
                } else { // If neutral also failed
                    console.error(`Image for ${speaker}_neutral.png also not found. Displaying text placeholder.`);
                    characterImg.style.display = 'none';
                    // system 등 별도 캐릭터들에 대한 이미지도 찾으려 하기에 일단 주석 처리
                    //characterPlaceholder.style.display = 'flex';
                    //characterPlaceholder.textContent = `${getSpeakerNameKR(speaker)}\n(${emotion || 'neutral'}) - Image N/A`;
                }
            };
            characterImg.src = path;
        };

        // Start loading the desired emotion image
        loadImage(initialImagePath);

        // Animation trigger (simple fade)
        characterContainer.animate([
            { opacity: 0.5, transform: 'scale(0.98)' },
            { opacity: 1, transform: 'scale(1)' }
        ], { duration: 300 });
    }

    function updateControls(state) {
        actionButtonsArea.innerHTML = ''; // Clear dynamic buttons

        const currentPhase = state.current_phase;

        // ===== CRITICAL: Block ALL user input after judgment submission =====
        if (currentPhase === 'result') {
            // Hide user input completely - game is over
            userInputContainer.style.display = 'none';
            btnNext.classList.add('hidden');
            btnFinalJudgment.classList.add('hidden');

            // Disable all interactive elements
            userQueryInput.disabled = true;
            btnSendQuery.disabled = true;

            // Show "Return to Start" button in action area
            const returnBtn = document.createElement('button');
            returnBtn.className = 'action-btn primary-btn';
            returnBtn.textContent = '🏠 처음으로';
            returnBtn.style.marginTop = '20px';
            returnBtn.onclick = () => {
                if (confirm("초기 화면으로 돌아가시겠습니까?")) {
                    returnToStart();
                }
            };
            actionButtonsArea.appendChild(returnBtn);

            return; // Stop processing - no more user interaction allowed
        }

        // ===== Reset visibility for active phases =====
        userInputContainer.style.display = 'flex';
        userQueryInput.disabled = false;
        btnSendQuery.disabled = false;

        // Logic for Input Field visibility during debate/user_judge phases
        const isJudgment = currentPhase === 'judgment' || state.speaker === 'user_judge';

        if (isJudgment) {
            btnNext.classList.add('hidden'); // Hide simple Next
            // Note: judgment phase is now handled by form overlay, not this input
            userQueryInput.placeholder = "질문이나 이의를 입력하세요.";
            btnFinalJudgment.classList.remove('hidden'); // Show final judgment button
        } else {
            btnNext.classList.remove('hidden');
            userQueryInput.placeholder = "질문이나 이의를 입력하세요.";
            btnFinalJudgment.classList.remove('hidden'); // Keep visible during debate
        }

        // Choices (if any)
        if (state.choices && state.choices.length > 0) {
            btnNext.classList.add('hidden'); // Hide Next, force choice
            state.choices.forEach(choice => {
                const btn = document.createElement('button');
                btn.className = 'choice-btn';
                btn.textContent = choice.label;
                btn.onclick = () => sendAction('choice', { choice_id: choice.id });
                actionButtonsArea.appendChild(btn);
            });
        }
    }

    // 종합 평가(Round Summary) UI 개선
    function showEvaluationOverlay(round_summary) {
        const overlay = document.getElementById("overlay-evaluation");
        const container = document.getElementById("evaluation-results");
        container.innerHTML = "";

        const verdictColor = round_summary.verdict === "prosecutor" ? "#c0392b" : "#3498db";
        const verdictLabel = round_summary.verdict === "prosecutor" ? "⚔️ 검사 측 우세" : "🛡️ 변호인 측 우세";

        container.innerHTML = `
            <h3><div class="verdict-banner" style="color:${verdictColor}">
                ${verdictLabel}
            </div><h3>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1; border-right: 1px solid #444; padding-right: 10px;">
                    <p style="color:#c0392b; font-weight:bold;">⚔️ 검사 리포트</p>
                    <p style="font-size:0.9rem;">${round_summary.prosecutor_summary}</p>
                </div>
                <div style="flex: 1; padding-left: 10px;">
                    <p style="color:#3498db; font-weight:bold;">🛡️ 변호인 리포트</p>
                    <p style="font-size:0.9rem;">${round_summary.defense_summary}</p>
                </div>
            </div>
            <div class="eval-feedback-box" style="margin-top:20px; border-left-color: #f1c40f;">
                <strong>종합 판단 이유</br></strong> ${round_summary.reason}
            </div>
        `;

        overlay.classList.remove("hidden");
    }
    

    function showEvaluationHistoryOverlay() {
        const overlay = document.getElementById("overlay-evaluationlog");
        const container = document.getElementById("evaluation-history");
        container.innerHTML = "";

        if (!evaluationLog || evaluationLog.length === 0) {
            container.innerHTML = "<p>아직 판사의 평가 기록이 없습니다.</p>";
        } else {
            evaluationLog.forEach((evalItem, idx) => {
                const div = document.createElement("div");
                div.classList.add("evaluation-block");
                div.innerHTML = `
                    <p><strong>🔥 ${evalItem.round ?? idx} 라운드</strong> </p>
                    <p><strong>🧑 발언자:</strong> ${getSpeakerNameKR(evalItem.speaker)}</p>
                    <p><strong>🎯 점수:</strong> ${evalItem.score} / 10</p>
                    <p><strong>👁️ 사실 검증 </br></strong> ${evalItem.fact_check}</p>
                    <p><strong>🦠 논리적 허점 </br></strong> ${evalItem.logical_flaw}</p>
                    <p><strong>💬 AI판사 의견 </br></strong> ${evalItem.feedback}</p>
                    <hr>
                `;
                container.appendChild(div);
            });
        }

        overlay.classList.remove("hidden");
    }

    function showAnalysisResult(state) {
        console.log("AnalysisResultState");
        console.log(state);
        // Extract analysis data from state
        const analysisResult = state.analysis_result || {};

        // Get user judgment data from state
        const userVerdict = state.user_verdict || 'guilty';
        const userSentenceText = state.user_sentence_text || '';
        const userReasoning = state.user_reasoning || '';

        // Get actual judgment data
        const actualJudgment = state.actual_judgment || {};
        let actualLabel = actualJudgment.actual_label || '정보 없음';
        let actualRule = actualJudgment.actual_rule || '';
        const actualReason = actualJudgment.actual_reason || '정보 없음';

        // Handle object structures for label and rule
        if (typeof actualLabel === 'object' && actualLabel !== null) {
             actualLabel = actualLabel.text || '';
        }
        
        if (typeof actualRule === 'object' && actualRule !== null) {
             actualRule = actualRule.text || '';
        }

        // Format user verdict display
        const userVerdictDisplay = userSentenceText || (userVerdict === 'guilty' ? '유죄' : '무죄');

        // Format actual verdict display
        const actualVerdictDisplay = actualRule ?
            `${actualLabel} (${actualRule})` :
            actualLabel;

        // Update overlay content
        document.getElementById('user-verdict-text').textContent = userVerdictDisplay;
        document.getElementById('user-reasoning-text').textContent = userReasoning || '-';

        document.getElementById('actual-verdict-text').textContent = actualVerdictDisplay;
        document.getElementById('actual-reasoning-text').textContent = actualReason || '-';

        document.getElementById('analysis-summary').textContent =
            analysisResult.comparison_summary || '-';
        document.getElementById('analysis-strength').textContent =
            analysisResult.user_strength || '-';
        document.getElementById('analysis-weakness').textContent =
            analysisResult.user_weakness || '-';

        // Update overlooked factors list
        const overlookedList = document.getElementById('analysis-overlooked');
        overlookedList.innerHTML = '';
        if (analysisResult.overlooked_factors && analysisResult.overlooked_factors.length > 0) {
            analysisResult.overlooked_factors.forEach(factor => {
                const li = document.createElement('li');
                li.textContent = factor;
                overlookedList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = '없음';
            overlookedList.appendChild(li);
        }

        // Update learning points list
        const learningList = document.getElementById('analysis-learning');
        learningList.innerHTML = '';
        if (analysisResult.learning_points && analysisResult.learning_points.length > 0) {
            analysisResult.learning_points.forEach(point => {
                const li = document.createElement('li');
                li.textContent = point;
                learningList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = '없음';
            learningList.appendChild(li);
        }

        // Show the overlay
        document.getElementById('overlay-analysis-result').classList.remove('hidden');
    }

    function showReferenceOverlay() {
        // Update reference overlay content with stored materials
        document.getElementById('reference-legal-content').innerHTML =
            referenceMaterials.legalContext || '정보가 제공되지 않았습니다.';

        document.getElementById('reference-sentencing-content').innerHTML =
            referenceMaterials.sentencingInfo || '정보가 제공되지 않았습니다.';

        // Show the overlay
        overlays['reference'].classList.remove('hidden');
    }


    // --- Utilities ---
    function startTyping(text) {
        fullText = text;
        dialogueContent.textContent = "";
        isTyping = true;
        isUserScrolledUp = false; // Reset scroll state on new turn
        clearInterval(typingInterval);
        
        let index = 0;
        typingInterval = setInterval(() => {
            dialogueContent.textContent += fullText.charAt(index);
            
            // Smart Auto-Scroll: Only scroll if user hasn't scrolled up
            if (!isUserScrolledUp) {
                dialogueBox.scrollTop = dialogueBox.scrollHeight;
            }
            
            index++;
            if (index >= fullText.length) {
                finishTyping();
            }
        }, 20); // Faster speed for game feel
    }

    function finishTyping() {
        clearInterval(typingInterval);
        dialogueContent.textContent = fullText;
        isTyping = false;
        
        // Final scroll update if user hasn't scrolled up
        if (!isUserScrolledUp) {
            dialogueBox.scrollTop = dialogueBox.scrollHeight;
        }
    }

    function toggleOverlay(id) {
        overlays[id].classList.toggle('hidden');
    }

    function showLoading(show, type = 'interaction') {
        isLoading = show; // Update global lock
        
        const loader = type === 'global' ? globalLoader : interactionLoader;
        
        if (show) {
            loader.classList.remove('hidden');
        } else {
            // Hide both to be safe, or track which one was shown.
            // Simple approach: hide both on false
            globalLoader.classList.add('hidden');
            interactionLoader.classList.add('hidden');
        }
    }

    function formatText(text) {
        return text ? text.replace(/\n/g, '<br>') : "";
    }

    function getSpeakerNameKR(role) {
        const map = {
            'prosecutor': '검사 (Prosecutor)',
            'defense': '변호인 (Defense)',
            'judge': '판사 (Judge)',
            'user_judge': '재판장 (YOU)',
            'legal_advisor': '법률 자문',
            'system': 'System',
            'clerk': '서기'
        };
        // Partial match check
        for (const key in map) {
            if (role.includes(key)) return map[key];
        }
        return role;
    }

    // --- Judgment Form Functions ---

    function showJudgmentForm() {
        resetJudgmentForm();
        judgmentFormOverlay.classList.remove('hidden');
    }

    function resetJudgmentForm() {
        prisonYearsInput.value = '0';
        prisonMonthsInput.value = '0';
        suspensionYearsInput.value = '0';
        suspensionMonthsInput.value = '0';
        judgmentReasoningTextarea.value = '';
    }

    async function submitJudgment() {
        if (isLoading) return;

        // Get form values
        const prisonYears = parseInt(prisonYearsInput.value) || 0;
        const prisonMonths = parseInt(prisonMonthsInput.value) || 0;
        const suspensionYears = parseInt(suspensionYearsInput.value) || 0;
        const suspensionMonths = parseInt(suspensionMonthsInput.value) || 0;
        const reasoning = judgmentReasoningTextarea.value.trim();

        // Validation: Must have at least prison sentence or reasoning
        if (prisonYears === 0 && prisonMonths === 0 && !reasoning) {
            alert('형량 또는 양형 이유를 입력해주세요.');
            return;
        }

        // Format sentence text
        let sentenceText = '';
        if (prisonYears > 0 || prisonMonths > 0) {
            sentenceText = '징역 ';
            if (prisonYears > 0) sentenceText += `${prisonYears}년`;
            if (prisonMonths > 0) sentenceText += ` ${prisonMonths}월`;
        }

        if (suspensionYears > 0 || suspensionMonths > 0) {
            if (sentenceText) sentenceText += ', ';
            sentenceText += '집행유예 ';
            if (suspensionYears > 0) sentenceText += `${suspensionYears}년`;
            if (suspensionMonths > 0) sentenceText += ` ${suspensionMonths}월`;
        }

        // Hide form
        judgmentFormOverlay.classList.add('hidden');

        // Send structured judgment data
        showLoading(true, 'interaction');
        try {
            const response = await fetch('/api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    action_type: 'judgment',
                    payload: {
                        verdict: 'guilty',  // Always guilty as per requirement
                        sentence: {
                            prison_years: prisonYears,
                            prison_months: prisonMonths,
                            suspension_years: suspensionYears,
                            suspension_months: suspensionMonths,
                            // fine: 0  // Commented out for future use
                        },
                        sentence_text: sentenceText,
                        reasoning: reasoning
                    }
                })
            });

            if (!response.ok) throw new Error('Judgment submission failed');
            const data = await response.json();
            updateUI(data);

        } catch (error) {
            console.error(error);
            alert('판결 전송 중 오류가 발생했습니다.');
        } finally {
            showLoading(false);
        }
    }

    // --- Return to Start Screen ---
    async function returnToStart() {
        if (bgmPlayer) {
            bgmPlayer.pause();
            bgmPlayer.currentTime = 0;
            btnBgmToggle.classList.remove('active');
            btnBgmToggle.textContent = '🎵';
        }

        // Cleanup session resources on server
        if (sessionId) {
            try {
                await fetch('/api/cleanup-session', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                });
                console.log('Session cleaned up successfully');
            } catch (error) {
                console.error('Failed to cleanup session:', error);
            }
        }

        // Reset all state
        sessionId = null;
        isTyping = false;
        isLoading = false;
        evaluationLog = [];
        referenceMaterials = {
            legalContext: '대기 중...',
            sentencingInfo: '대기 중...'
        };

        /*
        // Clear UI elements
        dialogueContent.textContent = '';
        speakerNameTag.textContent = 'System';
        referenceText.textContent = '';
        actionButtonsArea.innerHTML = '';

        // Reset character
        characterImg.style.display = 'none';
        characterPlaceholder.style.display = 'none';

        // Reset input
        userQueryInput.value = '';
        userQueryInput.disabled = false;
        btnSendQuery.disabled = false;

        // Reset visibility
        userInputContainer.style.display = 'flex';
        btnNext.classList.remove('hidden');
        btnFinalJudgment.classList.remove('hidden');
        */
        // Redirect to scenario selection page instead of showing start screen
        window.location.href = '/';
    }
});