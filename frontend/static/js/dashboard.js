// HireAgent Resume Matcher Workspace JS
document.addEventListener('DOMContentLoaded', () => {
    const matcherForm = document.getElementById('resumeMatcherForm');
    const resumeFileInput = document.getElementById('dashboardResumeFile');
    const dropZone = document.getElementById('resumeDropZone');
    const fileSelectedName = document.getElementById('fileSelectedName');
    const uploadErrorAlert = document.getElementById('dashboardUploadError');
    const currentResumeId = document.getElementById('currentResumeId');
    const btnFindMatches = document.getElementById('btnFindMatchingJobs');

    const prefTargetRole = document.getElementById('prefTargetRole');
    const prefLocation = document.getElementById('prefLocation');
    const prefCompany = document.getElementById('prefCompany');

    const processingSection = document.getElementById('dashboardProcessingSection');
    const initialSection = document.getElementById('dashboardInitialSection');
    const resultsSection = document.getElementById('dashboardResultsSection');
    const aiProcessingTitle = document.getElementById('aiProcessingTitle');
    const aiProcessingSubtitle = document.getElementById('aiProcessingSubtitle');

    let isSubmitting = false;
    let processingTimer = null;
    let overallTimeoutTimer = null;
    const OVERALL_TIMEOUT_MS = 60000; // 60s client-side timeout for pipeline completion

    function setButtonLoading(loading) {
        if (!btnFindMatches) return;
        if (loading) {
            isSubmitting = true;
            btnFindMatches.disabled = true;
            btnFindMatches.classList.add('disabled');
            btnFindMatches.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span><span class="fw-semibold">Processing...</span>`;
        } else {
            isSubmitting = false;
            btnFindMatches.disabled = false;
            btnFindMatches.classList.remove('disabled');
            btnFindMatches.innerHTML = `<span class="fw-semibold">Find Matching Jobs →</span>`;
        }
    }

    function showError(msg) {
        clearOverallTimeout();
        setButtonLoading(false);
        if (uploadErrorAlert) {
            uploadErrorAlert.innerHTML = `
                <div class="d-flex align-items-center justify-content-between flex-wrap gap-2">
                    <div class="d-flex align-items-center gap-2">
                        <svg class="ha-icon text-danger flex-shrink-0" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                        <span>${msg}</span>
                    </div>
                    <button type="button" class="btn btn-sm btn-outline-light font-mono" id="btnRetryMatching">Try Again</button>
                </div>
            `;
            uploadErrorAlert.classList.remove('d-none');
            uploadErrorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            const retryBtn = document.getElementById('btnRetryMatching');
            if (retryBtn) {
                retryBtn.onclick = () => {
                    clearError();
                    if (matcherForm && !isSubmitting) matcherForm.requestSubmit();
                };
            }
        } else {
            alert(msg);
        }
    }

    function clearError() {
        if (uploadErrorAlert) {
            uploadErrorAlert.textContent = '';
            uploadErrorAlert.classList.add('d-none');
        }
    }

    // Dynamic processing message sequence
    function showProcessingState() {
        if (initialSection) initialSection.classList.add('d-none');
        if (resultsSection) resultsSection.classList.add('d-none');
        if (processingSection) {
            processingSection.classList.remove('d-none');
            processingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        const messages = [
            { title: "Analyzing your resume...", sub: "Evaluating your skills, experience, and role alignment across available jobs..." },
            { title: "Finding relevant jobs...", sub: "Discovering opportunities matching your competencies..." },
            { title: "Ranking your best matches...", sub: "Scoring relevance and preparing your top job recommendations..." }
        ];
        let step = 0;
        if (processingTimer) clearInterval(processingTimer);
        processingTimer = setInterval(() => {
            step = (step + 1) % messages.length;
            if (aiProcessingTitle) aiProcessingTitle.textContent = messages[step].title;
            if (aiProcessingSubtitle) aiProcessingSubtitle.textContent = messages[step].sub;
        }, 2200);

        // Overall safety timeout
        clearTimeout(overallTimeoutTimer);
        overallTimeoutTimer = setTimeout(() => {
            hideProcessingState();
            setButtonLoading(false);
            showError("Resume analysis and matching is taking longer than expected. Please check your network connection or try again.");
        }, OVERALL_TIMEOUT_MS);
    }

    function hideProcessingState() {
        clearOverallTimeout();
        stopPolling();
        if (processingTimer) clearInterval(processingTimer);
        if (processingSection) processingSection.classList.add('d-none');
        if (resultsSection) {
            resultsSection.classList.remove('d-none');
        } else if (initialSection) {
            initialSection.classList.remove('d-none');
        }
    }

    function clearOverallTimeout() {
        if (overallTimeoutTimer) {
            clearTimeout(overallTimeoutTimer);
            overallTimeoutTimer = null;
        }
    }

    // Drag and drop visual cues
    if (dropZone && resumeFileInput) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('border-primary', 'bg-surface-2');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-primary', 'bg-surface-2');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            if (e.dataTransfer && e.dataTransfer.files.length > 0) {
                resumeFileInput.files = e.dataTransfer.files;
                handleFileSelected(resumeFileInput.files[0]);
            }
        });
    }

    function handleFileSelected(file) {
        if (!file) return;
        const allowed = ['.pdf', '.docx'];
        const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (!allowed.includes(ext)) {
            showError('Unsupported file format. Please choose a PDF or DOCX file.');
            resumeFileInput.value = '';
            return;
        }

        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            showError('File size exceeds the 10 MB limit.');
            resumeFileInput.value = '';
            return;
        }

        clearError();
        if (fileSelectedName) {
            fileSelectedName.innerHTML = `<svg class="ha-icon text-accent me-1" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg> <span>${file.name}</span>`;
        }
        const fileActionLabel = document.getElementById('fileActionLabel');
        if (fileActionLabel) {
            fileActionLabel.innerHTML = `<span class="btn btn-sm btn-ha-secondary pointer-events-none" style="pointer-events: none;">Replace Resume</span>`;
        }
    }

    if (resumeFileInput) {
        resumeFileInput.addEventListener('change', () => {
            if (resumeFileInput.files && resumeFileInput.files[0]) {
                handleFileSelected(resumeFileInput.files[0]);
            }
        });
    }

    // Form submission
    if (matcherForm) {
        matcherForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (isSubmitting) return; // Prevent duplicate concurrent submissions
            clearError();

            const file = resumeFileInput && resumeFileInput.files ? resumeFileInput.files[0] : null;
            const hasExistingResume = Boolean(currentResumeId && currentResumeId.value);

            if (!file && !hasExistingResume) {
                showError('Please select or upload a resume file first.');
                return;
            }

            setButtonLoading(true);
            showProcessingState();

            try {
                if (file) {
                    // Upload new file + preferences
                    const formData = new FormData();
                    formData.append('resume_file', file);
                    if (prefTargetRole && prefTargetRole.value.trim()) {
                        formData.append('target_role', prefTargetRole.value.trim());
                    }
                    if (prefLocation && prefLocation.value.trim()) {
                        formData.append('preferred_location', prefLocation.value.trim());
                    }
                    if (prefCompany && prefCompany.value.trim()) {
                        formData.append('preferred_company', prefCompany.value.trim());
                    }

                    const res = await fetch('/dashboard', {
                        method: 'POST',
                        headers: {
                            'X-Requested-With': 'XMLHttpRequest'
                        },
                        body: formData
                    });

                    if (res.ok) {
                        clearError();
                        clearOverallTimeout();
                        // Poll until backend background pipeline finishes and reload automatically
                        pollUntilFinished();
                    } else {
                        const data = await res.json().catch(() => ({}));
                        if (res.status === 401 || (data && data.redirect)) {
                            window.location.href = (data && data.redirect) ? data.redirect : '/login';
                            return;
                        }
                        showError(data.error || 'Failed to process resume. Please try again.');
                        hideProcessingState();
                        setButtonLoading(false);
                    }
                } else {
                    // Match existing resume + preferences
                    const payload = {
                        resume_id: parseInt(currentResumeId.value, 10),
                        target_role: (prefTargetRole && prefTargetRole.value.trim()) ? prefTargetRole.value.trim() : null,
                        preferred_location: (prefLocation && prefLocation.value.trim()) ? prefLocation.value.trim() : null,
                        preferred_company: (prefCompany && prefCompany.value.trim()) ? prefCompany.value.trim() : null
                    };

                    const res = await fetch('/dashboard/rerun', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Requested-With': 'XMLHttpRequest'
                        },
                        body: JSON.stringify(payload)
                    });

                    if (res.ok) {
                        clearError();
                        clearOverallTimeout();
                        // Poll until backend background pipeline finishes and reload automatically
                        pollUntilFinished();
                    } else {
                        const data = await res.json().catch(() => ({}));
                        if (res.status === 401 || (data && data.redirect)) {
                            window.location.href = (data && data.redirect) ? data.redirect : '/login';
                            return;
                        }
                        showError(data.error || 'Failed to re-match jobs. Please try again.');
                        hideProcessingState();
                        setButtonLoading(false);
                    }
                }
            } catch (err) {
                clearOverallTimeout();
                hideProcessingState();
                setButtonLoading(false);
                showError('Network communication error. Please check your connection or try again.');
            }
        });
    }

    // Polling for processing completion with non-overlapping requests and controlled backoff
    let pollAttempts = 0;
    const maxPollAttempts = 40;
    let isPollInFlight = false;
    let pollTimeoutId = null;

    function stopPolling() {
        if (pollTimeoutId) {
            clearTimeout(pollTimeoutId);
            pollTimeoutId = null;
        }
        isPollInFlight = false;
    }

    async function executePollStep() {
        if (document.hidden) {
            // Document hidden: pause active fetch and check back in 2s
            pollTimeoutId = setTimeout(executePollStep, 2000);
            return;
        }

        if (isPollInFlight) {
            return;
        }

        pollAttempts++;
        if (pollAttempts > maxPollAttempts) {
            stopPolling();
            hideProcessingState();
            setButtonLoading(false);
            showError("Resume analysis and matching timed out. Please refresh or try again.");
            return;
        }

        isPollInFlight = true;

        try {
            const controller = new AbortController();
            const fetchTimeout = setTimeout(() => controller.abort(), 8000); // 8s timeout per poll
            let res;
            try {
                res = await fetch('/dashboard/poll', {
                    headers: { 'X-Requested-With': 'XMLHttpRequest' },
                    signal: controller.signal
                });
            } finally {
                clearTimeout(fetchTimeout);
            }

            if (res.status === 401) {
                stopPolling();
                window.location.href = '/login';
                return;
            }

            if (res.ok) {
                const data = await res.json();

                // On transient polling errors reported by backend, preserve last known state and wait
                if (data && data.transient_error) {
                    // Do not update UI or treat as completion; fall through to bounded backoff
                } else if (data && data.is_processing === false) {
                    // Terminal state reached: pipeline completed or stopped
                    stopPolling();
                    clearOverallTimeout();
                    setButtonLoading(false);

                    if (data.last_error) {
                        hideProcessingState();
                        showError(data.last_error);
                    } else {
                        // Mark completion in session storage to prevent auto-polling on reload
                        sessionStorage.setItem('hireagent_matching_finished', 'true');
                        window.location.reload();
                    }
                    return;
                } else if (data && data.is_processing === true) {
                    // Dynamic progress display based on real backend pipeline stages
                    if (data.pipeline && Array.isArray(data.pipeline)) {
                        const currentStage = data.pipeline.find(s => s.status === 'processing');
                        if (currentStage) {
                            if (aiProcessingTitle && currentStage.stage) {
                                aiProcessingTitle.textContent = currentStage.stage.endsWith('...') ? currentStage.stage : `${currentStage.stage}...`;
                            }
                            if (aiProcessingSubtitle && currentStage.detail) {
                                aiProcessingSubtitle.textContent = currentStage.detail;
                            }
                        }
                    }
                }
            }
        } catch (e) {
            // Transient fetch or abort error during poll; continue with backoff
        } finally {
            isPollInFlight = false;
        }

        // Controlled backoff: 1500ms initially, 2000ms after 5 attempts, 2500ms after 15 attempts
        let delayMs = 1500;
        if (pollAttempts > 15) {
            delayMs = 2500;
        } else if (pollAttempts > 5) {
            delayMs = 2000;
        }
        pollTimeoutId = setTimeout(executePollStep, delayMs);
    }

    function pollUntilFinished() {
        stopPolling();
        pollAttempts = 0;
        executePollStep();
    }

    // Completion guard from previous reload
    if (sessionStorage.getItem('hireagent_matching_finished') === 'true') {
        sessionStorage.removeItem('hireagent_matching_finished');
        hideProcessingState();
        stopPolling();
        setButtonLoading(false);
    }

    // If completed results are already visible in DOM, never start automatic polling
    const hasCompletedResults = Boolean(resultsSection && !resultsSection.classList.contains('d-none'));
    if (hasCompletedResults) {
        hideProcessingState();
        stopPolling();
    } else if (processingSection && !processingSection.classList.contains('d-none')) {
        pollUntilFinished();
    }

    // Expand / Collapse All Skills logic
    const toggleSkillsBtn = document.getElementById('toggleAllSkillsBtn');
    const collapseSkillsBtn = document.getElementById('collapseSkillsBtn');
    const allSkillsContainer = document.getElementById('allSkillsContainer');

    if (toggleSkillsBtn && allSkillsContainer) {
        toggleSkillsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            allSkillsContainer.classList.remove('d-none');
            toggleSkillsBtn.classList.add('d-none');
            toggleSkillsBtn.setAttribute('aria-expanded', 'true');
        });
    }

    if (collapseSkillsBtn && allSkillsContainer && toggleSkillsBtn) {
        collapseSkillsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            allSkillsContainer.classList.add('d-none');
            toggleSkillsBtn.classList.remove('d-none');
            toggleSkillsBtn.setAttribute('aria-expanded', 'false');
        });
    }

    // Recommended Role selection pill buttons
    const rolePills = document.querySelectorAll('.role-pill-btn');
    rolePills.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const selectedRole = btn.getAttribute('data-role');
            if (selectedRole && prefTargetRole) {
                prefTargetRole.value = selectedRole;
                rolePills.forEach(p => p.classList.remove('active', 'border-accent', 'text-accent'));
                btn.classList.add('active', 'border-accent', 'text-accent');
                
                const explanationEl = document.getElementById('recommendedRoleExplanation');
                const expText = btn.getAttribute('data-explanation');
                if (explanationEl && expText) {
                    explanationEl.textContent = `Match rationale: ${expText}`;
                }
            }
        });
    });
});
