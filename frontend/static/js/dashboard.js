// HireAgent Resume Matcher Workspace JS
document.addEventListener('DOMContentLoaded', () => {
    const matcherForm = document.getElementById('resumeMatcherForm');
    const resumeFileInput = document.getElementById('dashboardResumeFile');
    const dropZone = document.getElementById('resumeDropZone');
    const fileSelectedName = document.getElementById('fileSelectedName');
    const uploadErrorAlert = document.getElementById('dashboardUploadError');
    const currentResumeId = document.getElementById('currentResumeId');

    const prefTargetRole = document.getElementById('prefTargetRole');
    const prefLocation = document.getElementById('prefLocation');
    const prefCompany = document.getElementById('prefCompany');

    const processingSection = document.getElementById('dashboardProcessingSection');
    const initialSection = document.getElementById('dashboardInitialSection');
    const resultsSection = document.getElementById('dashboardResultsSection');
    const aiProcessingTitle = document.getElementById('aiProcessingTitle');
    const aiProcessingSubtitle = document.getElementById('aiProcessingSubtitle');

    function showError(msg) {
        if (uploadErrorAlert) {
            uploadErrorAlert.innerHTML = `
                <div class="d-flex align-items-center justify-content-between flex-wrap gap-2">
                    <span>${msg}</span>
                    <button type="button" class="btn btn-sm btn-outline-light font-mono" onclick="document.getElementById('btnFindMatchingJobs').click()">Try Again</button>
                </div>
            `;
            uploadErrorAlert.classList.remove('d-none');
            uploadErrorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
    let processingTimer = null;
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
    }

    function hideProcessingState() {
        if (processingTimer) clearInterval(processingTimer);
        if (processingSection) processingSection.classList.add('d-none');
        if (resultsSection) {
            resultsSection.classList.remove('d-none');
        } else if (initialSection) {
            initialSection.classList.remove('d-none');
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
            clearError();

            const file = resumeFileInput && resumeFileInput.files ? resumeFileInput.files[0] : null;
            const hasExistingResume = Boolean(currentResumeId && currentResumeId.value);

            if (!file && !hasExistingResume) {
                showError('Please select or upload a resume file first.');
                return;
            }

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
                        pollUntilFinished();
                    } else {
                        const data = await res.json().catch(() => ({}));
                        if (res.status === 401 || (data && data.redirect)) {
                            window.location.href = (data && data.redirect) ? data.redirect : '/login';
                            return;
                        }
                        showError(data.error || 'Failed to process resume. Please try again.');
                        hideProcessingState();
                    }
                } else {
                    // Match existing resume + preferences
                    const payload = {
                        resume_id: parseInt(currentResumeId.value, 10),
                        target_role: prefTargetRole ? prefTargetRole.value.trim() : null,
                        preferred_location: prefLocation ? prefLocation.value.trim() : null,
                        preferred_company: prefCompany ? prefCompany.value.trim() : null
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
                        pollUntilFinished();
                    } else {
                        const data = await res.json().catch(() => ({}));
                        if (res.status === 401 || (data && data.redirect)) {
                            window.location.href = (data && data.redirect) ? data.redirect : '/login';
                            return;
                        }
                        showError(data.error || 'Failed to re-match jobs. Please try again.');
                        hideProcessingState();
                    }
                }
            } catch (err) {
                // If fetch fails, try traditional form submit
                matcherForm.submit();
            }
        });
    }

    // Polling for processing completion
    let pollInterval = 2000;
    let pollAttempts = 0;
    const maxPollAttempts = 15;

    async function pollUntilFinished() {
        pollAttempts = 0;
        const intervalId = setInterval(async () => {
            if (document.hidden) return;

            pollAttempts++;
            if (pollAttempts > maxPollAttempts) {
                clearInterval(intervalId);
                window.location.reload();
                return;
            }

            try {
                const res = await fetch('/dashboard/poll');
                if (res.status === 401) {
                    clearInterval(intervalId);
                    window.location.href = '/login';
                    return;
                }
                if (res.ok) {
                    const data = await res.json();
                    const isProcessing = Boolean(data.is_processing);
                    if (!isProcessing) {
                        clearInterval(intervalId);
                        window.location.reload();
                    }
                }
            } catch (e) {
                // Ignore transient network errors during poll
            }
        }, pollInterval);
    }

    if (processingSection && !processingSection.classList.contains('d-none')) {
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
});
