// HireAgent Accessible Top Navigation & Notification System
document.addEventListener('DOMContentLoaded', () => {
    // ── 1. Top Navigation Bar ──
    const navToggler = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');

    if (navToggler && navMenu) {
        function closeNav() {
            if (navMenu.classList.contains('show')) {
                navMenu.classList.remove('show');
                navToggler.setAttribute('aria-expanded', 'false');
                navToggler.focus();
            }
        }

        function toggleNav() {
            const isExpanded = navMenu.classList.contains('show');
            if (isExpanded) {
                navMenu.classList.remove('show');
                navToggler.setAttribute('aria-expanded', 'false');
            } else {
                navMenu.classList.add('show');
                navToggler.setAttribute('aria-expanded', 'true');
                const firstLink = navMenu.querySelector('a, button');
                if (firstLink) firstLink.focus();
            }
        }

        navToggler.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleNav();
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && navMenu.classList.contains('show')) {
                closeNav();
            }
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (navMenu.classList.contains('show') && !navMenu.contains(e.target) && !navToggler.contains(e.target)) {
                closeNav();
            }
        });

        // Close when a nav link is clicked on mobile
        const navLinks = navMenu.querySelectorAll('.nav-link, .btn');
        navLinks.forEach((link) => {
            link.addEventListener('click', () => {
                if (window.innerWidth <= 991.98) {
                    closeNav();
                }
            });
        });
    }

    // ── 2. Notifications System (Auto-Dismiss, Smooth Upward Fade, Deduplication) ──
    initNotificationSystem();
});

function dismissAlert(alertEl) {
    if (!alertEl || alertEl.classList.contains('ha-dismissing')) return;
    alertEl.classList.add('ha-dismissing');
    alertEl.classList.remove('show');
    setTimeout(() => {
        if (alertEl.parentElement) {
            const container = alertEl.closest('#haFlashContainer');
            alertEl.remove();
            if (container && container.querySelectorAll('.alert').length === 0) {
                container.remove();
            }
        }
    }, 380);
}

function initNotificationSystem() {
    const flashContainer = document.getElementById('haFlashContainer');
    if (!flashContainer) return;

    // Deduplicate identical notifications displayed in this container
    const seenTexts = new Set();
    const alerts = flashContainer.querySelectorAll('.alert');
    alerts.forEach((alertEl) => {
        const text = alertEl.innerText.trim();
        if (seenTexts.has(text)) {
            alertEl.remove();
            return;
        }
        seenTexts.add(text);

        // Auto-dismiss setup
        const autoDismissAttr = alertEl.getAttribute('data-auto-dismiss');
        const timeoutMs = autoDismissAttr ? parseInt(autoDismissAttr, 10) : 0;

        if (timeoutMs > 0) {
            let dismissTimer = null;
            let remainingMs = timeoutMs;
            let startTime = Date.now();

            const startTimer = () => {
                startTime = Date.now();
                dismissTimer = setTimeout(() => {
                    dismissAlert(alertEl);
                }, remainingMs);
            };

            const pauseTimer = () => {
                if (dismissTimer) {
                    clearTimeout(dismissTimer);
                    dismissTimer = null;
                    remainingMs -= (Date.now() - startTime);
                    if (remainingMs < 500) remainingMs = 500;
                }
            };

            startTimer();
            alertEl.addEventListener('mouseenter', pauseTimer);
            alertEl.addEventListener('mouseleave', startTimer);
        }

        // Close button click
        const closeBtn = alertEl.querySelector('.btn-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.preventDefault();
                dismissAlert(alertEl);
            });
        }
    });

    // Global Escape key dismiss for visible flash notifications
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const openAlerts = document.querySelectorAll('#haFlashContainer .alert.show');
            openAlerts.forEach((alertEl) => dismissAlert(alertEl));
        }
    });
}

// Prevent stale flash notifications when restored from bfcache
window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        document.querySelectorAll('#haFlashContainer .alert[data-auto-dismiss]').forEach((el) => {
            el.remove();
        });
    }
});

