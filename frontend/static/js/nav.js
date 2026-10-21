// HireAgent Accessible Top Navigation
document.addEventListener('DOMContentLoaded', () => {
    const navToggler = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');

    if (!navToggler || !navMenu) return;

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
            // Move focus to first interactive link inside nav
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

    // Close when a nav link is clicked (e.g. hash anchor or mobile navigation)
    const navLinks = navMenu.querySelectorAll('.nav-link, .btn');
    navLinks.forEach((link) => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 991.98) {
                closeNav();
            }
        });
    });
});
