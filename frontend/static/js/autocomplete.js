// HireAgent Job Role Autocomplete
// Provides instant, keyboard-accessible role suggestions from the first typed character.
(function () {
    'use strict';

    const ROLE_CATALOG = [
        "AI Developer",
        "AI Engineer",
        "AI Research Scientist",
        "AI/ML Engineer",
        "Analytics Engineer",
        "Applied AI Engineer",
        "Backend Developer",
        "Backend Engineer",
        "Big Data Engineer",
        "Blockchain Developer",
        "Cloud Architect",
        "Cloud Engineer",
        "Computer Vision Engineer",
        "Cybersecurity Analyst",
        "Cybersecurity Engineer",
        "Data Analyst",
        "Data Engineer",
        "Data Scientist",
        "Database Administrator",
        "Deep Learning Engineer",
        "DevOps Engineer",
        "Frontend Developer",
        "Frontend Engineer",
        "Full Stack Developer",
        "Full Stack Engineer",
        "Generative AI Engineer",
        "Java Developer",
        "JavaScript Developer",
        "Machine Learning Engineer",
        "ML Engineer",
        "MLOps Engineer",
        "Mobile App Developer",
        "Mobile Developer",
        "NLP Engineer",
        "Network Engineer",
        "Product Manager (Tech)",
        "Python Developer",
        "Python Engineer",
        "QA Automation Engineer",
        "Quality Assurance Engineer",
        "React Developer",
        "Robotics Engineer",
        "Security Engineer",
        "Site Reliability Engineer",
        "Software Developer",
        "Software Engineer",
        "Solutions Architect",
        "Systems Engineer"
    ];

    function escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function initRoleAutocomplete(input) {
        if (!input || input.dataset.autocompleteAttached) return;
        input.dataset.autocompleteAttached = "true";
        input.setAttribute('autocomplete', 'off');
        input.setAttribute('aria-autocomplete', 'list');
        input.setAttribute('aria-expanded', 'false');

        // Ensure parent wrapper exists for absolute positioning
        let wrapper = input.closest('.ha-autocomplete-wrapper');
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.className = 'ha-autocomplete-wrapper position-relative';
            input.parentNode.insertBefore(wrapper, input);
            wrapper.appendChild(input);
        }

        // Create dropdown container
        const dropdown = document.createElement('ul');
        dropdown.className = 'ha-autocomplete-dropdown d-none';
        dropdown.setAttribute('role', 'listbox');
        dropdown.id = (input.id || 'roleInput') + '-suggestions';
        input.setAttribute('aria-controls', dropdown.id);
        wrapper.appendChild(dropdown);

        let activeIndex = -1;
        let currentSuggestions = [];

        function closeDropdown() {
            dropdown.classList.add('d-none');
            dropdown.innerHTML = '';
            input.setAttribute('aria-expanded', 'false');
            input.removeAttribute('aria-activedescendant');
            activeIndex = -1;
            currentSuggestions = [];
        }

        function filterRoles(query) {
            const cleanQuery = query.trim().toLowerCase();
            if (!cleanQuery) return [];

            // Partition into startsWith vs contains for natural ranking
            const startsWith = [];
            const contains = [];

            for (const role of ROLE_CATALOG) {
                const lowerRole = role.toLowerCase();
                if (lowerRole.startsWith(cleanQuery)) {
                    startsWith.push(role);
                } else if (lowerRole.includes(cleanQuery)) {
                    contains.push(role);
                }
            }

            return [...startsWith, ...contains].slice(0, 8);
        }

        function highlightMatch(role, query) {
            const regex = new RegExp(`(${escapeRegExp(query.trim())})`, 'gi');
            return role.replace(regex, '<span class="match-highlight">$1</span>');
        }

        function renderSuggestions(query) {
            const suggestions = filterRoles(query);
            currentSuggestions = suggestions;
            dropdown.innerHTML = '';
            activeIndex = -1;

            if (suggestions.length === 0) {
                const emptyItem = document.createElement('li');
                emptyItem.className = 'ha-autocomplete-empty';
                emptyItem.textContent = `No matching roles found — press Enter to use "${query.trim()}"`;
                dropdown.appendChild(emptyItem);
                dropdown.classList.remove('d-none');
                input.setAttribute('aria-expanded', 'true');
                return;
            }

            suggestions.forEach((role, idx) => {
                const item = document.createElement('li');
                item.className = 'ha-autocomplete-item';
                item.id = `${dropdown.id}-item-${idx}`;
                item.setAttribute('role', 'option');
                item.setAttribute('aria-selected', 'false');
                item.innerHTML = highlightMatch(role, query);

                // Mouse selection
                item.addEventListener('mousedown', (e) => {
                    e.preventDefault(); // Prevent input blur before value is committed
                    selectRole(role);
                });

                item.addEventListener('mouseenter', () => {
                    setActiveIndex(idx);
                });

                dropdown.appendChild(item);
            });

            dropdown.classList.remove('d-none');
            input.setAttribute('aria-expanded', 'true');
        }

        function setActiveIndex(idx) {
            const items = dropdown.querySelectorAll('.ha-autocomplete-item');
            items.forEach((item, i) => {
                if (i === idx) {
                    item.classList.add('active');
                    item.setAttribute('aria-selected', 'true');
                    input.setAttribute('aria-activedescendant', item.id);
                    item.scrollIntoView({ block: 'nearest' });
                } else {
                    item.classList.remove('active');
                    item.setAttribute('aria-selected', 'false');
                }
            });
            activeIndex = idx;
        }

        function selectRole(role) {
            input.value = role;
            closeDropdown();
            input.focus();
            // Dispatch input and change events so listening scripts know the value changed
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }

        // Input listener (triggers on first character)
        input.addEventListener('input', () => {
            const val = input.value;
            if (val.trim().length >= 1) {
                renderSuggestions(val);
            } else {
                closeDropdown();
            }
        });

        // Focus listener: show suggestions if text already exists
        input.addEventListener('focus', () => {
            const val = input.value;
            if (val.trim().length >= 1) {
                renderSuggestions(val);
            }
        });

        // Keyboard navigation
        input.addEventListener('keydown', (e) => {
            const isOpen = !dropdown.classList.contains('d-none');

            if (e.key === 'ArrowDown') {
                if (!isOpen) {
                    if (input.value.trim().length >= 1) renderSuggestions(input.value);
                    return;
                }
                e.preventDefault();
                if (currentSuggestions.length > 0) {
                    const next = (activeIndex + 1) % currentSuggestions.length;
                    setActiveIndex(next);
                }
            } else if (e.key === 'ArrowUp') {
                if (!isOpen) return;
                e.preventDefault();
                if (currentSuggestions.length > 0) {
                    const prev = (activeIndex - 1 + currentSuggestions.length) % currentSuggestions.length;
                    setActiveIndex(prev);
                }
            } else if (e.key === 'Enter') {
                if (isOpen && activeIndex >= 0 && activeIndex < currentSuggestions.length) {
                    e.preventDefault();
                    selectRole(currentSuggestions[activeIndex]);
                } else {
                    // Allow normal form submission with free-text custom role
                    closeDropdown();
                }
            } else if (e.key === 'Escape') {
                if (isOpen) {
                    e.preventDefault();
                    closeDropdown();
                }
            } else if (e.key === 'Tab') {
                if (isOpen && activeIndex >= 0 && activeIndex < currentSuggestions.length) {
                    selectRole(currentSuggestions[activeIndex]);
                } else {
                    closeDropdown();
                }
            }
        });

        // Close on blur / outside click
        document.addEventListener('click', (e) => {
            if (!wrapper.contains(e.target)) {
                closeDropdown();
            }
        });
    }

    // Auto-initialize on DOM ready for known role input fields
    document.addEventListener('DOMContentLoaded', () => {
        const targetRoleField = document.getElementById('prefTargetRole');
        if (targetRoleField) initRoleAutocomplete(targetRoleField);

        const jobFilterRoleField = document.getElementById('role-select');
        if (jobFilterRoleField) initRoleAutocomplete(jobFilterRoleField);

        // Also bind any inputs marked with data-autocomplete="roles"
        document.querySelectorAll('input[data-autocomplete="roles"]').forEach(initRoleAutocomplete);
    });

    window.initRoleAutocomplete = initRoleAutocomplete;
})();
