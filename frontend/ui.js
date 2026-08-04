/* =====================================================
   CorpoCalm Gateway — Shared UI Utilities
   ===================================================== */

// ── Toast Notification System ──
(function () {
    let container = null;

    function getContainer() {
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        return container;
    }

    const ICONS = {
        success: '✅',
        error:   '❌',
        warning: '⚠️',
        info:    'ℹ️',
    };

    window.showToast = function (message, type = 'info', title = '', duration = 3500) {
        const c = getContainer();
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const resolvedTitle = title || { success: 'Success', error: 'Error', warning: 'Warning', info: 'Info' }[type];

        toast.innerHTML = `
            <span class="toast-icon">${ICONS[type] || ICONS.info}</span>
            <div class="toast-body">
                <div class="toast-title">${resolvedTitle}</div>
                <div class="toast-msg">${message}</div>
            </div>`;

        c.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-out');
            toast.addEventListener('animationend', () => toast.remove(), { once: true });
        }, duration);
    };
})();
