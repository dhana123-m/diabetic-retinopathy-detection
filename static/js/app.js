/**
 * RetinaAI — Main JavaScript Module
 * Handles navigation and notifications
 */

document.addEventListener('DOMContentLoaded', function () {
    // ─── Mobile Navigation ───
    var navToggle = document.getElementById('navToggle') || document.querySelector('.nav-toggle');
    var navLinks = document.getElementById('navLinks') || document.querySelector('.nav-links');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function (e) {
            e.stopPropagation();
            navLinks.classList.toggle('open');
            navToggle.textContent = navLinks.classList.contains('open') ? '\u2715' : '\u2630';
        });

        document.addEventListener('click', function (e) {
            if (!navLinks.contains(e.target) && !navToggle.contains(e.target)) {
                navLinks.classList.remove('open');
                navToggle.textContent = '\u2630';
            }
        });
    }
});

/**
 * Format confidence value as percentage string
 */
function formatConfidence(value) {
    return (value * 100).toFixed(1) + '%';
}

/**
 * Get risk badge class from risk level string
 */
function getRiskClass(riskLevel) {
    return 'risk-' + riskLevel.toLowerCase().replace(/[- ]/g, '');
}

/**
 * Show a notification toast message
 */
function showNotification(message, type) {
    var existing = document.querySelector('.notification');
    if (existing) existing.remove();

    var notification = document.createElement('div');
    notification.className = 'notification notification-' + (type || 'info');
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(function () {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(20px)';
        notification.style.transition = 'opacity 0.3s, transform 0.3s';
        setTimeout(function () { notification.remove(); }, 300);
    }, 4000);
}

/**
 * Scroll to element
 */
function scrollToElement(selector) {
    var el = document.querySelector(selector);
    if (el) {
        el.scrollIntoView({ behavior: 'auto', block: 'start' });
    }
}