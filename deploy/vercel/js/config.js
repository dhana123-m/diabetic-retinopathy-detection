/**
 * RetinaAI — API configuration.
 * In production (Vercel) the /api prefix is rewritten by vercel.json
 * to the ML API host (Render). In local development it points at the
 * locally-running Flask API.
 */
window.RA_CONFIG = {
    base: (function () {
        var host = window.location.hostname;
        if (host === 'localhost' || host === '127.0.0.1') {
            return 'http://127.0.0.1:5000/api';
        }
        return '/api';
    })()
};