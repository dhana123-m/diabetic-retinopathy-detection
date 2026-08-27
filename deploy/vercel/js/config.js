/**
 * RetinaAI — API configuration.
 * In production the frontend talks to the ML API (Render) directly so
 * that predictions aren't subject to the reverse-proxy request timeout.
 * In local development it points at the locally-running Flask API.
 */
window.RA_CONFIG = {
    base: (function () {
        var host = window.location.hostname;
        if (host === 'localhost' || host === '127.0.0.1') {
            return 'http://127.0.0.1:5000/api';
        }
        return 'https://retinaai-api.onrender.com/api';
    })()
};