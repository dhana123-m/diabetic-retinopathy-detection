/**
 * RetinaAI — API client.
 * Thin wrapper around fetch for all ML API endpoints.
 */
(function () {
    'use strict';

    function api(path, options) {
        var url = window.RA_CONFIG.base + path;
        return fetch(url, options).then(function (resp) {
            return resp.json().then(function (data) {
                if (data && data.error) {
                    var err = new Error(data.error);
                    err.status = resp.status;
                    throw err;
                }
                if (!resp.ok) {
                    var err2 = new Error('Request failed (' + resp.status + ')');
                    err2.status = resp.status;
                    throw err2;
                }
                return data;
            });
        });
    }

    window.RA_API = {
        predict: function (formData) {
            return api('/predict', { method: 'POST', body: formData });
        },
        health: function () {
            return api('/health');
        },
        warmUp: function () {
            return fetch(window.RA_CONFIG.base + '/health').then(function (resp) {
                return resp.json();
            }).catch(function () {
                return null;
            });
        },
        stats: function () {
            return api('/stats');
        },
        history: function () {
            return api('/history');
        },
        historyDetail: function (id) {
            return api('/history/' + encodeURIComponent(id));
        },
        clearHistory: function () {
            return api('/history', { method: 'DELETE' });
        },
        media: function (file) {
            return window.RA_CONFIG.base + '/media/' + encodeURIComponent(file);
        }
    };
})();