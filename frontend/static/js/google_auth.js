// HireAgent Google Sign-In with Firebase Web SDK
(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", function () {
        const btn = document.getElementById("google-signin-btn");
        const alertBox = document.getElementById("google-auth-alert");

        if (!btn) return;

        const originalBtnHtml = btn.innerHTML;

        function showError(message) {
            if (alertBox) {
                alertBox.textContent = message;
                alertBox.classList.remove("d-none");
            } else {
                alert(message);
            }
        }

        function hideError() {
            if (alertBox) {
                alertBox.textContent = "";
                alertBox.classList.add("d-none");
            }
        }

        function setButtonLoading(isLoading) {
            if (isLoading) {
                btn.disabled = true;
                btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Authenticating...';
            } else {
                btn.disabled = false;
                btn.innerHTML = originalBtnHtml;
            }
        }

        // Check if Firebase configuration is provided
        const hasConfig = typeof FIREBASE_CONFIG !== "undefined" && FIREBASE_CONFIG && FIREBASE_CONFIG.apiKey;

        let authInitialized = false;
        if (hasConfig && typeof firebase !== "undefined") {
            try {
                if (!firebase.apps.length) {
                    firebase.initializeApp(FIREBASE_CONFIG);
                }
                authInitialized = true;
            } catch (err) {
                console.error("Firebase initialization failed:", err);
            }
        }

        btn.addEventListener("click", async function (e) {
            e.preventDefault();
            hideError();

            if (!hasConfig) {
                showError("Google Sign-In is not configured yet. Please configure FIREBASE_API_KEY and FIREBASE_AUTH_DOMAIN in the environment.");
                return;
            }

            if (!authInitialized || typeof firebase === "undefined" || !firebase.auth) {
                showError("Firebase Authentication library could not be loaded. Please check your network connection.");
                return;
            }

            setButtonLoading(true);

            try {
                const provider = new firebase.auth.GoogleAuthProvider();
                provider.setCustomParameters({ prompt: "select_account" });

                const result = await firebase.auth().signInWithPopup(provider);
                if (!result || !result.user) {
                    throw new Error("No user returned from Google sign-in.");
                }

                const idToken = await result.user.getIdToken();

                // Send ID token to HireAgent Flask backend endpoint
                const response = await fetch("/login/firebase", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    body: JSON.stringify({ id_token: idToken })
                });

                const data = await response.json();

                if (response.ok && data.ok) {
                    window.location.href = data.redirect || "/dashboard";
                } else {
                    showError(data.error || "Authentication backend could not verify your Google account.");
                    setButtonLoading(false);
                }
            } catch (err) {
                setButtonLoading(false);

                // Handle known Firebase popup errors gracefully
                if (err.code === "auth/popup-closed-by-user" || err.code === "auth/cancelled-popup-request") {
                    // User simply closed the popup, do not show error banner
                    return;
                } else if (err.code === "auth/popup-blocked") {
                    showError("Sign-in pop-up was blocked by your browser. Please allow pop-ups for this site.");
                } else if (err.code === "auth/unauthorized-domain") {
                    showError("Current domain is not authorized in Firebase Console (Authentication > Settings > Authorized Domains).");
                } else {
                    showError(err.message || "An error occurred during Google Sign-In. Please try again.");
                }
            }
        });
    });
})();
