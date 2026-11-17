
const API_URL = "http://localhost:8000";


// Elements

const loginForm = document.getElementById("login-form");

const emailInput = document.getElementById("email");
const passwordInput = document.getElementById("password");

const loginButton = document.getElementById("login-submit");
const googleButton = document.getElementById("google-login");
const togglePassword = document.getElementById("toggle-password");

const loginError = document.getElementById("login-error");


// Display error messages

function showError(message) {

    loginError.textContent = message;
    loginError.classList.add("visible");

}


// Clear error messages

function clearError() {

    loginError.textContent = "";
    loginError.classList.remove("visible");

}


// Password visibility

togglePassword.addEventListener("click", () => {

    const isHidden = passwordInput.type === "password";

    passwordInput.type = isHidden ? "text" : "password";

    togglePassword.setAttribute(
        "aria-label",
        isHidden ? "Hide password" : "Show password"
    );

});


// Clear errors when the user edits the form

emailInput.addEventListener("input", clearError);
passwordInput.addEventListener("input", clearError);


// Email and password login

loginForm.addEventListener("submit", async (event) => {

    event.preventDefault();

    clearError();

    const email = emailInput.value.trim();
    const password = passwordInput.value;

    if (!email || !password) {
        showError("Please enter your email address and password.");
        return;
    }

    loginButton.disabled = true;
    loginButton.textContent = "Signing in...";

    try {

        const response = await fetch(
            `${API_URL}/auth/login`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                credentials: "include",

                body: JSON.stringify({
                    email: email,
                    password: password
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {

            let errorMessage = "Unable to sign in.";

            if (typeof data.detail === "string") {

                errorMessage = data.detail;

            } else if (Array.isArray(data.detail)) {

                errorMessage = data.detail
                    .map(error => error.msg)
                    .join(" ");

            }

            showError(errorMessage);
            return;

        }

        // The backend has issued the refresh cookie.
        // The dashboard will obtain an access token through /auth/refresh.

        window.location.href = "./dashboard.html";

    } catch (error) {

        showError(
            "Unable to connect to Olin. Please try again."
        );

    } finally {

        loginButton.disabled = false;
        loginButton.textContent = "Sign in";

    }

});


// Google login

googleButton.addEventListener("click", () => {

    window.location.href = `${API_URL}/auth/google/login`;

});
