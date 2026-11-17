
const API_URL = "http://localhost:8000";

const registerForm = document.getElementById("register-form");

const nameInput = document.getElementById("name");
const emailInput = document.getElementById("email");
const passwordInput = document.getElementById("password");
const confirmPasswordInput = document.getElementById("confirm-password");

const registerButton = document.getElementById("register-button");
const googleButton = document.getElementById("google-signup-button");

const registerMessage = document.getElementById("register-message");
const confirmError = document.getElementById("confirm-error");

const passwordRules = {
    min: false,
    max: false,
    upper: false,
    number: false,
    special: false
};


// Password visibility

document.querySelectorAll("[data-password-toggle]").forEach(button => {

    button.addEventListener("click", () => {

        const inputId = button.dataset.passwordToggle;
        const input = document.getElementById(inputId);

        const isHidden = input.type === "password";

        input.type = isHidden ? "text" : "password";

        button.setAttribute(
            "aria-label",
            isHidden ? "Hide password" : "Show password"
        );

    });

});


// Password requirements

function validatePassword() {

    const value = passwordInput.value;

    passwordRules.min = value.length >= 8;
    passwordRules.max = value.length > 0 && value.length <= 128;
    passwordRules.upper = /[A-Z]/.test(value);
    passwordRules.number = /[0-9]/.test(value);
    passwordRules.special = /[^a-zA-Z0-9]/.test(value);

    document.querySelectorAll("[data-rule]").forEach(item => {

        const rule = item.dataset.rule;
        const passed = passwordRules[rule];

        const icon = item.querySelector(".requirement-icon");

        item.classList.toggle("valid", passed);
        icon.textContent = passed ? "✓" : "○";

    });

    return Object.values(passwordRules).every(Boolean);

}


// Confirm password

function validateConfirmation() {

    const password = passwordInput.value;
    const confirmation = confirmPasswordInput.value;

    if (!confirmation) {

        confirmError.textContent = "";
        confirmError.className = "form-message";

        return false;

    }

    if (password !== confirmation) {

        confirmError.textContent = "Passwords do not match.";
        confirmError.className = "form-message visible";

        return false;

    }

    confirmError.textContent = "";
    confirmError.className = "form-message";

    return true;

}


// Display registration messages

function showMessage(message, type = "error") {

    registerMessage.textContent = message;
    registerMessage.className = `form-message visible ${type}`;

}


function clearMessage() {

    registerMessage.textContent = "";
    registerMessage.className = "form-message";

}


// Password input events

passwordInput.addEventListener("input", () => {

    validatePassword();
    validateConfirmation();

    clearMessage();

});


confirmPasswordInput.addEventListener("input", () => {

    validateConfirmation();
    clearMessage();

});


emailInput.addEventListener("input", clearMessage);
nameInput.addEventListener("input", clearMessage);


// Email and password registration

registerForm.addEventListener("submit", async (event) => {

    event.preventDefault();

    clearMessage();

    const passwordValid = validatePassword();
    const confirmationValid = validateConfirmation();

    if (!passwordValid) {

        showMessage("Please satisfy all password requirements.");
        return;

    }

    if (!confirmationValid) {

        showMessage("Please make sure both passwords match.");
        return;

    }

    const registrationData = {
        name: nameInput.value.trim(),
        email: emailInput.value.trim(),
        password: passwordInput.value
    };

    if (!registrationData.name || !registrationData.email) {

        showMessage("Please enter your name and email address.");
        return;

    }

    registerButton.disabled = true;
    registerButton.textContent = "Creating account...";

    try {

        const response = await fetch(
            `${API_URL}/auth/register`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(registrationData)
            }
        );

        const data = await response.json();

        if (!response.ok) {

            let errorMessage = "Registration failed.";

            if (Array.isArray(data.detail)) {

                errorMessage = data.detail
                    .map(error => error.msg)
                    .join(" ");

            } else if (typeof data.detail === "string") {

                errorMessage = data.detail;

            }

            showMessage(errorMessage);
            return;

        }

        // Keep the email available for the verification page.

        sessionStorage.setItem(
            "olin_verification_email",
            registrationData.email
        );

        // Continue to email verification.

        window.location.href = "./verification.html";

    } catch (error) {

        showMessage(
            "Unable to connect to Olin. Please try again."
        );

    } finally {

        registerButton.disabled = false;
        registerButton.textContent = "Create account";

    }

});


// Google registration

googleButton.addEventListener("click", () => {

    window.location.href = `${API_URL}/auth/google/login`;

});


// Initial password requirement state

validatePassword();
