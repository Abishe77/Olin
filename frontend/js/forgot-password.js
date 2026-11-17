
const API_URL = "http://localhost:8000";

const form = document.getElementById("forgot-password-form");

const emailInput = document.getElementById("email");

const otpGroup = document.getElementById("otp-group");
const otpInput = document.getElementById("otp");

const passwordFields = document.getElementById("password-fields");
const newPasswordInput = document.getElementById("new-password");
const confirmPasswordInput = document.getElementById("confirm-password");

const submitButton = document.getElementById("forgot-password-button");

const messageElement = document.getElementById("forgot-password-message");
const confirmError = document.getElementById("confirm-error");
const descriptionElement = document.getElementById("recovery-description");

let currentStage = "email";
let resetCompleted = false;

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

    const value = newPasswordInput.value;

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

    const password = newPasswordInput.value;
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


// General messages

function showMessage(message, type = "error") {

    messageElement.textContent = message;
    messageElement.className = `form-message visible ${type}`;

}


function clearMessage() {

    messageElement.textContent = "";
    messageElement.className = "form-message";

}


// Loading state

function setLoading(isLoading) {

    submitButton.disabled = isLoading;

    if (isLoading) {

        const loadingLabels = {
            email: "Sending reset code...",
            otp: "Verifying code...",
            password: "Resetting password..."
        };

        submitButton.textContent = loadingLabels[currentStage];
        return;

    }

    const buttonLabels = {
        email: "Send reset code",
        otp: "Submit OTP",
        password: "Reset password"
    };

    submitButton.textContent = buttonLabels[currentStage];

}


// Stage management

function moveToStage(stage) {

    currentStage = stage;

    otpGroup.hidden = stage === "email";
    passwordFields.hidden = stage !== "password";

    emailInput.readOnly = stage !== "email";

    otpInput.required = stage === "otp" || stage === "password";

    newPasswordInput.required = stage === "password";
    confirmPasswordInput.required = stage === "password";

    if (stage === "email") {

        descriptionElement.textContent =
            "Enter the email address associated with your account. We'll send you a reset code.";

    }

    if (stage === "otp") {

        descriptionElement.textContent =
            "Enter the six-digit reset code sent to your email address.";

        otpInput.focus();

    }

    if (stage === "password") {

        descriptionElement.textContent =
            "Your code has been verified. Choose a new password for your account.";

        validatePassword();
        newPasswordInput.focus();

    }

    setLoading(false);

}


// Stage 1: Request reset code

async function requestResetCode() {

    const email = emailInput.value.trim();

    const response = await fetch(`${API_URL}/auth/forgot-password`, {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            email: email
        })

    });

    const data = await readResponse(response);

    if (!response.ok) {

        throw new Error(
            data.detail || "Unable to process your request. Please try again."
        );

    }

    moveToStage("otp");

    showMessage(
        data.message ||
        "If the account exists, a password reset code has been sent."
    );

}


// Stage 2: Verify OTP

async function verifyResetCode() {

    const email = emailInput.value.trim();
    const otp = otpInput.value.trim();

    if (!/^[0-9]{6}$/.test(otp)) {

        throw new Error("Please enter a valid 6-digit reset code.");

    }

    const response = await fetch(`${API_URL}/auth/verify-reset-otp`, {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            email: email,
            otp: otp
        })

    });

    const data = await readResponse(response);

    if (!response.ok) {

        throw new Error(
            data.detail || "Invalid or expired reset code."
        );

    }

    moveToStage("password");

    showMessage("Your reset code has been verified.", "success");

}


// Read API response

async function readResponse(response) {

    try {
        return await response.json();
    } catch {
        return {};
    }

}


// Stage 3: Reset password

async function resetPassword() {

    const email = emailInput.value.trim();
    const otp = otpInput.value.trim();

    const newPassword = newPasswordInput.value;
    const confirmPassword = confirmPasswordInput.value;

    const passwordValid = validatePassword();
    const confirmationValid = validateConfirmation();

    if (!passwordValid) {

        throw new Error("Please satisfy all password requirements.");

    }

    if (!confirmationValid) {

        throw new Error("Please make sure both passwords match.");

    }

    const response = await fetch(`${API_URL}/auth/reset-password`, {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            email: email,
            otp: otp,
            new_password: newPassword
        })

    });

    const data = await readResponse(response);

    if (!response.ok) {

        throw new Error(
            data.detail || "Unable to reset your password. Please try again."
        );

    }

    resetCompleted = true;

    submitButton.disabled = true;
    submitButton.textContent = "Password reset";

    showMessage(
        "Password reset successfully. Redirecting to sign in...",
        "success"
    );

    setTimeout(() => {

        window.location.href = "./login.html?password_reset=1";

    }, 1500);

}


// Password input events

newPasswordInput.addEventListener("input", () => {

    validatePassword();
    validateConfirmation();

    clearMessage();

});


confirmPasswordInput.addEventListener("input", () => {

    validateConfirmation();
    clearMessage();

});


// OTP input restriction

otpInput.addEventListener("input", () => {

    otpInput.value = otpInput.value.replace(/\D/g, "").slice(0, 6);

    clearMessage();

});


// Email input

emailInput.addEventListener("input", clearMessage);


// Form submission

form.addEventListener("submit", async (event) => {

    event.preventDefault();

    clearMessage();

    setLoading(true);

    try {

        if (currentStage === "email") {

            await requestResetCode();

        } else if (currentStage === "otp") {

            await verifyResetCode();

        } else if (currentStage === "password") {

            await resetPassword();

        }

    } catch (error) {

        showMessage(error.message);

    } finally {

        if (!resetCompleted) {
            setLoading(false);
        }

    }

});
