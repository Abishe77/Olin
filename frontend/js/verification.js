
const API_URL = "http://localhost:8000";

const verificationForm = document.getElementById("verification-form");

const emailInput = document.getElementById("email");
const otpInput = document.getElementById("otp");

const verifyButton = document.getElementById("verify-button");
const resendButton = document.getElementById("resend-button");

const verificationMessage = document.getElementById(
    "verification-message"
);

let resendTimer = null;
let resendSeconds = 0;


// Load the email saved during registration.

const savedEmail = sessionStorage.getItem("olin_verification_email");

if (savedEmail) {
    emailInput.value = savedEmail;
}


// Display messages.

function showMessage(message, type = "error") {

    verificationMessage.textContent = message;
    verificationMessage.className = `form-message visible ${type}`;

}


function clearMessage() {

    verificationMessage.textContent = "";
    verificationMessage.className = "form-message";

}


// Safely read API responses.

async function readResponse(response) {

    try {
        return await response.json();
    } catch {
        return {};
    }

}


// Resend cooldown.

function startResendCooldown(seconds = 60) {

    clearInterval(resendTimer);

    resendSeconds = seconds;
    resendButton.disabled = true;

    resendButton.textContent =
        `Resend code in ${resendSeconds}s`;

    resendTimer = setInterval(() => {

        resendSeconds--;

        if (resendSeconds <= 0) {

            clearInterval(resendTimer);
            resendTimer = null;

            resendButton.disabled = false;
            resendButton.textContent = "Resend verification code";

            return;

        }

        resendButton.textContent =
            `Resend code in ${resendSeconds}s`;

    }, 1000);

}


// Verify email.

verificationForm.addEventListener("submit", async (event) => {

    event.preventDefault();
    clearMessage();

    const email = emailInput.value.trim().toLowerCase();
    const otp = otpInput.value.trim();

    if (!email) {
        showMessage("Please enter your email address.");
        return;
    }

    if (!/^[0-9]{6}$/.test(otp)) {
        showMessage("Please enter a valid 6-digit verification code.");
        return;
    }

    verifyButton.disabled = true;
    verifyButton.textContent = "Verifying...";

    try {

        const response = await fetch(
            `${API_URL}/auth/verify-email`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    email: email,
                    otp: otp
                })
            }
        );

        const data = await readResponse(response);

        if (!response.ok) {

            const message =
                typeof data.detail === "string"
                    ? data.detail
                    : "Email verification failed. Please try again.";

            showMessage(message);
            return;

        }

        showMessage(
            data.message || "Email verified successfully.",
            "success"
        );

        // Preserve the verified email for the login page.
        sessionStorage.setItem("olin_verified_email", email);
        sessionStorage.removeItem("olin_verification_email");

        // Verification does not create an authenticated session.
        setTimeout(() => {
            window.location.href = "./login.html?verified=1";
        }, 1200);

    } catch (error) {

        showMessage(
            "Unable to connect to Olin. Please try again."
        );

    } finally {

        verifyButton.disabled = false;
        verifyButton.textContent = "Verify email";

    }

});


// Resend verification code.

resendButton.addEventListener("click", async () => {

    clearMessage();

    const email = emailInput.value.trim().toLowerCase();

    if (!email) {
        showMessage("Please enter your email address first.");
        emailInput.focus();
        return;
    }

    resendButton.disabled = true;
    resendButton.textContent = "Sending...";

    try {

        const response = await fetch(
            `${API_URL}/auth/resend-verification`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    email: email
                })
            }
        );

        const data = await readResponse(response);

        if (!response.ok) {

            const message =
                typeof data.detail === "string"
                    ? data.detail
                    : "Unable to resend the verification code.";

            showMessage(message);

            // The backend allows one resend per minute.
            if (response.status === 429) {
                startResendCooldown(60);
            } else {
                resendButton.disabled = false;
                resendButton.textContent =
                    "Resend verification code";
            }

            return;

        }

        showMessage(
            data.message ||
            "If the account requires verification, a new code has been sent.",
            "success"
        );

        sessionStorage.setItem("olin_verification_email", email);

        startResendCooldown(60);

    } catch (error) {

        showMessage(
            "Unable to connect to Olin. Please try again."
        );

        resendButton.disabled = false;
        resendButton.textContent = "Resend verification code";

    }

});


// Clear messages when the user edits the fields.

emailInput.addEventListener("input", clearMessage);
otpInput.addEventListener("input", clearMessage);
