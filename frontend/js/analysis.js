
const API_URL = "http://localhost:8000";

let accessToken = null;
let currentAnalysis = null;

const PARAMETER_LABELS = {
    theta_E: "Einstein radius (θE)",
    e1: "Ellipticity e1",
    e2: "Ellipticity e2",
    gamma1: "External shear γ1",
    gamma2: "External shear γ2"
};

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[character]);
}

function formatValue(value) {
    if (typeof value === "number" && Number.isFinite(value)) {
        return value.toFixed(6);
    }

    return value === null || value === undefined
        ? "Not available"
        : String(value);
}

function formatDate(value) {
    if (!value) return "Not available";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) return "Not available";

    return date.toLocaleString();
}

function getInitial(value) {
    return (value || "O").trim().charAt(0).toUpperCase();
}

async function refreshAccessToken() {
    const response = await fetch(`${API_URL}/auth/refresh`, {
        method: "POST",
        credentials: "include"
    });

    if (!response.ok) {
        throw new Error("Your session has expired. Please log in again.");
    }

    const data = await response.json();
    accessToken = data.access_token;

    if (!accessToken) {
        throw new Error("Unable to refresh your session.");
    }
}

async function authenticatedFetch(url, options = {}, retry = true) {
    const response = await fetch(url, {
        ...options,
        credentials: "include",
        headers: {
            ...(options.headers || {}),
            Authorization: `Bearer ${accessToken}`
        }
    });

    if (response.status === 401 && retry) {
        await refreshAccessToken();
        return authenticatedFetch(url, options, false);
    }

    return response;
}

async function getCurrentUser() {
    const response = await authenticatedFetch(`${API_URL}/auth/me`);

    if (!response.ok) {
        throw new Error("Unable to retrieve account details.");
    }

    const data = await response.json();
    return data.user;
}

function updateAccountUI(user) {
    const displayName = user.name || user.username || "Account";
    const initial = getInitial(displayName);

    document.getElementById("sidebar-avatar").textContent = initial;
    document.getElementById("topbar-avatar").textContent = initial;
    document.getElementById("drawer-avatar").textContent = initial;

    document.getElementById("sidebar-name").textContent = displayName;
    document.getElementById("drawer-name").textContent = displayName;
    document.getElementById("drawer-email").textContent = user.email || "";
}

function setupAccountDrawer() {
    const drawer = document.getElementById("account-drawer");
    const backdrop = document.getElementById("drawer-backdrop");

    function openDrawer() {
        drawer.classList.add("open");
        backdrop.classList.add("open");
    }

    function closeDrawer() {
        drawer.classList.remove("open");
        backdrop.classList.remove("open");
    }

    document.getElementById("account-trigger").addEventListener("click", openDrawer);
    document.getElementById("topbar-avatar").addEventListener("click", openDrawer);
    document.getElementById("drawer-close").addEventListener("click", closeDrawer);
    backdrop.addEventListener("click", closeDrawer);

    document.addEventListener("keydown", event => {
        if (event.key === "Escape") closeDrawer();
    });
}

function setupMobileSidebar() {
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebar-backdrop");

    document.getElementById("mobile-menu-button").addEventListener("click", () => {
        sidebar.classList.add("open");
        backdrop.classList.add("open");
    });

    backdrop.addEventListener("click", () => {
        sidebar.classList.remove("open");
        backdrop.classList.remove("open");
    });
}

async function logout() {
    const button = document.getElementById("logout-button");
    button.disabled = true;

    try {
        await authenticatedFetch(`${API_URL}/auth/logout`, {
            method: "POST"
        });
    } catch (error) {
        console.error("Logout error:", error);
    }

    accessToken = null;
    window.location.href = "./login.html";
}

function getAnalysisId() {
    const params = new URLSearchParams(window.location.search);
    return params.get("id");
}

function showPageError(message) {
    const loading = document.getElementById("page-loading");
    const error = document.getElementById("page-error");

    loading.hidden = true;
    error.hidden = false;
    error.textContent = message;
}

function renderParameterRows(predictions, uncertainty) {
    const container = document.getElementById("parameter-rows");

    if (!predictions || typeof predictions !== "object") {
        container.innerHTML = `
            <div class="loading-state">
                Prediction values are not available for this analysis.
            </div>
        `;
        return;
    }

    const rows = Object.keys(PARAMETER_LABELS).map(key => {
        const prediction = predictions[key];

        const uncertaintyValue = uncertainty?.[key];

        let displayedUncertainty = "Not available";

        if (typeof uncertaintyValue === "number") {
            displayedUncertainty = formatValue(uncertaintyValue);
        } else if (uncertaintyValue && typeof uncertaintyValue === "object") {
            const calibrated =
                uncertaintyValue.calibrated ??
                uncertaintyValue.calibrated_uncertainty ??
                uncertaintyValue.calibrated_std;

            displayedUncertainty = calibrated !== undefined
                ? formatValue(calibrated)
                : "Not available";
        }

        return `
            <div class="parameter-row">
                <span class="parameter-name">${escapeHTML(PARAMETER_LABELS[key])}</span>
                <span class="parameter-value">${escapeHTML(formatValue(prediction))}</span>
                <span class="parameter-uncertainty">${escapeHTML(displayedUncertainty)}</span>
            </div>
        `;
    });

    container.innerHTML = rows.join("");
}

function updateOutputAvailability(outputs) {
    const reconstructionAvailable = Boolean(outputs?.reconstruction_available);
    const residualAvailable = Boolean(outputs?.residual_available);

    document.getElementById("reconstruction-status").textContent =
        reconstructionAvailable ? "Generated" : "Not available";

    document.getElementById("residual-status").textContent =
        residualAvailable ? "Generated" : "Not available";

    // The current detail endpoint only confirms file availability.
    // It does not provide authenticated image URLs.
}

function renderAnalysis(data) {
    currentAnalysis = data;

    document.getElementById("analysis-id").textContent = data.analysis_id ?? "—";

    const status = document.getElementById("analysis-status");
    status.textContent = data.status || "Unknown";
    status.className = `status ${(data.status || "").toLowerCase()}`;

    document.getElementById("analysis-date").textContent =
        formatDate(data.created_at);

    document.getElementById("model-version").textContent =
        data.model_version || "Not available";

    renderParameterRows(data.predictions, data.uncertainty);
    updateOutputAvailability(data.outputs);

    document.getElementById("page-loading").hidden = true;
    document.getElementById("analysis-results").hidden = false;
}

async function loadAnalysis(analysisId) {
    const response = await authenticatedFetch(
        `${API_URL}/analyses/${encodeURIComponent(analysisId)}`
    );

    if (response.status === 404) {
        throw new Error("This analysis could not be found, or you do not have access to it.");
    }

    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "Unable to load analysis results.");
    }

    const data = await response.json();
    renderAnalysis(data);
}

function showReportMessage(message) {
    const element = document.getElementById("report-message");
    element.hidden = false;
    element.textContent = message;
}

async function downloadReport() {
    if (!currentAnalysis) return;

    const button = document.getElementById("report-button");
    const label = document.getElementById("report-button-label");

    button.disabled = true;
    label.textContent = "Preparing report...";

    try {
        // Request PDF generation. The backend reuses an existing report
        // when one is already available.
        const generateResponse = await authenticatedFetch(
            `${API_URL}/analyses/${encodeURIComponent(currentAnalysis.analysis_id)}/report`,
            { method: "POST" }
        );

        if (!generateResponse.ok) {
            const data = await generateResponse.json().catch(() => ({}));
            throw new Error(data.detail || "Unable to generate the PDF report.");
        }

        const reportData = await generateResponse.json();

        const downloadResponse = await authenticatedFetch(
            `${API_URL}/reports/${encodeURIComponent(reportData.report_id)}/download`
        );

        if (!downloadResponse.ok) {
            throw new Error("The report was generated, but its download failed.");
        }

        const blob = await downloadResponse.blob();
        const objectURL = URL.createObjectURL(blob);

        const disposition = downloadResponse.headers.get("Content-Disposition");
        const filenameMatch = disposition?.match(/filename="?([^"]+)"?/i);

        const filename = filenameMatch
            ? filenameMatch[1]
            : `olin-analysis-${currentAnalysis.analysis_id}.pdf`;

        const link = document.createElement("a");
        link.href = objectURL;
        link.download = filename;

        document.body.appendChild(link);
        link.click();
        link.remove();

        setTimeout(() => URL.revokeObjectURL(objectURL), 1000);

        showReportMessage("Your PDF report has been downloaded.");
    } catch (error) {
        showReportMessage(error.message);
    } finally {
        button.disabled = false;
        label.textContent = "Generate PDF report";
    }
}

async function initializeAnalysisPage() {
    setupAccountDrawer();
    setupMobileSidebar();

    document.getElementById("logout-button").addEventListener("click", logout);
    document.getElementById("report-button").addEventListener("click", downloadReport);

    try {
        await refreshAccessToken();

        const user = await getCurrentUser();
        updateAccountUI(user);

        const analysisId = getAnalysisId();

        if (!analysisId) {
            throw new Error("No analysis ID was provided in the page URL.");
        }

        await loadAnalysis(analysisId);

    } catch (error) {
        showPageError(error.message);

        if (error.message.toLowerCase().includes("session")) {
            setTimeout(() => {
                window.location.href = "./login.html";
            }, 1500);
        }
    }
}

document.addEventListener("DOMContentLoaded", initializeAnalysisPage);
