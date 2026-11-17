
const API_URL = "http://localhost:8000";

let accessToken = null;

const accountTrigger = document.getElementById("account-trigger");
const topbarAvatar = document.getElementById("topbar-avatar");

const accountDrawer = document.getElementById("account-drawer");
const drawerBackdrop = document.getElementById("drawer-backdrop");
const drawerClose = document.getElementById("drawer-close");

const logoutButton = document.getElementById("logout-button");

const sidebar = document.getElementById("sidebar");
const mobileMenuButton = document.getElementById("mobile-menu-button");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");

const recentAnalyses = document.getElementById("recent-analyses");
const recentReports = document.getElementById("recent-reports");

const newAnalysisSection = document.getElementById("new-analysis-section");




function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => {
        const entities = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        };

        return entities[character];
    });
}


function formatDate(value) {
    if (!value) {
        return "—";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return "—";
    }

    return new Intl.DateTimeFormat("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric"
    }).format(date);
}


function getInitial(name) {
    const normalizedName = String(name ?? "").trim();

    return normalizedName
        ? normalizedName.charAt(0).toUpperCase()
        : "?";
}


function showAuthenticationError() {
    window.location.replace("./login.html");
}




async function refreshAccessToken() {
    const response = await fetch(`${API_URL}/auth/refresh`, {
        method: "POST",
        credentials: "include"
    });

    if (!response.ok) {
        throw new Error("Session expired.");
    }

    const data = await response.json();

    if (!data.access_token) {
        throw new Error("Access token unavailable.");
    }

    accessToken = data.access_token;

    return accessToken;
}


async function authenticatedFetch(url, options = {}) {
    if (!accessToken) {
        throw new Error("Authentication required.");
    }

    const headers = new Headers(options.headers || {});

    headers.set("Authorization", `Bearer ${accessToken}`);

    let response = await fetch(url, {
        ...options,
        headers,
        credentials: "include"
    });

    // Attempt one token refresh if the access token expires.
    if (response.status === 401) {
        await refreshAccessToken();

        headers.set("Authorization", `Bearer ${accessToken}`);

        response = await fetch(url, {
            ...options,
            headers,
            credentials: "include"
        });
    }

    if (response.status === 401) {
        throw new Error("Session expired.");
    }

    return response;
}


async function getCurrentUser() {
    const response = await authenticatedFetch(
        `${API_URL}/auth/me`
    );

    if (!response.ok) {
        throw new Error("Unable to retrieve account information.");
    }

    const data = await response.json();

    return data.user;
}




function updateAccountUI(user) {
    const name = user.name || "Olin User";
    const email = user.email || "";

    const initial = getInitial(name);

    document.getElementById("sidebar-avatar").textContent = initial;
    document.getElementById("topbar-avatar").textContent = initial;
    document.getElementById("drawer-avatar").textContent = initial;

    document.getElementById("sidebar-name").textContent = name;

    document.getElementById("drawer-name").textContent = name;
    document.getElementById("drawer-email").textContent = email;

    document.getElementById("welcome-description").textContent =
        `Welcome back, ${name}. Your gravitational lensing research workspace.`;
}



function getAnalysisStatus(status) {
    const normalizedStatus = String(status || "").toLowerCase();

    const statusLabels = {
        completed: "Completed",
        processing: "Processing",
        pending: "Pending",
        running: "Processing",
        failed: "Failed"
    };

    const label = statusLabels[normalizedStatus] || "Unknown";

    const className = [
        "completed",
        "processing",
        "pending",
        "running",
        "failed"
    ].includes(normalizedStatus)
        ? normalizedStatus === "running"
            ? "processing"
            : normalizedStatus
        : "";

    return {
        label,
        className
    };
}


function renderAnalyses(analyses) {

    // Show the welcome card only when the user has no analyses.
    if (!Array.isArray(analyses) || analyses.length === 0) {

        newAnalysisSection.hidden = false;

        recentAnalyses.innerHTML = `
            <div class="empty-state">
                <strong>No analyses yet.</strong>
                Your research history will appear here after your first analysis.
                <br>
                <a href="./new-analysis.html">Start your first analysis →</a>
            </div>
        `;

        document.getElementById("analysis-count").textContent =
            "No analysis records found.";

        return;
    }

    // Hide the welcome card for returning users.
    newAnalysisSection.hidden = true;

    recentAnalyses.innerHTML = analyses.map(analysis => {
        const analysisId = escapeHTML(analysis.analysis_id);
        const modelVersion = escapeHTML(analysis.model_version);
        const date = escapeHTML(formatDate(analysis.created_at));

        const status = getAnalysisStatus(analysis.status);

        return `
            <div class="data-row analysis-row">

                <span class="data-primary">
                    <a
                        class="analysis-id-link"
                        href="./analysis.html?id=${encodeURIComponent(analysisId)}"
                    >
                        #${analysisId}
                    </a>
                </span>

                <span class="data-secondary">
                    ${modelVersion}
                </span>

                <span class="status ${status.className}">
                    ${status.label}
                </span>

                <span class="data-secondary">
                    ${date}
                </span>

            </div>
        `;
    }).join("");

    document.getElementById("analysis-count").textContent =
        `Showing ${analyses.length} recent analysis record${analyses.length === 1 ? "" : "s"}.`;
}


async function loadAnalyses() {
    try {
        const response = await authenticatedFetch(
            `${API_URL}/analyses?limit=5&offset=0`
        );

        if (!response.ok) {
            throw new Error("Unable to load analyses.");
        }

        const data = await response.json();

        renderAnalyses(data.analysis || []);

    } catch (error) {
        if (error.message === "Session expired.") {
            throw error;
        }

        // Do not incorrectly show the first-time welcome card
        // when the analysis history could not be retrieved.
        newAnalysisSection.hidden = true;

        recentAnalyses.innerHTML = `
            <div class="error-state">
                Unable to load your analyses. Please try refreshing the page.
            </div>
        `;
    }
}



function renderReports(reports) {
    if (!Array.isArray(reports) || reports.length === 0) {
        recentReports.innerHTML = `
            <div class="empty-state">
                <strong>No reports generated yet.</strong>
                Completed analyses will have downloadable scientific reports.
            </div>
        `;

        return;
    }

    recentReports.innerHTML = reports.map(report => {
        const reportId = escapeHTML(report.report_id);
        const analysisId = escapeHTML(report.analysis_id);
        const fileName = escapeHTML(report.file_name || "report.pdf");
        const date = escapeHTML(formatDate(report.created_at));

        return `
            <div class="data-row report-row">

                <span class="data-primary">
                    ${fileName}
                </span>

                <span class="data-secondary">
                    #${analysisId}
                </span>

                <span class="data-secondary">
                    ${date}
                </span>

                <button
                    class="download-button"
                    type="button"
                    data-report-id="${reportId}"
                >
                    Download
                </button>

            </div>
        `;
    }).join("");
}


async function loadReports() {
    try {
        const response = await authenticatedFetch(
            `${API_URL}/reports?limit=5&offset=0`
        );

        if (!response.ok) {
            throw new Error("Unable to load reports.");
        }

        const data = await response.json();

        renderReports(data.reports || []);

    } catch (error) {
        if (error.message === "Session expired.") {
            throw error;
        }

        recentReports.innerHTML = `
            <div class="error-state">
                Unable to load your reports. Please try refreshing the page.
            </div>
        `;
    }
}


async function downloadReport(reportId, button) {
    const originalText = button.textContent;

    button.disabled = true;
    button.textContent = "Preparing...";

    try {
        const response = await authenticatedFetch(
            `${API_URL}/reports/${encodeURIComponent(reportId)}/download`
        );

        if (!response.ok) {
            throw new Error("Report download failed.");
        }

        const blob = await response.blob();

        const downloadURL = URL.createObjectURL(blob);

        const link = document.createElement("a");

        link.href = downloadURL;
        link.download = "report.pdf";

        document.body.appendChild(link);
        link.click();
        link.remove();

        URL.revokeObjectURL(downloadURL);

    } catch (error) {
        if (error.message === "Session expired.") {
            showAuthenticationError();
            return;
        }

        alert(error.message || "Unable to download the report.");

    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}




function openAccountDrawer() {
    accountDrawer.classList.add("open");
    drawerBackdrop.classList.add("open");

    accountDrawer.setAttribute("aria-hidden", "false");

    document.body.classList.add("drawer-open");

    drawerClose.focus();
}


function closeAccountDrawer() {
    accountDrawer.classList.remove("open");
    drawerBackdrop.classList.remove("open");

    accountDrawer.setAttribute("aria-hidden", "true");

    document.body.classList.remove("drawer-open");
}


accountTrigger.addEventListener("click", openAccountDrawer);

topbarAvatar.addEventListener("click", openAccountDrawer);

drawerClose.addEventListener("click", closeAccountDrawer);

drawerBackdrop.addEventListener("click", closeAccountDrawer);




function openSidebar() {
    sidebar.classList.add("open");
    sidebarBackdrop.classList.add("open");
}


function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarBackdrop.classList.remove("open");
}


mobileMenuButton.addEventListener("click", () => {
    if (sidebar.classList.contains("open")) {
        closeSidebar();
    } else {
        openSidebar();
    }
});


sidebarBackdrop.addEventListener("click", closeSidebar);


// Close the mobile sidebar after navigation.
document.querySelectorAll(".sidebar-navigation .nav-link").forEach(link => {
    link.addEventListener("click", closeSidebar);
});


async function logout() {
    logoutButton.disabled = true;

    logoutButton.querySelector("strong").textContent = "Signing out...";

    try {
        await fetch(`${API_URL}/auth/logout`, {
            method: "POST",
            credentials: "include"
        });

    } catch (error) {
        // Redirect even if the server cannot be reached.
        // The next login attempt will require a valid session.
    }

    accessToken = null;

    window.location.replace("./login.html");
}


logoutButton.addEventListener("click", logout);



document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
        closeAccountDrawer();
        closeSidebar();
    }
});





recentReports.addEventListener("click", event => {
    const button = event.target.closest("[data-report-id]");

    if (!button) {
        return;
    }

    const reportId = button.dataset.reportId;

    downloadReport(reportId, button);
});



async function initializeDashboard() {
    try {
        // The welcome card stays hidden until history is successfully loaded.
        newAnalysisSection.hidden = true;

        await refreshAccessToken();

        const user = await getCurrentUser();

        updateAccountUI(user);

        await Promise.all([
            loadAnalyses(),
            loadReports()
        ]);

    } catch (error) {
        console.error("Dashboard initialization failed:", error);

        showAuthenticationError();
    }
}


initializeDashboard();
