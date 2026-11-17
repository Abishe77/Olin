
const API_URL = "http://localhost:8000";

let accessToken = null;

const PAGE_SIZE = 20;
let currentPage = 0;
let totalCount = 0;

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[character]);
}

function formatDate(value) {
    if (!value) return "Not available";

    const date = new Date(value);

    return Number.isNaN(date.getTime())
        ? "Not available"
        : date.toLocaleString();
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

async function loadCurrentUser() {
    const response = await authenticatedFetch(`${API_URL}/auth/me`);

    if (!response.ok) {
        throw new Error("Unable to retrieve account details.");
    }

    const data = await response.json();
    const user = data.user;

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

function renderReports(reports) {
    const container = document.getElementById("reports-table-body");

    container.innerHTML = reports.map(report => {
        return `
            <div class="reports-row">

                <div class="reports-primary">
                    #${escapeHTML(report.report_id)}
                </div>

                <div class="reports-secondary">
                    Analysis #${escapeHTML(report.analysis_id)}
                </div>

                <div class="reports-file">
                    ${escapeHTML(report.file_name || "report.pdf")}
                </div>

                <div class="reports-date">
                    ${escapeHTML(formatDate(report.created_at))}
                </div>

                <button
                    class="reports-download"
                    type="button"
                    data-report-id="${escapeHTML(report.report_id)}">
                    Download PDF ↓
                </button>

            </div>
        `;
    }).join("");

    container.querySelectorAll(".reports-download").forEach(button => {
        button.addEventListener("click", () => {
            downloadReport(button.dataset.reportId, button);
        });
    });
}

function updatePagination() {
    const pagination = document.getElementById("reports-pagination");
    const totalPages = Math.max(1, Math.ceil(totalCount / PAGE_SIZE));

    pagination.hidden = totalCount <= PAGE_SIZE;

    document.getElementById("page-indicator").textContent =
        `Page ${currentPage + 1} of ${totalPages}`;

    document.getElementById("previous-page").disabled =
        currentPage === 0;

    document.getElementById("next-page").disabled =
        currentPage >= totalPages - 1;
}

async function loadReports() {
    const loading = document.getElementById("reports-loading");
    const errorElement = document.getElementById("reports-error");
    const emptyElement = document.getElementById("reports-empty");
    const tableContainer = document.getElementById("reports-table-container");

    loading.hidden = false;
    errorElement.hidden = true;
    emptyElement.hidden = true;
    tableContainer.hidden = true;

    document.getElementById("reports-pagination").hidden = true;

    try {
        const offset = currentPage * PAGE_SIZE;

        const response = await authenticatedFetch(
            `${API_URL}/reports?limit=${PAGE_SIZE}&offset=${offset}`
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || "Unable to load reports.");
        }

        const data = await response.json();

        const reports = data.reports || [];

        totalCount = data.count ?? reports.length;

        document.getElementById("reports-count").textContent =
            `${totalCount} REPORT${totalCount === 1 ? "" : "S"}`;

        loading.hidden = true;

        if (reports.length === 0) {
            emptyElement.hidden = false;
            return;
        }

        renderReports(reports);
        tableContainer.hidden = false;

        updatePagination();

    } catch (error) {
        loading.hidden = true;
        errorElement.hidden = false;
        errorElement.textContent = error.message;
    }
}

async function downloadReport(reportId, button) {
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = "Downloading...";

    try {
        const response = await authenticatedFetch(
            `${API_URL}/reports/${encodeURIComponent(reportId)}/download`
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || "Unable to download this report.");
        }

        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);

        const disposition = response.headers.get("Content-Disposition") || "";
        const filenameMatch = disposition.match(/filename="?([^"]+)"?/i);
        const filename = filenameMatch
            ? filenameMatch[1]
            : `olin-report-${reportId}.pdf`;

        const link = document.createElement("a");
        link.href = objectUrl;
        link.download = filename;

        document.body.appendChild(link);
        link.click();
        link.remove();

        URL.revokeObjectURL(objectUrl);

    } catch (error) {
        alert(error.message);
    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}

async function initializeReportsPage() {
    setupAccountDrawer();
    setupMobileSidebar();

    document.getElementById("logout-button").addEventListener("click", logout);

    document.getElementById("previous-page").addEventListener("click", async () => {
        if (currentPage > 0) {
            currentPage--;
            await loadReports();
        }
    });

    document.getElementById("next-page").addEventListener("click", async () => {
        const totalPages = Math.ceil(totalCount / PAGE_SIZE);

        if (currentPage < totalPages - 1) {
            currentPage++;
            await loadReports();
        }
    });

    try {
        await refreshAccessToken();
        await loadCurrentUser();
        await loadReports();

    } catch (error) {
        document.getElementById("reports-loading").hidden = true;

        const errorElement = document.getElementById("reports-error");
        errorElement.hidden = false;
        errorElement.textContent = error.message;

        if (error.message.toLowerCase().includes("session")) {
            setTimeout(() => {
                window.location.href = "./login.html";
            }, 1500);
        }
    }
}

document.addEventListener("DOMContentLoaded", initializeReportsPage);
