
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

function renderAnalyses(analyses) {
    const container = document.getElementById("history-table-body");

    container.innerHTML = analyses.map(analysis => {
        const id = analysis.analysis_id;
        const status = analysis.status || "Unknown";

        return `
            <div class="history-row">

                <div class="history-primary">
                    #${escapeHTML(id)}
                </div>

                <div class="history-secondary">
                    ${escapeHTML(analysis.model_version || "—")}
                </div>

                <div>
                    <span class="status ${escapeHTML(status.toLowerCase())}">
                        ${escapeHTML(status)}
                    </span>
                </div>

                <div class="history-date">
                    ${escapeHTML(formatDate(analysis.created_at))}
                </div>

                <a class="history-view-link"
                   href="./analysis.html?id=${encodeURIComponent(id)}">
                    View results →
                </a>

            </div>
        `;
    }).join("");
}

function updatePagination() {
    const pagination = document.getElementById("history-pagination");

    const totalPages = Math.max(1, Math.ceil(totalCount / PAGE_SIZE));

    pagination.hidden = totalCount <= PAGE_SIZE;

    document.getElementById("page-indicator").textContent =
        `Page ${currentPage + 1} of ${totalPages}`;

    document.getElementById("previous-page").disabled =
        currentPage === 0;

    document.getElementById("next-page").disabled =
        currentPage >= totalPages - 1;
}

async function loadAnalyses() {
    const loading = document.getElementById("history-loading");
    const errorElement = document.getElementById("history-error");
    const emptyElement = document.getElementById("history-empty");
    const tableContainer = document.getElementById("history-table-container");

    loading.hidden = false;
    errorElement.hidden = true;
    emptyElement.hidden = true;
    tableContainer.hidden = true;

    document.getElementById("history-pagination").hidden = true;

    try {
        const offset = currentPage * PAGE_SIZE;

        const response = await authenticatedFetch(
            `${API_URL}/analyses?limit=${PAGE_SIZE}&offset=${offset}`
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || "Unable to load analysis history.");
        }

        const data = await response.json();

        // The backend returns "analysis", not "analyses".
        const analyses = data.analysis || [];

        totalCount = data.count ?? analyses.length;

        document.getElementById("history-count").textContent =
            `${totalCount} RECORD${totalCount === 1 ? "" : "S"}`;

        loading.hidden = true;

        if (analyses.length === 0) {
            emptyElement.hidden = false;
            return;
        }

        renderAnalyses(analyses);

        tableContainer.hidden = false;

        updatePagination();

    } catch (error) {
        loading.hidden = true;
        errorElement.hidden = false;
        errorElement.textContent = error.message;
    }
}

async function initializeHistoryPage() {
    setupAccountDrawer();
    setupMobileSidebar();

    document.getElementById("logout-button").addEventListener("click", logout);

    document.getElementById("previous-page").addEventListener("click", async () => {
        if (currentPage > 0) {
            currentPage--;
            await loadAnalyses();
        }
    });

    document.getElementById("next-page").addEventListener("click", async () => {
        const totalPages = Math.ceil(totalCount / PAGE_SIZE);

        if (currentPage < totalPages - 1) {
            currentPage++;
            await loadAnalyses();
        }
    });

    try {
        await refreshAccessToken();
        await loadCurrentUser();
        await loadAnalyses();

    } catch (error) {
        document.getElementById("history-loading").hidden = true;

        const errorElement = document.getElementById("history-error");
        errorElement.hidden = false;
        errorElement.textContent = error.message;

        if (error.message.toLowerCase().includes("session")) {
            setTimeout(() => {
                window.location.href = "./login.html";
            }, 1500);
        }
    }
}

document.addEventListener("DOMContentLoaded", initializeHistoryPage);
