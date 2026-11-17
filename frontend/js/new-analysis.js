
const API_URL = "http://localhost:8000";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const ALLOWED_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
];

let accessToken = null;
let selectedFile = null;
let previewURL = null;


// DOM elements

const imageInput = document.getElementById("image-input");
const uploadZone = document.getElementById("upload-zone");
const selectedFilePanel = document.getElementById("selected-file");

const previewImage = document.getElementById("preview-image");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");

const uploadError = document.getElementById("upload-error");
const submitButton = document.getElementById("submit-analysis");
const removeButton = document.getElementById("remove-file");

const uploadPanel = document.querySelector(".upload-panel");
const processingPanel = document.getElementById("processing-panel");


// Account elements

const accountTrigger = document.getElementById("account-trigger");
const topbarAvatar = document.getElementById("topbar-avatar");

const accountDrawer = document.getElementById("account-drawer");
const drawerBackdrop = document.getElementById("drawer-backdrop");
const drawerClose = document.getElementById("drawer-close");

const logoutButton = document.getElementById("logout-button");


// Sidebar elements

const sidebar = document.getElementById("sidebar");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const mobileMenuButton = document.getElementById("mobile-menu-button");


// Utility functions

function getInitial(name) {
    return name?.trim()?.charAt(0)?.toUpperCase() || "?";
}


function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}


function showError(message) {
    uploadError.textContent = message;
    uploadError.classList.add("visible");
}


function clearError() {
    uploadError.textContent = "";
    uploadError.classList.remove("visible");
}


// Authentication

async function refreshAccessToken() {
    const response = await fetch(`${API_URL}/auth/refresh`, {
        method: "POST",
        credentials: "include"
    });

    if (!response.ok) {
        throw new Error("Authentication expired.");
    }

    const data = await response.json();

    if (!data.access_token) {
        throw new Error("Could not refresh authentication.");
    }

    accessToken = data.access_token;

    return accessToken;
}


async function authenticatedFetch(url, options = {}) {
    if (!accessToken) {
        await refreshAccessToken();
    }

    const makeRequest = () => {
        const headers = new Headers(options.headers || {});

        headers.set("Authorization", `Bearer ${accessToken}`);

        return fetch(url, {
            ...options,
            headers,
            credentials: "include"
        });
    };

    let response = await makeRequest();

    if (response.status === 401) {
        await refreshAccessToken();
        response = await makeRequest();
    }

    if (response.status === 401) {
        throw new Error("Authentication expired.");
    }

    return response;
}


async function getCurrentUser() {
    const response = await authenticatedFetch(
        `${API_URL}/auth/me`
    );

    if (!response.ok) {
        throw new Error("Could not retrieve account details.");
    }

    const data = await response.json();

    return data.user;
}


function updateAccountUI(user) {
    const name = user.name || user.username || "User";
    const email = user.email || "";
    const initial = getInitial(name);

    document.getElementById("sidebar-avatar").textContent = initial;
    topbarAvatar.textContent = initial;
    document.getElementById("drawer-avatar").textContent = initial;

    document.getElementById("sidebar-name").textContent = name;
    document.getElementById("drawer-name").textContent = name;
    document.getElementById("drawer-email").textContent = email;
}


// File validation

function validateFile(file) {
    if (!file) {
        return "Please select an image.";
    }

    const extension = file.name
        .substring(file.name.lastIndexOf("."))
        .toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(extension)) {
        return "Please select a PNG, JPEG or WebP image.";
    }

    if (file.size === 0) {
        return "The selected file is empty.";
    }

    if (file.size > MAX_FILE_SIZE) {
        return "The selected image exceeds the 10 MB limit.";
    }

    return null;
}


// File selection

function selectFile(file) {
    clearError();

    const error = validateFile(file);

    if (error) {
        showError(error);
        return;
    }

    if (previewURL) {
        URL.revokeObjectURL(previewURL);
    }

    selectedFile = file;
    previewURL = URL.createObjectURL(file);

    previewImage.src = previewURL;

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    uploadZone.classList.add("hidden");
    selectedFilePanel.classList.add("visible");
    uploadPanel.classList.add("has-file");

    submitButton.disabled = false;
}


function removeFile() {
    selectedFile = null;

    if (previewURL) {
        URL.revokeObjectURL(previewURL);
        previewURL = null;
    }

    imageInput.value = "";
    previewImage.removeAttribute("src");

    selectedFilePanel.classList.remove("visible");
    uploadZone.classList.remove("hidden");
    uploadPanel.classList.remove("has-file");

    submitButton.disabled = true;

    clearError();
}


// Upload interactions

uploadZone.addEventListener("click", () => {
    imageInput.click();
});


uploadZone.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        imageInput.click();
    }
});


imageInput.addEventListener("change", event => {
    const file = event.target.files[0];

    if (file) {
        selectFile(file);
    }

    imageInput.value = "";
});


uploadZone.addEventListener("dragover", event => {
    event.preventDefault();

    uploadZone.classList.add("drag-over");
});


uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});


uploadZone.addEventListener("drop", event => {
    event.preventDefault();

    uploadZone.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (file) {
        selectFile(file);
    }
});


removeButton.addEventListener("click", removeFile);


// Analysis submission

async function submitAnalysis() {
    clearError();

    if (!selectedFile) {
        showError("Please select an image before starting the analysis.");
        return;
    }

    const validationError = validateFile(selectedFile);

    if (validationError) {
        showError(validationError);
        return;
    }

    submitButton.disabled = true;
    submitButton.querySelector("span:first-child").textContent = "Processing...";

    removeButton.disabled = true;
    uploadZone.style.pointerEvents = "none";

    processingPanel.classList.add("visible");
    processingPanel.setAttribute("aria-hidden", "false");

    try {
        const formData = new FormData();

        formData.append("file", selectedFile);

        const response = await authenticatedFetch(
            `${API_URL}/analyses`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Analysis processing failed."
            );
        }

        if (!data.analysis_id) {
            throw new Error(
                "The server did not return an analysis ID."
            );
        }

        // Open the results page for this analysis.
        window.location.href =
            `./analysis.html?id=${encodeURIComponent(data.analysis_id)}`;

    } catch (error) {
        console.error("Analysis submission failed:", error);

        processingPanel.classList.remove("visible");
        processingPanel.setAttribute("aria-hidden", "true");

        submitButton.disabled = false;
        submitButton.querySelector("span:first-child").textContent =
            "Run inference";

        removeButton.disabled = false;
        uploadZone.style.pointerEvents = "auto";

        if (error.message.includes("Authentication")) {
            window.location.href = "./login.html";
            return;
        }

        showError(
            error.message || "Something went wrong while processing the image."
        );
    }
}


submitButton.addEventListener("click", submitAnalysis);


// Account drawer

function openAccountDrawer() {
    accountDrawer.classList.add("open");
    drawerBackdrop.classList.add("open");

    accountDrawer.setAttribute("aria-hidden", "false");
}


function closeAccountDrawer() {
    accountDrawer.classList.remove("open");
    drawerBackdrop.classList.remove("open");

    accountDrawer.setAttribute("aria-hidden", "true");
}


accountTrigger.addEventListener("click", openAccountDrawer);

topbarAvatar.addEventListener("click", openAccountDrawer);

drawerClose.addEventListener("click", closeAccountDrawer);

drawerBackdrop.addEventListener("click", closeAccountDrawer);


// Mobile sidebar

function openSidebar() {
    sidebar.classList.add("open");
    sidebarBackdrop.classList.add("open");
}


function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarBackdrop.classList.remove("open");
}


mobileMenuButton.addEventListener("click", openSidebar);

sidebarBackdrop.addEventListener("click", closeSidebar);


// Keyboard interactions

document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
        closeAccountDrawer();
        closeSidebar();
    }
});


// Logout

logoutButton.addEventListener("click", async () => {
    logoutButton.disabled = true;

    try {
        await authenticatedFetch(`${API_URL}/auth/logout`, {
            method: "POST"
        });
    } catch (error) {
        console.error("Logout request failed:", error);
    }

    accessToken = null;

    window.location.href = "./login.html";
});


// Initialize page

async function initializePage() {
    try {
        await refreshAccessToken();

        const user = await getCurrentUser();

        updateAccountUI(user);

    } catch (error) {
        console.error("Page initialization failed:", error);

        window.location.href = "./login.html";
    }
}


initializePage();
