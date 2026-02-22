/**
 * InsureAI — Frontend Application Logic
 * Handles form submission, API calls, and result rendering.
 */

const API_BASE = "http://localhost:8000";

// ── DOM References ────────────────────────────────────────────────────────────
const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");

const emptyState = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");
const errorState = document.getElementById("error-state");
const errorMessage = document.getElementById("error-message");

const bundleBadge = document.getElementById("bundle-badge");
const bundleName = document.getElementById("bundle-name");
const confidenceValue = document.getElementById("confidence-value");
const confidenceBar = document.getElementById("confidence-bar");
const top3List = document.getElementById("top3-list");
const factorsList = document.getElementById("factors-list");

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

// ── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.status === "ok" && data.model_loaded) {
            statusDot.className = "status-dot online";
            statusText.textContent = `API Online · ${data.num_classes} bundles · Model ${data.model_version}`;
        } else {
            throw new Error("Model not loaded");
        }
    } catch (err) {
        statusDot.className = "status-dot offline";
        statusText.textContent = "API Offline — start the backend server to use predictions";
    }
}

// ── Form → Payload ────────────────────────────────────────────────────────────
function buildPayload(form) {
    const fd = new FormData(form);
    const payload = {};

    for (const [key, value] of fd.entries()) {
        if (value === "" || value === null) {
            payload[key] = null;
            continue;
        }
        // Numeric fields
        const numericFields = [
            "estimated_annual_income", "adult_dependents", "child_dependents",
            "infant_dependents", "previous_policy_duration_months",
            "grace_period_extensions", "years_without_claims",
            "policy_amendments_count", "vehicles_on_policy",
            "custom_riders_requested", "days_since_quote",
            "policy_start_year", "policy_start_week",
        ];
        if (numericFields.includes(key)) {
            const num = parseFloat(value);
            payload[key] = isNaN(num) ? null : num;
        } else {
            payload[key] = value || null;
        }
    }

    // broker_id: null if no broker type selected
    if (!payload.broker_agency_type) {
        payload.broker_id = null;
    }

    return payload;
}

// ── UI State Helpers ──────────────────────────────────────────────────────────
function setLoading(loading) {
    submitBtn.disabled = loading;
    if (loading) {
        btnSpinner.classList.add("visible");
        btnText.style.display = "none";
    } else {
        btnSpinner.classList.remove("visible");
        btnText.style.display = "flex";
    }
}

function showEmpty() {
    emptyState.classList.remove("hidden");
    resultContent.classList.add("hidden");
    errorState.classList.add("hidden");
}

function showError(msg) {
    emptyState.classList.add("hidden");
    resultContent.classList.add("hidden");
    errorState.classList.remove("hidden");
    errorMessage.textContent = msg;
}

function showResults(data) {
    emptyState.classList.add("hidden");
    errorState.classList.add("hidden");
    resultContent.classList.remove("hidden");
    renderResults(data);
}

// ── Result Rendering ──────────────────────────────────────────────────────────
const RANK_CLASSES = ["rank-1", "rank-2", "rank-3"];
const FILL_CLASSES = ["fill-1", "fill-2", "fill-3"];
const RANK_LABELS = ["1st", "2nd", "3rd"];

function renderResults(data) {
    // ─ Top Prediction ─────────────────────────────────────────────────────
    bundleBadge.textContent = data.predicted_bundle;
    bundleName.textContent = data.predicted_bundle_name;

    const pct = Math.round(data.confidence * 100);
    confidenceValue.textContent = `${pct}%`;

    // Small delay so CSS transition fires after display:block
    requestAnimationFrame(() => {
        confidenceBar.style.width = `${pct}%`;
    });

    // ─ Top 3 ──────────────────────────────────────────────────────────────
    top3List.innerHTML = "";
    data.top_3.forEach((item, i) => {
        const pctItem = Math.round(item.confidence * 100);
        const li = document.createElement("div");
        li.className = "top3-item";
        li.innerHTML = `
            <div class="rank-badge ${RANK_CLASSES[i]}">${RANK_LABELS[i]}</div>
            <div class="top3-bundle-info">
                <div class="top3-name">Bundle ${item.bundle_id} — ${item.bundle_name}</div>
                <div class="top3-bar-track">
                    <div class="top3-bar-fill ${FILL_CLASSES[i]}" data-pct="${pctItem}" style="width:0%"></div>
                </div>
            </div>
            <div class="top3-pct">${pctItem}%</div>
        `;
        top3List.appendChild(li);
    });

    // Animate bars after render
    requestAnimationFrame(() => {
        document.querySelectorAll(".top3-bar-fill").forEach(bar => {
            bar.style.width = `${bar.dataset.pct}%`;
        });
    });

    // ─ Key Factors ────────────────────────────────────────────────────────
    factorsList.innerHTML = "";
    if (data.key_factors && data.key_factors.length > 0) {
        data.key_factors.forEach((factor, i) => {
            const li = document.createElement("li");
            li.style.animationDelay = `${i * 0.05}s`;
            li.innerHTML = `<span class="factor-dot"></span>${factor}`;
            factorsList.appendChild(li);
        });
    } else {
        const li = document.createElement("li");
        li.innerHTML = `<span class="factor-dot"></span>Standard profile — no strong outlier attributes detected.`;
        factorsList.appendChild(li);
    }
}

// ── Form Submit Handler ───────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    setLoading(true);

    const payload = buildPayload(form);

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(15000),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error ${res.status}`);
        }

        const data = await res.json();
        showResults(data);

    } catch (err) {
        if (err.name === "TimeoutError") {
            showError("Request timed out. Is the API server running? Start it with: uvicorn backend.main:app --reload");
        } else if (err.message.includes("Failed to fetch") || err.name === "TypeError") {
            showError("Cannot reach API. Make sure the backend is running on http://localhost:8000");
        } else {
            showError(err.message || "An unexpected error occurred.");
        }
    } finally {
        setLoading(false);
    }
});

// ── Retry Button ──────────────────────────────────────────────────────────────
document.getElementById("retry-btn").addEventListener("click", () => {
    showEmpty();
});

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    // Recheck every 30 seconds
    setInterval(checkHealth, 30_000);
});
