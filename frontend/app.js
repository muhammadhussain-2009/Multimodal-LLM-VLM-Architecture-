/**
 * Socratica v2.0 — Frontend Application Logic
 * =============================================
 * - Async upload → 202 job_id → WebSocket/polling → results render
 * - Ollama-only inference (no NIM references)
 * - Real-time job progress bar
 * - Misconception heatmaps + longitudinal timeline
 */

// ─── Application State ──────────────────────────────────────────
let currentDomain = "physics";
let currentTab = "student";
let selectedFile = null;
let parsedResponseData = null;
let isAudioNarratorOn = false;
let activeJobId = null;
let activeWebSocket = null;

// ─── Constants ───────────────────────────────────────────────────
const API_BASE = window.location.origin;
const MAX_UPLOAD_SIZE = 5 * 1024 * 1024; // 5 MB
const POLL_INTERVAL_MS = 2000;

// ─── Initialize on Page Load ─────────────────────────────────────
window.addEventListener("load", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");

    // Click-to-browse
    dropZone.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File selected via browse dialog
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("drag-active");
    });

    dropZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-active");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("drag-active");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // Load teacher metrics if available
    refreshTeacherMetrics();

    // Check Ollama status
    checkOllamaStatus();
});


// ─── Tab Navigation ──────────────────────────────────────────────
function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll(".nav-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));

    if (tabName === "student") {
        document.getElementById("btn-tab-student").classList.add("active");
        document.getElementById("tab-student").classList.add("active");
        document.getElementById("header-main-text").innerText = "Student Diagram Workspace";
        document.getElementById("header-sub-text").innerText = "Upload a hand-drawn STEM diagram to receive real-time, guided Socratic feedback.";
    } else if (tabName === "teacher") {
        document.getElementById("btn-tab-teacher").classList.add("active");
        document.getElementById("tab-teacher").classList.add("active");
        document.getElementById("header-main-text").innerText = "Teacher Analytics Dashboard";
        document.getElementById("header-sub-text").innerText = "Monitor class-level misconception trends, learning gains, and student error persistence.";
        refreshTeacherMetrics();
    } else if (tabName === "stats") {
        document.getElementById("btn-tab-stats").classList.add("active");
        document.getElementById("tab-stats").classList.add("active");
        document.getElementById("header-main-text").innerText = "Statistical Analysis Lab";
        document.getElementById("header-sub-text").innerText = "Configure pre-registered parameters and run hypothesis significance tests.";
    }
}


// ─── File Upload Handler ─────────────────────────────────────────
function handleFileUpload(file) {
    // Validate file type
    const allowedTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/bmp"];
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|webp|gif|bmp)$/i)) {
        showNotification("Unsupported file type. Please upload PNG, JPG, WebP, GIF, or BMP.", "error");
        return;
    }

    // Validate file size (5 MB limit enforced on client too)
    if (file.size > MAX_UPLOAD_SIZE) {
        showNotification(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum is 5 MB.`, "error");
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const dropZone = document.getElementById("drop-zone");
        const imageWrapper = document.getElementById("image-wrapper");
        const uploadedImg = document.getElementById("uploaded-img");

        dropZone.style.display = "none";
        imageWrapper.style.display = "flex";
        uploadedImg.src = e.target.result;

        showNotification(`"${file.name}" loaded (${(file.size / 1024).toFixed(0)} KB). Click "Analyze Diagram" to begin.`, "success");
    };
    reader.readAsDataURL(file);
}


// ─── Domain Selector ─────────────────────────────────────────────
function onDomainChange() {
    const select = document.getElementById("domain-select");
    currentDomain = select.value;
}


// ─── Submit Diagram (Async Job Pattern) ──────────────────────────
async function submitDiagram() {
    if (!selectedFile) {
        showNotification("Please upload a diagram first.", "warning");
        return;
    }

    const submitBtn = document.getElementById("btn-submit-diagram");
    const statusBar = document.getElementById("job-status-bar");
    const statusText = document.getElementById("job-status-text");

    // Disable button, show progress
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Uploading...';
    statusBar.style.display = "block";
    statusText.textContent = "Uploading diagram...";

    // Reset previous results
    resetFeedbackPanel();

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("student_id", getStudentId());
        formData.append("context", `${currentDomain} diagram analysis`);

        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: "POST",
            body: formData,
        });

        if (response.status === 413) {
            throw new Error("File too large. Maximum upload size is 5 MB.");
        }

        if (response.status === 422) {
            const err = await response.json();
            throw new Error(err.detail || "Invalid image file.");
        }

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        activeJobId = data.job_id;

        statusText.textContent = "Job queued. Connecting to real-time updates...";

        // Try WebSocket first, fall back to polling
        connectWebSocket(data.job_id, data.ws_url);

    } catch (error) {
        console.error("Upload failed:", error);
        showNotification(error.message, "error");
        resetSubmitButton();
    }
}


// ─── WebSocket Connection ────────────────────────────────────────
function connectWebSocket(jobId, wsPath) {
    const statusText = document.getElementById("job-status-text");
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}${wsPath}`;

    try {
        activeWebSocket = new WebSocket(wsUrl);

        activeWebSocket.onopen = () => {
            statusText.textContent = "Connected — waiting for analysis...";
        };

        activeWebSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === "ping") return; // keepalive

            if (data.status === "processing") {
                const stageNames = {
                    1: "Stage 1: Analyzing diagram with VLM (llava:13b)...",
                    2: "Stage 2: Generating Socratic feedback (llama3.1:8b)...",
                    3: "Stage 3: Rendering feedback overlays...",
                };
                statusText.textContent = stageNames[data.stage] || data.message || "Processing...";
            }

            if (data.status === "done") {
                statusText.textContent = "✓ Analysis complete!";
                parsedResponseData = data.result;
                renderResults(data.result);
                resetSubmitButton();
                activeWebSocket.close();
            }

            if (data.status === "failed") {
                statusText.textContent = `✗ Analysis failed: ${data.error}`;
                showNotification(`Pipeline failed: ${data.error}`, "error");
                resetSubmitButton();
                activeWebSocket.close();
            }
        };

        activeWebSocket.onerror = () => {
            // WebSocket failed — fall back to polling
            console.warn("WebSocket connection failed, falling back to polling.");
            startPolling(jobId);
        };

        activeWebSocket.onclose = () => {
            activeWebSocket = null;
        };

    } catch (e) {
        console.warn("WebSocket not supported, using polling.");
        startPolling(jobId);
    }
}


// ─── Polling Fallback ────────────────────────────────────────────
function startPolling(jobId) {
    const statusText = document.getElementById("job-status-text");
    statusText.textContent = "Processing (polling for updates)...";

    const pollInterval = setInterval(async () => {
        try {
            const resp = await fetch(`${API_BASE}/api/job/${jobId}`);
            if (!resp.ok) return;

            const data = await resp.json();

            if (data.status.startsWith("processing")) {
                const stage = data.status.split(":")[1] || "";
                statusText.textContent = `Processing${stage ? ` (Stage ${stage.replace("stage", "")})` : ""}...`;
            }

            if (data.status === "done") {
                clearInterval(pollInterval);
                statusText.textContent = "✓ Analysis complete!";
                parsedResponseData = data.result;
                renderResults(data.result);
                resetSubmitButton();
            }

            if (data.status === "failed") {
                clearInterval(pollInterval);
                statusText.textContent = `✗ Failed: ${data.error}`;
                showNotification(`Pipeline failed: ${data.error}`, "error");
                resetSubmitButton();
            }

        } catch (err) {
            console.error("Polling error:", err);
        }
    }, POLL_INTERVAL_MS);
}


// ─── Results Rendering ───────────────────────────────────────────
function renderResults(result) {
    if (!result) return;

    // Expand scene graph card and render JSON
    const sgCard = document.getElementById("scenegraph-card");
    sgCard.classList.remove("collapsed");

    const sgPlaceholder = document.getElementById("scenegraph-placeholder");
    const sgJson = document.getElementById("scenegraph-json");

    const sceneGraph = result.scene_graph || result;
    sgPlaceholder.style.display = "none";
    sgJson.style.display = "block";
    sgJson.textContent = JSON.stringify(sceneGraph, null, 2);

    // Render feedback
    const feedbackPlaceholder = document.getElementById("feedback-placeholder");
    const feedbackContent = document.getElementById("feedback-content");
    const misconceptionList = document.getElementById("misconception-list");
    const promptBubble = document.getElementById("socratic-prompt-bubble");

    feedbackPlaceholder.style.display = "none";
    feedbackContent.style.display = "block";

    const feedbackItems = result.evaluated_elements || result.feedback_items || result.feedback || [];
    const misconceptions = feedbackItems.filter(f => f.status === "incorrect" || f.misconception_type);

    // Clear and render misconceptions
    misconceptionList.innerHTML = "";
    
    if (result.identified_subdomain) {
        const subEl = document.createElement("div");
        subEl.className = "misconception-tag";
        subEl.style.backgroundColor = "#4a90e2";
        subEl.style.color = "white";
        subEl.innerHTML = `
            <span class="tag-icon"><i class="fa-solid fa-layer-group"></i></span>
            <span class="tag-text">${escapeHtml(result.identified_subdomain)}</span>
        `;
        misconceptionList.appendChild(subEl);
    }

    misconceptions.forEach((misc, idx) => {
        if (misc.misconception_type) {
            const el = document.createElement("div");
            el.className = "misconception-tag";
            el.innerHTML = `
                <span class="tag-icon"><i class="fa-solid fa-triangle-exclamation"></i></span>
                <span class="tag-text">${escapeHtml(misc.misconception_type)}</span>
            `;
            misconceptionList.appendChild(el);
        }
    });

    if (misconceptions.length === 0 && !result.identified_subdomain) {
        misconceptionList.innerHTML = '<div class="no-issues"><i class="fa-solid fa-circle-check"></i> No major misconceptions detected.</div>';
    }

    // Build Socratic prompt bubble with typewriter effect
    promptBubble.innerHTML = "";
    if (feedbackItems.length === 0) {
        promptBubble.innerHTML = "<p>No feedback generated.</p>";
    } else {
        function typeWriter(itemIndex) {
            if (itemIndex >= feedbackItems.length) return;
            const item = feedbackItems[itemIndex];
            
            // Backwards compatibility with old schema
            const isCorrect = item.status === "correct" || item.type === "affirmative";
            const textToType = item.feedback || item.text || "";
            
            const icon = isCorrect
                ? '<i class="fa-solid fa-circle-check text-green"></i>'
                : '<i class="fa-solid fa-lightbulb text-yellow"></i>';
                
            const div = document.createElement("div");
            div.className = "feedback-item";
            div.innerHTML = `<span class="fi-icon">${icon}</span><p class="typewriter-text"></p>`;
            promptBubble.appendChild(div);
            
            const p = div.querySelector(".typewriter-text");
            const words = textToType.split(" ");
            let wIdx = 0;
            
            function typeWord() {
                if (wIdx < words.length) {
                    p.innerHTML += (wIdx > 0 ? " " : "") + escapeHtml(words[wIdx]);
                    wIdx++;
                    setTimeout(typeWord, 30);
                } else {
                    setTimeout(() => typeWriter(itemIndex + 1), 200);
                }
            }
            typeWord();
        }
        typeWriter(0);
    }

    // Draw overlay annotations on canvas (if bounding boxes exist)
    drawOverlayAnnotations(sceneGraph);

    showNotification("Analysis complete! Review the feedback below.", "success");
}


// ─── Canvas Overlay Drawing ──────────────────────────────────────
function drawOverlayAnnotations(sceneGraph) {
    const canvas = document.getElementById("overlay-canvas");
    const img = document.getElementById("uploaded-img");
    if (!canvas || !img) return;

    const wrapper = document.getElementById("image-wrapper");
    canvas.width = wrapper.offsetWidth;
    canvas.height = wrapper.offsetHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const elements = sceneGraph.elements || [];
    elements.forEach((el, idx) => {
        if (!el.bbox || !Array.isArray(el.bbox) || el.bbox.length < 4) return;

        const [x1, y1, x2, y2] = el.bbox;
        const w = canvas.width;
        const h = canvas.height;

        const px1 = x1 * w, py1 = y1 * h;
        const px2 = x2 * w, py2 = y2 * h;

        // Color by type
        const colors = { vector: "#f59e0b", label: "#22c55e", node: "#6366f1" };
        const color = colors[el.type] || "#94a3b8";

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);

        // Label
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = "11px 'Outfit', sans-serif";
        ctx.fillText(el.label || el.id, px1 + 4, py1 - 4);
    });
}


// ─── Collapsible Cards ──────────────────────────────────────────
function toggleCard(cardId) {
    const card = document.getElementById(cardId);
    card.classList.toggle("collapsed");
}


// ─── Accessibility / Audio Narrator ──────────────────────────────
function toggleAccessibilityMode() {
    isAudioNarratorOn = !isAudioNarratorOn;
    const icon = document.getElementById("audio-icon");

    if (isAudioNarratorOn) {
        icon.className = "fa-solid fa-volume-high";
        showNotification("Audio narrator enabled.", "info");

        // Read latest feedback aloud
        if (parsedResponseData) {
            const items = parsedResponseData.feedback_items || [];
            const text = items.map(i => i.text).join(". ");
            if (text && "speechSynthesis" in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 0.9;
                utterance.pitch = 1.0;
                speechSynthesis.speak(utterance);
            }
        }
    } else {
        icon.className = "fa-solid fa-volume-xmark";
        if ("speechSynthesis" in window) speechSynthesis.cancel();
        showNotification("Audio narrator disabled.", "info");
    }
}


// ─── Ollama Status Check ─────────────────────────────────────────
async function checkOllamaStatus() {
    const statusEl = document.getElementById("ollama-status");
    if (!statusEl) return;

    try {
        const resp = await fetch(`${API_BASE}/api/health`);
        const data = await resp.json();

        if (data.ollama && data.ollama.reachable) {
            const models = data.ollama.models || [];
            statusEl.innerHTML = `<i class="fa-solid fa-circle-check" style="color:#22c55e;"></i> Ollama connected (${models.length} model${models.length !== 1 ? 's' : ''} loaded)`;

            // Update connection status indicator in header
            const indicator = document.querySelector(".status-indicator");
            if (indicator) {
                indicator.classList.add("online");
                indicator.classList.remove("offline");
            }
        } else {
            statusEl.innerHTML = '<i class="fa-solid fa-circle-xmark" style="color:#ef4444;"></i> Ollama not reachable — will use fallback mode';
        }
    } catch (e) {
        statusEl.innerHTML = '<i class="fa-solid fa-circle-xmark" style="color:#ef4444;"></i> API server not running';
    }
}


// ─── Teacher Analytics Dashboard ─────────────────────────────────
async function refreshTeacherMetrics() {
    try {
        const resp = await fetch(`${API_BASE}/api/analytics?top_n=10`);
        if (!resp.ok) return;

        const data = await resp.json();
        const miscs = data.top_misconceptions || [];

        // Update heatmap bar chart
        const barChart = document.getElementById("bar-chart");
        if (barChart) {
            barChart.innerHTML = "";
            if (miscs.length === 0) {
                barChart.innerHTML = '<p style="color: #94a3b8; text-align:center;">No misconception data yet. Analyze diagrams to populate.</p>';
            } else {
                const maxCount = Math.max(...miscs.map(m => m.total_occurrences || 1));
                miscs.forEach(misc => {
                    const pct = ((misc.total_occurrences || 1) / maxCount * 100).toFixed(0);
                    const row = document.createElement("div");
                    row.className = "bar-row";
                    row.innerHTML = `
                        <span class="bar-label">${escapeHtml(misc.misconception_class)}</span>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: ${pct}%;"></div>
                        </div>
                        <span class="bar-value">${misc.total_occurrences}×</span>
                    `;
                    barChart.appendChild(row);
                });
            }
        }

        // Update longitudinal timeline (from recent jobs)
        const jobsResp = await fetch(`${API_BASE}/api/jobs?limit=10`);
        if (jobsResp.ok) {
            const jobs = await jobsResp.json();
            renderLongitudinalTimeline(jobs);
        }

    } catch (e) {
        console.log("Could not fetch analytics:", e);
        renderFallbackAnalytics();
    }
}

function renderLongitudinalTimeline(jobs) {
    const timeline = document.getElementById("longitudinal-timeline");
    if (!timeline) return;

    timeline.innerHTML = "";

    if (!jobs || jobs.length === 0) {
        timeline.innerHTML = '<p style="color: #94a3b8; text-align:center;">No job history yet.</p>';
        return;
    }

    jobs.slice(0, 8).forEach((job, idx) => {
        const date = new Date((job.submitted_at || 0) * 1000);
        const dateStr = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
        const statusIcon = job.status === "done"
            ? '<i class="fa-solid fa-circle-check text-green"></i>'
            : job.status === "failed"
                ? '<i class="fa-solid fa-circle-xmark text-red"></i>'
                : '<i class="fa-solid fa-clock text-yellow"></i>';

        const point = document.createElement("div");
        point.className = "timeline-point";
        point.innerHTML = `
            <div class="timeline-marker">${statusIcon}</div>
            <div class="timeline-label">
                <strong>${dateStr}</strong>
                <span>${escapeHtml(job.domain || "unknown")} · ${job.status}</span>
            </div>
        `;
        timeline.appendChild(point);
    });
}

function renderFallbackAnalytics() {
    // Pre-populate with demo data when API is unavailable
    const barChart = document.getElementById("bar-chart");
    if (!barChart) return;

    const demoData = [
        { label: "Force Vector Direction", pct: 85, count: 12 },
        { label: "Energy Conservation", pct: 65, count: 9 },
        { label: "Photosynthesis Source", pct: 50, count: 7 },
        { label: "Bohr Model Overextension", pct: 40, count: 5 },
        { label: "Current Consumption", pct: 25, count: 3 },
    ];

    barChart.innerHTML = "";
    demoData.forEach(d => {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
            <span class="bar-label">${d.label}</span>
            <div class="bar-track"><div class="bar-fill" style="width: ${d.pct}%;"></div></div>
            <span class="bar-value">${d.count}×</span>
        `;
        barChart.appendChild(row);
    });

    // Demo longitudinal timeline
    const timeline = document.getElementById("longitudinal-timeline");
    if (timeline) {
        const demoTimeline = [
            { date: "Jun 15", domain: "Physics", status: "3 errors", icon: "text-red" },
            { date: "Jun 22", domain: "Physics", status: "2 errors", icon: "text-yellow" },
            { date: "Jun 29", domain: "Biology", status: "1 error", icon: "text-yellow" },
            { date: "Jul 06", domain: "Physics", status: "0 errors", icon: "text-green" },
        ];
        timeline.innerHTML = "";
        demoTimeline.forEach(t => {
            const point = document.createElement("div");
            point.className = "timeline-point";
            point.innerHTML = `
                <div class="timeline-marker"><i class="fa-solid fa-circle ${t.icon}"></i></div>
                <div class="timeline-label"><strong>${t.date}</strong><span>${t.domain} · ${t.status}</span></div>
            `;
            timeline.appendChild(point);
        });
    }

    // Taxonomy explorer
    renderTaxonomyExplorer();
}


// ─── Taxonomy Explorer ───────────────────────────────────────────
function renderTaxonomyExplorer() {
    const explorer = document.getElementById("taxonomy-explorer");
    if (!explorer) return;

    const taxonomy = {
        "Physics": [
            { id: "PHY-01", name: "Gravity Misconception", desc: "Heavier objects fall faster" },
            { id: "PHY-02", name: "Force Vector Error", desc: "Confuses force and velocity direction" },
            { id: "PHY-03", name: "Energy Transfer", desc: "Energy is 'used up' not transferred" },
            { id: "PHY-04", name: "Current Consumption", desc: "Current consumed by resistors" },
        ],
        "Biology": [
            { id: "BIO-01", name: "Photosynthesis Source", desc: "Plants get food from soil" },
            { id: "BIO-02", name: "Lamarckian Inheritance", desc: "Acquired traits inherited" },
            { id: "BIO-03", name: "Teleological Evolution", desc: "Evolution has a goal" },
        ],
        "Chemistry": [
            { id: "CHM-01", name: "Bohr Model Overextension", desc: "Electrons in fixed orbits" },
            { id: "CHM-02", name: "Physical vs Chemical", desc: "Bonds break in physical change" },
            { id: "CHM-03", name: "Conservation of Mass", desc: "Products disappear in reactions" },
        ],
    };

    explorer.innerHTML = "";
    Object.entries(taxonomy).forEach(([domain, items]) => {
        const section = document.createElement("div");
        section.className = "taxonomy-section";
        section.innerHTML = `
            <div class="taxonomy-header" onclick="this.parentElement.classList.toggle('open')">
                <i class="fa-solid fa-chevron-right"></i>
                <strong>${domain}</strong> (${items.length} misconceptions)
            </div>
            <div class="taxonomy-items">
                ${items.map(item => `
                    <div class="taxonomy-item">
                        <span class="taxonomy-id">${item.id}</span>
                        <span class="taxonomy-name">${item.name}</span>
                        <span class="taxonomy-desc">${item.desc}</span>
                    </div>
                `).join("")}
            </div>
        `;
        explorer.appendChild(section);
    });
}


// ─── Statistical Lab ─────────────────────────────────────────────
function updateStatsParams() {
    const power = document.getElementById("slider-power").value;
    const alpha = document.getElementById("slider-alpha").value;
    const effect = document.getElementById("slider-effect").value;

    document.getElementById("val-power").textContent = `${Math.round(power * 100)}%`;
    document.getElementById("val-alpha").textContent = parseFloat(alpha).toFixed(2);
    document.getElementById("val-effect").textContent = parseFloat(effect).toFixed(2);
}

function triggerStatisticalRun() {
    const power = parseFloat(document.getElementById("slider-power").value);
    const alpha = parseFloat(document.getElementById("slider-alpha").value);
    const d = parseFloat(document.getElementById("slider-effect").value);

    // Two-tailed z-test sample size formula: n = ((z_α/2 + z_β) / d)²
    const z_alpha = jstat_qnorm(1 - alpha / 2);
    const z_beta = jstat_qnorm(power);
    const n = Math.ceil(((z_alpha + z_beta) / d) ** 2);

    document.getElementById("stats-n-value").textContent = n;

    // Simulate t-test with demo data
    const n_sim = Math.min(n, 200);
    const control_mean = 0.38;
    const treatment_mean = control_mean + d * 0.35;
    const se = 0.35 / Math.sqrt(n_sim);
    const t_stat = (treatment_mean - control_mean) / (se * Math.sqrt(2));
    const df = 2 * n_sim - 2;
    const p_value = 2 * (1 - t_cdf(Math.abs(t_stat), df));

    document.getElementById("stat-mean-control").textContent = control_mean.toFixed(3);
    document.getElementById("stat-mean-treatment").textContent = treatment_mean.toFixed(3);
    document.getElementById("stat-effect-size").textContent = d.toFixed(3);
    document.getElementById("stat-t-value").textContent = t_stat.toFixed(3);
    document.getElementById("stats-p-value").textContent = `p = ${p_value.toFixed(4)}`;

    const verdictEl = document.getElementById("stats-verdict-text");
    if (p_value < alpha) {
        verdictEl.textContent = `Significant (p < ${alpha})`;
        verdictEl.className = "p-subtext text-green";
    } else {
        verdictEl.textContent = `Not Significant (p ≥ ${alpha})`;
        verdictEl.className = "p-subtext text-red";
    }
}


// ─── Statistical Helper Functions ────────────────────────────────
function jstat_qnorm(p) {
    // Rational approximation for inverse normal CDF (Abramowitz & Stegun)
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    if (p === 0.5) return 0;

    const a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    ];
    const b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01
    ];
    const c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00
    ];
    const d_coeff = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    ];

    const p_low = 0.02425, p_high = 1 - p_low;
    let q, r;

    if (p < p_low) {
        q = Math.sqrt(-2 * Math.log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d_coeff[0]*q+d_coeff[1])*q+d_coeff[2])*q+d_coeff[3])*q+1);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    } else {
        q = Math.sqrt(-2 * Math.log(1 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d_coeff[0]*q+d_coeff[1])*q+d_coeff[2])*q+d_coeff[3])*q+1);
    }
}

function t_cdf(t, df) {
    // Approximation of Student's t CDF using regularized incomplete beta
    const x = df / (df + t * t);
    return 1 - 0.5 * regularizedBeta(x, df / 2, 0.5);
}

function regularizedBeta(x, a, b) {
    // Simple continued fraction approximation
    const lnBeta = gammaln(a) + gammaln(b) - gammaln(a + b);
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta) / a;
    let f = 1, c = 1, d_val = 0;

    for (let i = 0; i <= 200; i++) {
        let m = i / 2;
        let numerator;
        if (i === 0) {
            numerator = 1;
        } else if (i % 2 === 0) {
            numerator = (m * (b - m) * x) / ((a + 2*m - 1) * (a + 2*m));
        } else {
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2*m) * (a + 2*m + 1));
        }

        d_val = 1 + numerator * d_val;
        if (Math.abs(d_val) < 1e-30) d_val = 1e-30;
        d_val = 1 / d_val;

        c = 1 + numerator / c;
        if (Math.abs(c) < 1e-30) c = 1e-30;

        f *= d_val * c;
        if (Math.abs(d_val * c - 1) < 1e-8) break;
    }

    return front * (f - 1);
}

function gammaln(x) {
    const g = 7;
    const coef = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - gammaln(1 - x);
    x -= 1;
    let a = coef[0];
    const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += coef[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}


// ─── Utility Functions ───────────────────────────────────────────
function getStudentId() {
    // Use sessionStorage to persist a simple device-session ID
    let sid = sessionStorage.getItem("socratica_student_id");
    if (!sid) {
        sid = "student_" + Math.random().toString(36).substring(2, 10);
        sessionStorage.setItem("socratica_student_id", sid);
    }
    return sid;
}

function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function resetSubmitButton() {
    const btn = document.getElementById("btn-submit-diagram");
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-microchip"></i> Analyze Diagram';
}

function resetFeedbackPanel() {
    const feedbackPlaceholder = document.getElementById("feedback-placeholder");
    const feedbackContent = document.getElementById("feedback-content");
    const sgPlaceholder = document.getElementById("scenegraph-placeholder");
    const sgJson = document.getElementById("scenegraph-json");

    if (feedbackPlaceholder) feedbackPlaceholder.style.display = "block";
    if (feedbackContent) feedbackContent.style.display = "none";
    if (sgPlaceholder) sgPlaceholder.style.display = "block";
    if (sgJson) { sgJson.style.display = "none"; sgJson.textContent = ""; }
}

function showNotification(message, type = "info") {
    // Remove existing notifications
    document.querySelectorAll(".toast-notification").forEach(t => t.remove());

    const toast = document.createElement("div");
    toast.className = `toast-notification toast-${type}`;

    const icons = {
        success: "fa-circle-check",
        error: "fa-circle-xmark",
        warning: "fa-triangle-exclamation",
        info: "fa-circle-info",
    };

    toast.innerHTML = `
        <i class="fa-solid ${icons[type] || icons.info}"></i>
        <span>${escapeHtml(message)}</span>
    `;

    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => toast.classList.add("visible"));

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        toast.classList.remove("visible");
        setTimeout(() => toast.remove(), 400);
    }, 5000);
}

