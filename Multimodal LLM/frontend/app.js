// Application State
let currentDomain = "physics";
let currentTab = "student";
let selectedFile = null;
let parsedResponseData = null;
let isAudioNarratorOn = false;

// Initialize on page load
window.addEventListener("load", () => {
    // Set up file upload handlers
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");

    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--color-primary)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--border-color)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // Load initial preset
    loadDemoPreset("physics");
    // Retrieve metrics for analytics tab
    refreshTeacherMetrics();
});

// Tab navigation controller
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
        document.getElementById("header-main-text").innerText = "Preregistered Statistics Lab";
        document.getElementById("header-sub-text").innerText = "Run significance power calculations and Welch independent T-tests on study cohorts.";
        triggerStatisticalRun();
    }
}

// Handle collapsible VLM Scene graph cards
function toggleCard(cardId) {
    const card = document.getElementById(cardId);
    card.classList.toggle("collapsed");
}

// Dropzone file processor
function handleFileUpload(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const dropZone = document.getElementById("drop-zone");
        const imgWrapper = document.getElementById("image-wrapper");
        const uploadedImg = document.getElementById("uploaded-img");

        dropZone.style.display = "none";
        imgWrapper.style.display = "block";
        uploadedImg.src = e.target.result;

        // Wait for image load to adjust canvas bounds
        uploadedImg.onload = () => {
            const canvas = document.getElementById("overlay-canvas");
            canvas.width = uploadedImg.clientWidth;
            canvas.height = uploadedImg.clientHeight;
            
            // Clear previous canvas overlays
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        };
    };
    reader.readAsDataURL(file);
}

// Domain Selector Listener
function onDomainChange() {
    currentDomain = document.getElementById("domain-select").value;
    loadDemoPreset(currentDomain);
}

// Load Diagram Visual Presets (Dynamic Canvas Drawing)
function loadDemoPreset(domain) {
    currentDomain = domain;
    document.getElementById("domain-select").value = domain;

    const dropZone = document.getElementById("drop-zone");
    const imgWrapper = document.getElementById("image-wrapper");
    const uploadedImg = document.getElementById("uploaded-img");
    const canvas = document.getElementById("overlay-canvas");

    // Hide drag-drop, show canvas image-wrapper
    dropZone.style.display = "none";
    imgWrapper.style.display = "block";

    // Since we don't have static png files, we construct a virtual 600x450 canvas,
    // export it as a base64 DataURL, and set it as the src of the uploadedImg.
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 600;
    tempCanvas.height = 450;
    const ctx = tempCanvas.getContext("2d");

    // Clear and fill dark slate background
    ctx.fillStyle = "#0c1222";
    ctx.fillRect(0, 0, 600, 450);

    if (domain === "physics") {
        drawPhysicsPreset(ctx);
    } else if (domain === "biology") {
        drawBiologyPreset(ctx);
    } else {
        drawChemistryPreset(ctx);
    }

    uploadedImg.src = tempCanvas.toDataURL("image/png");
    uploadedImg.onload = () => {
        canvas.width = uploadedImg.clientWidth;
        canvas.height = uploadedImg.clientHeight;
        const oCtx = canvas.getContext("2d");
        oCtx.clearRect(0, 0, canvas.width, canvas.height);
    };

    // Reset workspace outputs
    resetOutputs();
}

/* ================= DRAWING PRESETS ================= */
function drawPhysicsPreset(ctx) {
    // Draw Inclined Ramp
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(100, 380); // Bottom-left
    ctx.lineTo(500, 380); // Bottom-right
    ctx.lineTo(500, 180); // Top-right
    ctx.closePath();
    ctx.fillStyle = "#1e293b";
    ctx.fill();
    ctx.stroke();

    // Draw Incline Angle Theta
    ctx.font = "bold 16px 'Outfit'";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("θ = 30°", 160, 360);

    // Draw Block on Ramp (rotated)
    ctx.save();
    ctx.translate(300, 280);
    ctx.rotate(-Math.atan(200/400)); // Ramp slope
    
    // Draw Box representing block
    ctx.fillStyle = "#334155";
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 3;
    ctx.fillRect(-45, -45, 90, 90);
    ctx.strokeRect(-45, -45, 90, 90);
    ctx.restore();

    // Student drawn force vectors
    // 1. Gravity vector (Fg) pulling straight down (correctly drawn)
    drawArrow(ctx, 300, 280, 300, 390, "#10b981", 3);
    ctx.fillStyle = "#10b981";
    ctx.font = "14px 'Outfit'";
    ctx.fillText("Gravity (Fg)", 315, 380);

    // 2. Student drawn Friction Vector (Ff) pointing DOWN the ramp (Error!)
    // Angle of slope is ~26.5 degrees
    const rAngle = 26.5 * Math.PI / 180;
    const fX = 300 + 80 * Math.cos(rAngle);
    const fY = 280 + 80 * Math.sin(rAngle);
    drawArrow(ctx, 300, 280, fX, fY, "#ef4444", 3);
    ctx.fillStyle = "#ef4444";
    ctx.fillText("Friction (Ff)", fX + 10, fY + 10);
    
    // Label showing missing normal force placeholder
    ctx.fillStyle = "rgba(239, 68, 68, 0.1)";
    ctx.strokeStyle = "rgba(239, 68, 68, 0.4)";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(260, 160, 80, 80);
    ctx.fillStyle = "#ef4444";
    ctx.fillText("?", 295, 205);
    ctx.setLineDash([]);
}

function drawBiologyPreset(ctx) {
    // Draw Cell Wall boundary (Rectangular)
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 6;
    ctx.fillStyle = "#111c1e";
    ctx.fillRect(80, 60, 440, 330);
    ctx.strokeRect(80, 60, 440, 330);

    // Inner Plasma Membrane
    ctx.strokeStyle = "#059669";
    ctx.lineWidth = 2;
    ctx.strokeRect(90, 70, 420, 310);

    // Nucleus (Center)
    ctx.beginPath();
    ctx.arc(320, 220, 45, 0, 2 * Math.PI);
    ctx.fillStyle = "#1e1b4b";
    ctx.strokeStyle = "#4f46e5";
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#818cf8";
    ctx.font = "bold 13px 'Outfit'";
    ctx.fillText("Nucleus", 295, 225);

    // Large Vacuole (Right side)
    ctx.beginPath();
    ctx.ellipse(430, 220, 50, 80, 0, 0, 2 * Math.PI);
    ctx.fillStyle = "#0c2540";
    ctx.strokeStyle = "#0284c7";
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#38bdf8";
    ctx.fillText("Vacuole", 405, 225);

    // Chloroplast organelle labeled incorrectly on top left (mislabeled as Mitochondrion)
    ctx.save();
    ctx.translate(180, 130);
    ctx.beginPath();
    ctx.ellipse(0, 0, 30, 18, Math.PI/6, 0, 2 * Math.PI);
    ctx.fillStyle = "#064e3b";
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
    
    // Draw stacks representing thylakoids (indicates it is chloroplast)
    ctx.fillStyle = "#059669";
    ctx.fillRect(-15, -6, 10, 4);
    ctx.fillRect(-15, 2, 10, 4);
    ctx.fillRect(5, -4, 10, 4);
    ctx.restore();

    ctx.fillStyle = "#ef4444";
    ctx.fillText("Mitochondrion", 130, 180);
    drawArrow(ctx, 180, 165, 180, 145, "#ef4444", 2);
}

function drawChemistryPreset(ctx) {
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, 600, 450);

    // Draw linear geometry water molecule O-H-O style (linear mistake)
    ctx.font = "bold 42px 'Outfit'";
    
    // Atom H1
    ctx.fillStyle = "#60a5fa";
    ctx.fillText("H", 120, 230);
    
    // Atom O
    ctx.fillStyle = "#f87171";
    ctx.fillText("O", 285, 230);

    // Atom H2
    ctx.fillStyle = "#60a5fa";
    ctx.fillText("H", 450, 230);

    // Connective Single Covalent Bonds (lines)
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(170, 215);
    ctx.lineTo(265, 215);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(335, 215);
    ctx.lineTo(430, 215);
    ctx.stroke();

    // Labels
    ctx.font = "14px 'Outfit'";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("Single Bond", 180, 195);
    ctx.fillText("Single Bond", 350, 195);

    // Highlight missing lone pairs on oxygen
    ctx.fillStyle = "rgba(239, 68, 68, 0.05)";
    ctx.strokeStyle = "rgba(239, 68, 68, 0.4)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(300, 215, 36, 0, 2*Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
}

function drawArrow(ctx, fromX, fromY, toX, toY, color, width) {
    const headLength = 12; // Length of head
    const dx = toX - fromX;
    const dy = toY - fromY;
    const angle = Math.atan2(dy, dx);

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();

    // Draw arrow head
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
}

function resetOutputs() {
    document.getElementById("scenegraph-placeholder").style.display = "flex";
    document.getElementById("scenegraph-json").style.display = "none";
    document.getElementById("feedback-placeholder").style.display = "flex";
    document.getElementById("feedback-content").style.display = "none";
}

/* ================= API SUBMISSIONS ================= */

// Helper to convert base64 DataURL directly to binary Blob synchronously
function dataURLtoBlob(dataurl) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
}

// Submits diagram canvas/upload to API
async function submitDiagram() {
    const btn = document.getElementById("btn-submit-diagram");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing pipeline...';

    try {
        const formData = new FormData();
        formData.append("domain", currentDomain);
        formData.append("vlm_provider", "mock");
        formData.append("llm_provider", "mock");
        formData.append("rubric", "Standard grading rubric");
        formData.append("student_grade", "9th Grade");
        formData.append("support_tier", "standard");
        formData.append("student_id", "stud_" + Math.floor(Math.random() * 1000));

        // Get file blob from uploaded file or local preset
        let blob;
        if (selectedFile) {
            blob = selectedFile;
        } else {
            const imgEl = document.getElementById("uploaded-img");
            blob = dataURLtoBlob(imgEl.src);
        }
        formData.append("image", blob, selectedFile ? selectedFile.name : "diagram.png");

        const response = await fetch("/api/evaluate_diagram", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("API call failed.");
        }

        const data = await response.json();
        parsedResponseData = data;
        
        displayPipelineResults(data);

    } catch (err) {
        console.error(err);
        alert("Pipeline error occurred. Check backend logs.");
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-microchip"></i> Run Multimodal Pipeline';
    }
}

// Display scene graphs, boxes and text responses
function displayPipelineResults(data) {
    // 1. Show VLM Scene Graph JSON
    document.getElementById("scenegraph-placeholder").style.display = "none";
    const sgPre = document.getElementById("scenegraph-json");
    sgPre.style.display = "block";
    sgPre.innerText = JSON.stringify(data.scene_graph, null, 2);

    // 2. Render overlays onto canvas
    renderBoundingBoxes(data.scene_graph.elements);

    // 3. Render Socratic Feedback Bubble
    document.getElementById("feedback-placeholder").style.display = "none";
    document.getElementById("feedback-content").style.display = "block";

    document.getElementById("socratic-prompt-bubble").innerText = data.socratic_feedback;

    // 4. Fill misconception lists
    const errorList = document.getElementById("misconception-list");
    errorList.innerHTML = "";

    if (data.misconceptions_found.length === 0) {
        errorList.innerHTML = '<div class="placeholder-text" style="color: var(--color-emerald)">No misconceptions identified. Perfect!</div>';
    } else {
        data.misconceptions_found.forEach(m => {
            const badge = document.createElement("div");
            badge.className = "error-badge";
            badge.innerHTML = `
                <span class="code-tag">${m.id} [${m.category}]</span>
                <span>${m.error_description}</span>
            `;
            errorList.appendChild(badge);
        });
    }

    // Automatically trigger audio read-out if accessibility mode is turned on
    if (isAudioNarratorOn && data.accessibility_narration) {
        speakNarration(data.accessibility_narration);
    }
}

// Draw bounding boxes on canvas
function renderBoundingBoxes(elements) {
    const canvas = document.getElementById("overlay-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    elements.forEach(elem => {
        const bbox = elem.bbox; // [ymin, xmin, ymax, xmax] normalized
        if (!bbox) return;

        // Normalize scaling mapping
        const ymin = bbox[0] / 1000.0 * canvas.height;
        const xmin = bbox[1] / 1000.0 * canvas.width;
        const ymax = bbox[2] / 1000.0 * canvas.height;
        const xmax = bbox[3] / 1000.0 * canvas.width;

        const w = xmax - xmin;
        const h = ymax - ymin;

        // Color coding: red for vectors (forces/errors), green for standard elements
        let color = "#3b82f6"; // blue standard
        if (elem.grounding === "physics_force_vector" && elem.label.toLowerCase().includes("friction")) {
            color = "#ef4444"; // friction error red
        } else if (elem.grounding === "physics_force_vector") {
            color = "#10b981"; // gravity green
        } else if (elem.label.toLowerCase().includes("mitochondrion")) {
            color = "#ef4444"; // chloroplast error red
        } else if (elem.type === "vector") {
            color = "#ef4444";
        }

        // Draw bounding box box border
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(xmin, ymin, w, h);

        // Draw label tag backing
        ctx.fillStyle = color;
        ctx.font = "bold 11px 'Outfit'";
        const tagText = `${elem.label} (${Math.round(elem.confidence * 100)}%)`;
        const textWidth = ctx.measureText(tagText).width;

        ctx.fillRect(xmin, ymin - 16, textWidth + 10, 16);
        ctx.fillStyle = "#fff";
        ctx.fillText(tagText, xmin + 5, ymin - 4);
    });
}

/* ================= ACCESSIBILITY NARRATOR ================= */
function toggleAccessibilityMode() {
    isAudioNarratorOn = !isAudioNarratorOn;
    const btn = document.getElementById("audio-icon");
    if (isAudioNarratorOn) {
        btn.parentElement.classList.remove("outline");
        btn.parentElement.classList.add("btn-secondary");
        btn.className = "fa-solid fa-volume-high fa-bounce";
        
        // Speak active content if it exists
        if (parsedResponseData && parsedResponseData.accessibility_narration) {
            speakNarration(parsedResponseData.accessibility_narration);
        }
    } else {
        btn.parentElement.classList.remove("btn-secondary");
        btn.parentElement.classList.add("outline");
        btn.className = "fa-solid fa-volume-high";
        window.speechSynthesis.cancel();
    }
}

function speakNarration(text) {
    window.speechSynthesis.cancel(); // Stop current speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95; // Slightly slower for readability
    window.speechSynthesis.speak(utterance);
}

/* ================= TEACHER METRICS ================= */
async function refreshTeacherMetrics() {
    try {
        const response = await fetch("/api/classroom_metrics");
        if (!response.ok) return;

        const data = await response.json();

        // Update counts
        document.getElementById("metrics-class-size").innerText = data.heatmap.total_students_assessed;
        document.getElementById("metrics-learning-gain").innerText = data.learning_gains.Normalized_Gain;
        document.getElementById("metrics-active-errors").innerText = data.heatmap.attention_needed.length;

        // Render Heatmap chart
        renderHeatmapChart(data.heatmap.misconception_frequencies, data.heatmap.total_students_assessed);

        // Render Longitudinal nodes
        renderLongitudinalTrack(data.persistence_profile);

        // Render Taxonomy Collapsible Explorer
        const taxRes = await fetch("/api/misconception_taxonomy");
        if (taxRes.ok) {
            const taxData = await taxRes.json();
            renderTaxonomyExplorer(taxData);
        }

    } catch (err) {
        console.error("Failed to load analytics: ", err);
    }
}

function renderHeatmapChart(frequencies, total) {
    const chart = document.getElementById("bar-chart");
    chart.innerHTML = "";

    // Definition labels helper
    const labelsMap = {
        "PHYS-M1": "Normal Force Omission (Physics)",
        "PHYS-M2": "Friction Vector Sign Error (Physics)",
        "PHYS-M3": "Active Agent Bias (Physics)",
        "BIOL-M1": "Chloroplast/Mitochondria Organelle Confusion",
        "BIOL-M2": "Cell Wall Presence Confusion (Biology)",
        "CHEM-M1": "Linear Geometry Water Molecule Shape Error",
        "CHEM-M2": "Oxygen Valence Octet Incompleteness"
    };

    Object.keys(frequencies).forEach(mId => {
        const count = frequencies[mId];
        const percent = Math.round((count / total) * 100);
        const name = labelsMap[mId] || mId;

        const barRow = document.createElement("div");
        barRow.className = "bar-row";
        
        // Mark yellow/red if count is high (>40%)
        const fillClass = percent >= 40 ? "bar-fill warning" : "bar-fill";

        barRow.innerHTML = `
            <div class="bar-labels">
                <span>${name}</span>
                <span class="text-bold">${count}/${total} (${percent}%)</span>
            </div>
            <div class="bar-track">
                <div class="${fillClass}" style="width: ${percent}%"></div>
            </div>
        `;
        chart.appendChild(barRow);
    });
}

function renderLongitudinalTrack(profile) {
    const timeline = document.getElementById("longitudinal-timeline");
    timeline.innerHTML = "";

    const steps = [
        { task: "Task 1: Flat Surface", error: "PHYS-M1 (Normal Force Omitted)", resolved: false },
        { task: "Task 2: Mild Incline", error: "PHYS-M1 & PHYS-M2 (Friction Vector Wrong Direction)", resolved: false },
        { task: "Task 3: Steep Incline", error: "PHYS-M2 (Friction Wrong Direction), PHYS-M1 Resolved!", resolved: true },
        { task: "Task 4: Pulley System", error: "All Misconceptions Resolved", resolved: true }
    ];

    steps.forEach(step => {
        const point = document.createElement("div");
        point.className = "timeline-point";

        const bulletClass = step.resolved ? "timeline-bullet resolved" : "timeline-bullet";
        
        point.innerHTML = `
            <div class="${bulletClass}"></div>
            <div class="timeline-info">
                <span class="timeline-title">${step.task}</span>
                <span class="timeline-desc">${step.error}</span>
            </div>
        `;
        timeline.appendChild(point);
    });
}

function renderTaxonomyExplorer(taxonomy) {
    const container = document.getElementById("taxonomy-explorer");
    container.innerHTML = "";

    Object.keys(taxonomy).forEach(domainName => {
        const items = taxonomy[domainName];
        
        const domainSection = document.createElement("div");
        domainSection.style.marginBottom = "14px";
        domainSection.innerHTML = `<h4 style="text-transform: capitalize; margin-bottom: 8px; color: var(--color-primary);">${domainName} Error Library</h4>`;

        items.forEach(item => {
            const node = document.createElement("div");
            node.className = "taxonomy-node";
            node.innerHTML = `
                <div class="taxonomy-title" onclick="this.parentElement.classList.toggle('open')">
                    <span><i class="fa-solid fa-folder-open text-purple"></i> ${item.id}: ${item.category} (${item.concept})</span>
                    <i class="fa-solid fa-chevron-down toggle-arrow"></i>
                </div>
                <div class="taxonomy-details">
                    <p style="margin-bottom: 8px;"><strong>Diagnostic:</strong> ${item.description}</p>
                    <p style="margin-bottom: 8px;"><strong>Rubric Reference:</strong> AAAS Code: ${item.aaas_code} | MaLT Code: ${item.malt_code}</p>
                    <p style="color: var(--color-emerald)"><strong>Socratic Prompt Scaffold:</strong> "${item.socratic_prompt}"</p>
                </div>
            `;
            domainSection.appendChild(node);
        });

        container.appendChild(domainSection);
    });
}

/* ================= STATISTICAL RUN ================= */
async function triggerStatisticalRun() {
    const power = document.getElementById("slider-power").value;
    const alpha = document.getElementById("slider-alpha").value;
    const effect = document.getElementById("slider-effect").value;

    try {
        const res = await fetch(`/api/run_statistical_evaluation?power=${power}&alpha=${alpha}&effect_size=${effect}`);
        if (!res.ok) return;

        const data = await res.json();

        // Update displays
        document.getElementById("stats-n-value").innerText = data.required_sample_size_per_group;
        document.getElementById("stats-p-value").innerText = `p = ${data.welch_t_test.p_value}`;

        const verdict = document.getElementById("stats-verdict-text");
        if (data.welch_t_test.statistically_significant) {
            verdict.innerText = "Significant (p < 0.05)";
            verdict.className = "p-subtext text-green";
        } else {
            verdict.innerText = "Not Significant (p >= 0.05)";
            verdict.className = "p-subtext text-red";
        }

        document.getElementById("stat-mean-control").innerText = data.welch_t_test.mean_control;
        document.getElementById("stat-mean-treatment").innerText = data.welch_t_test.mean_treatment;
        document.getElementById("stat-effect-size").innerText = data.welch_t_test.cohens_d;
        document.getElementById("stat-t-value").innerText = data.welch_t_test.t_statistic;

    } catch (err) {
        console.error("Statistical run failed: ", err);
    }
}

function updateStatsParams() {
    const power = document.getElementById("slider-power").value;
    const alpha = document.getElementById("slider-alpha").value;
    const effect = document.getElementById("slider-effect").value;

    document.getElementById("val-power").innerText = Math.round(power * 100) + "%";
    document.getElementById("val-alpha").innerText = alpha;
    document.getElementById("val-effect").innerText = effect;
}
