// Base API URL configuration (persisted in localStorage)
let API_BASE = localStorage.getItem("perception_api_base") || "http://127.0.0.1:5000";

// Dashboard States
let isRunning = false;
let telemetryInterval = null;

// UI DOM References
const btnToggleStream = document.getElementById("btn-toggle-stream");
const statusBadge = document.getElementById("status-badge");
const sourceSelect = document.getElementById("source-select");
const customSourceGroup = document.getElementById("custom-source-group");
const customSourcePath = document.getElementById("custom-source-path");
const depthToggle = document.getElementById("depth-toggle");
const videoPlaceholder = document.getElementById("video-placeholder");
const hudStreamImg = document.getElementById("hud-stream-img");
const alertsList = document.getElementById("alerts-list");

// Telemetry indicators
const valAction = document.getElementById("val-action");
const valOffset = document.getElementById("val-offset");
const valCurvature = document.getElementById("val-curvature");

// Radar Canvas Config
const canvas = document.getElementById("radar-canvas");
const ctx = canvas.getContext("2d");

// ================== INTERACTIVE MOUSE BORDER GLOWS (aalind.org style) ==================
function initBorderGlows() {
    const cards = document.querySelectorAll(".border-glow-card");
    
    document.addEventListener("mousemove", (e) => {
        cards.forEach((card) => {
            const rect = card.getBoundingClientRect();
            
            // Calculate center of the card
            const cardX = rect.left + rect.width / 2;
            const cardY = rect.top + rect.height / 2;
            
            // Calculate distance from mouse to card center
            const dx = e.clientX - cardX;
            const dy = e.clientY - cardY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Determine max distance for proximity (diagonal length of viewport)
            const maxDistance = Math.sqrt(window.innerWidth ** 2 + window.innerHeight ** 2) / 3;
            
            // Proximity: 100 when mouse is directly on card, scale down to 0 far away
            let proximity = 100 * (1 - distance / maxDistance);
            proximity = Math.max(0, Math.min(100, proximity));
            
            // If hovering on card, set proximity to 100
            const mouseOnCard = e.clientX >= rect.left && e.clientX <= rect.right &&
                                e.clientY >= rect.top && e.clientY <= rect.bottom;
            if (mouseOnCard) {
                proximity = 100;
            }
            
            // Calculate angle of cursor relative to center
            let angle = Math.atan2(dy, dx) * (180 / Math.PI);
            angle = (angle + 360) % 360; // normalize to 0-360
            
            card.style.setProperty("--edge-proximity", proximity);
            card.style.setProperty("--cursor-angle", `${angle}deg`);
        });
    });
}

// ================== UI CONTROL PANEL HANDLERS ==================
sourceSelect.addEventListener("change", () => {
    if (sourceSelect.value === "custom") {
        customSourceGroup.classList.remove("hidden");
    } else {
        customSourceGroup.classList.add("hidden");
    }
});

btnToggleStream.addEventListener("click", () => {
    if (!isRunning) {
        startPipeline();
    } else {
        stopPipeline();
    }
});

function startPipeline() {
    let videoSource = sourceSelect.value;
    if (videoSource === "custom") {
        videoSource = customSourcePath.value;
    } else if (videoSource === "webcam") {
        videoSource = 0;
    } else {
        videoSource = "synthetic";
    }

    const payload = {
        command: "start",
        video_source: videoSource,
        use_dl_depth: depthToggle.checked
    };

    fetch(`${API_BASE}/api/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "started" || data.status === "running") {
            isRunning = true;
            btnToggleStream.textContent = "STOP RUN";
            btnToggleStream.className = "btn btn-active";
            
            statusBadge.textContent = "ACTIVE";
            statusBadge.className = "badge badge-active";
            
            // Display stream video source
            videoPlaceholder.classList.add("hidden");
            // Cache-buster parameter to force reload the MJPEG stream
            hudStreamImg.src = `${API_BASE}/video_feed?t=${Date.now()}`;
            hudStreamImg.classList.remove("hidden");
            
            // Start polling telemetry data (every 80ms)
            telemetryInterval = setInterval(pollTelemetry, 80);
        }
    })
    .catch(err => console.error("Error starting pipeline:", err));
}

function stopPipeline() {
    fetch(`${API_BASE}/api/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: "stop" })
    })
    .then(res => res.json())
    .then(data => {
        isRunning = false;
        btnToggleStream.textContent = "START RUN";
        btnToggleStream.className = "btn btn-primary";
        
        statusBadge.textContent = "STOPPED";
        statusBadge.className = "badge badge-inactive";
        
        // Hide stream image
        hudStreamImg.src = "";
        hudStreamImg.classList.add("hidden");
        videoPlaceholder.classList.remove("hidden");
        
        // Clear telemetry polling
        if (telemetryInterval) {
            clearInterval(telemetryInterval);
            telemetryInterval = null;
        }
        
        resetTelemetryUI();
    })
    .catch(err => console.error("Error stopping pipeline:", err));
}

function resetTelemetryUI() {
    valAction.textContent = "MOVE";
    valAction.className = "telemetry-number";
    valOffset.textContent = "0.00 m";
    valCurvature.textContent = "0.0 m";
    
    // Clear ledger
    alertsList.innerHTML = `<div class="alert-item alert-clear">Roadway Clear — No active warnings</div>`;
    
    // Clear radar
    drawRadar(0.0, 0.0, []);
}

// ================== TELEMETRY POLLING ==================
function pollTelemetry() {
    fetch(`${API_BASE}/api/telemetry`)
    .then(res => res.json())
    .then(data => {
        updateTelemetryUI(data);
        drawRadar(data.lane_offset, data.lane_curvature, data.objects);
    })
    .catch(err => console.error("Error fetching telemetry:", err));
}

function updateTelemetryUI(data) {
    const action = data.action || "MOVE";
    const offset = data.lane_offset || 0.0;
    const curvature = data.lane_curvature || 0.0;
    const warnings = data.warnings || [];
    
    // Update Action HUD State class
    valAction.textContent = action;
    if (action === "BRAKE") {
        document.body.className = "brake-ttc";
        valAction.className = "telemetry-number brake-ttc";
    } else if (action === "WARN") {
        document.body.className = "warn-ttc";
        valAction.className = "telemetry-number warn-ttc";
    } else {
        document.body.className = "";
        valAction.className = "telemetry-number";
    }
    
    // Offset formatting (+ sign for right drift, - for left drift)
    const offsetSign = offset >= 0 ? "+" : "";
    valOffset.textContent = `${offsetSign}${offset.toFixed(2)} m`;
    
    // Curvature formatting (If very large, road is straight)
    if (curvature > 3000) {
        valCurvature.textContent = "Straight";
    } else {
        valCurvature.textContent = `${Math.round(curvature)} m`;
    }
    
    // Update Threat Ledger List
    alertsList.innerHTML = "";
    if (warnings.length === 0) {
        alertsList.innerHTML = `<div class="alert-item alert-clear">Roadway Clear — No active warnings</div>`;
    } else {
        warnings.forEach(warn => {
            const isCritical = warn.includes("CRITICAL") || warn.includes("BRAKE");
            const alertClass = isCritical ? "alert-critical" : "alert-warning";
            const alertIcon = isCritical ? "🚨" : "⚠️";
            
            const alertDiv = document.createElement("div");
            alertDiv.className = `alert-item ${alertClass}`;
            alertDiv.innerHTML = `<span>${alertIcon}</span> <span>${warn}</span>`;
            alertsList.appendChild(alertDiv);
        });
    }
}

// ================== 2D TOP-DOWN RADAR DRAWING SOLVER ==================
function drawRadar(laneOffset, laneCurvature, objects) {
    const w = canvas.width;
    const h = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = "#FAFAF8"; // Matches card bg
    ctx.fillRect(0, 0, w, h);
    
    // Draw Radar Range Circles
    ctx.strokeStyle = "rgba(0, 0, 0, 0.05)";
    ctx.lineWidth = 1;
    for (let r = 50; r <= 300; r += 70) {
        ctx.beginPath();
        ctx.arc(w / 2, h - 40, r, Math.PI, 2 * Math.PI);
        ctx.stroke();
        
        // Add range labels
        const meters = Math.round(r / 5.6); // 5.6px per meter scale
        ctx.fillStyle = "rgba(0, 0, 0, 0.25)";
        ctx.font = "8px Inter";
        ctx.fillText(`${meters}m`, w / 2 + r - 16, h - 43);
    }
    
    // Horizontal center coordinates
    const cx = w / 2;
    const cy = h - 40;
    
    // Scaling metrics (pixels per meter)
    const scaleX = w / 30; // 30 meters horizontal range (-15m to +15m)
    const scaleY = 280 / 50; // 50 meters vertical range (mapped to 280px)
    
    // Draw Lane Boundaries (curved lane simulation)
    ctx.strokeStyle = "#8B6914"; // Gold lane boundary
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    // Draw left and right lane boundaries
    const laneWidth = 3.7; // standard lane width in meters
    
    // Curve scaling based on curvature
    // 1 / R is the curvature. We integrate this to get horizontal shift
    let curvatureFactor = 0;
    if (laneCurvature > 50 && laneCurvature < 3000) {
        curvatureFactor = 300.0 / laneCurvature; // scale offset coefficient
    }
    
    // Draw Left Lane
    for (let z = 0; z <= 50; z += 2) {
        const offsetLeft = -(laneWidth / 2) - laneOffset;
        // Curve equation: x_shift = z_dist^2 * c
        const xShift = (z * z) * 0.004 * curvatureFactor;
        
        const px = cx + (offsetLeft + xShift) * scaleX;
        const py = cy - z * scaleY;
        
        if (z === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
    
    // Draw Right Lane
    ctx.strokeStyle = "rgba(0, 0, 0, 0.15)"; // Soft grey right boundary
    ctx.beginPath();
    for (let z = 0; z <= 50; z += 2) {
        const offsetRight = (laneWidth / 2) - laneOffset;
        const xShift = (z * z) * 0.004 * curvatureFactor;
        
        const px = cx + (offsetRight + xShift) * scaleX;
        const py = cy - z * scaleY;
        
        if (z === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
    
    // Draw Dashed center dividing line
    ctx.strokeStyle = "rgba(0, 0, 0, 0.1)";
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    for (let z = 0; z <= 50; z += 2) {
        const xShift = (z * z) * 0.004 * curvatureFactor;
        const px = cx + (-laneOffset + xShift) * scaleX;
        const py = cy - z * scaleY;
        
        if (z === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash
    
    // Draw Ego Vehicle (our car at the bottom)
    ctx.fillStyle = "#1A1A1A"; // charcoal black
    ctx.fillRect(cx - 8, cy - 10, 16, 26);
    // Draw simple wheels
    ctx.fillStyle = "#8B6914";
    ctx.fillRect(cx - 10, cy - 6, 2, 6);
    ctx.fillRect(cx + 8, cy - 6, 2, 6);
    ctx.fillRect(cx - 10, cy + 10, 2, 6);
    ctx.fillRect(cx + 8, cy + 10, 2, 6);
    
    // Draw Tracked Obstacles
    objects.forEach(obj => {
        const ox = cx + obj.lateral_pos * scaleX;
        const oy = cy - obj.distance * scaleY;
        
        // Color based on safety risk (TTC warning)
        let color = "#2d8a4e"; // safe green
        if (obj.ttc !== null) {
            if (obj.ttc < 1.2) color = "#d93838"; // brake red
            else if (obj.ttc < 2.5) color = "#b58209"; // warn gold
        } else if (obj.distance < 8.0) {
            color = "#d93838";
        }
        
        ctx.fillStyle = color;
        
        if (obj.class_name === "person") {
            // Draw pedestrian as a dot
            ctx.beginPath();
            ctx.arc(ox, oy, 6, 0, 2 * Math.PI);
            ctx.fill();
            
            // Pedestrian outline
            ctx.strokeStyle = "#FAFAF8";
            ctx.lineWidth = 1;
            ctx.stroke();
        } else {
            // Draw vehicle as a box
            ctx.fillRect(ox - 10, oy - 14, 20, 24);
            
            // Draw windshield/headlights indicators
            ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
            ctx.fillRect(ox - 7, oy - 10, 14, 4);
        }
        
        // Add ID / Distance Text labels
        ctx.fillStyle = "#1A1A1A";
        ctx.font = "bold 9px Inter";
        const labelText = `${obj.class_name.toUpperCase()} ${obj.id} (${obj.distance.toFixed(1)}m)`;
        ctx.fillText(labelText, ox + 14, oy + 4);
    });
}

// ================== INIT ==================
document.addEventListener("DOMContentLoaded", () => {
    initBorderGlows();
    
    // Bind API URL Input & Sync
    const apiUrlInput = document.getElementById("api-url-input");
    if (apiUrlInput) {
        apiUrlInput.value = API_BASE;
        apiUrlInput.addEventListener("input", () => {
            API_BASE = apiUrlInput.value.trim();
            localStorage.setItem("perception_api_base", API_BASE);
        });
    }
    
    // Check initial backend status
    fetch(`${API_BASE}/api/status`)
    .then(res => res.json())
    .then(data => {
        if (data.is_running) {
            isRunning = true;
            btnToggleStream.textContent = "STOP RUN";
            btnToggleStream.className = "btn btn-active";
            statusBadge.textContent = "ACTIVE";
            statusBadge.className = "badge badge-active";
            
            videoPlaceholder.classList.add("hidden");
            hudStreamImg.src = `${API_BASE}/video_feed?t=${Date.now()}`;
            hudStreamImg.classList.remove("hidden");
            
            telemetryInterval = setInterval(pollTelemetry, 80);
        } else {
            resetTelemetryUI();
        }
    })
    .catch(err => {
        console.error("Backend not active:", err);
        resetTelemetryUI();
    });
});
