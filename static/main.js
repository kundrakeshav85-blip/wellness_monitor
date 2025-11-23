// static/main.js - Corrected version with fixed button IDs

// Global state
let reportData = null;
let lastScore = 100;

// ---------- Fetching ----------
async function fetchReport() {
    try {
        const res = await fetch("/api/report");
        if (!res.ok) throw new Error("Network response not ok");
        const json = await res.json();
        reportData = json;
        updateDashboard(reportData);
    } catch (e) {
        console.error("fetchReport error:", e);
    }
}

// ---------- Dashboard update ----------
function updateDashboard(data) {
    // Elements
    const scoreValue = document.getElementById("scoreValue");
    const scoreCircle = document.getElementById("scoreCircle");
    const detectionCount = document.getElementById("detectionCount");
    const updateTime = document.getElementById("updateTime");
    const videoWrapper = document.querySelector(".video-wrapper");
    const getSuggestionsBtn = document.getElementById("getSuggestionsBtn");

    // Safety checks
    if (!scoreValue || !scoreCircle || !detectionCount || !updateTime) {
        console.warn("Some dashboard elements are missing in DOM");
    }

    // Score
    const score = data && typeof data.overall_wellness_score === "number" 
        ? data.overall_wellness_score 
        : 100;
    lastScore = score;

    // Animate score number and circle
    animateScore(scoreValue, score);
    updateDetectionCount(detectionCount, data?.total_detections || 0);
    updateTime.textContent = `Updated: ${new Date().toLocaleTimeString()}`;

    // Score circle: compute stroke-dashoffset
    try {
        const r = 90;
        const circumference = 2 * Math.PI * r;
        scoreCircle.style.strokeDasharray = `${circumference}`;
        const offset = circumference - (Math.max(0, Math.min(score, 100)) / 100) * circumference;
        scoreCircle.style.strokeDashoffset = offset;

        // Set color depending on score
        let scoreColor = "#4CAF50";
        if (score >= 80) scoreColor = "#4CAF50";
        else if (score >= 70) scoreColor = "#66BB6A";
        else if (score >= 60) scoreColor = "#FFC107";
        else if (score >= 50) scoreColor = "#FF9800";
        else scoreColor = "#E53935";
        
        scoreCircle.style.stroke = scoreColor;
    } catch (e) {
        console.warn("score circle update failed", e);
    }

    // ---------- Posture border logic ----------
    if (videoWrapper && data && typeof data.latest_posture === "string") {
        if (data.latest_posture === "Incorrect Posture") {
            videoWrapper.classList.add("bad");
        } else {
            videoWrapper.classList.remove("bad");
        }
    }

    // ---------- Score breakdown ----------
    let postureImpact = 0, emotionImpact = 0, checkinImpact = 0;
    
    if (data) {
        if (typeof data.posture_penalty === "number") postureImpact = data.posture_penalty;
        if (typeof data.emotion_penalty === "number") emotionImpact = data.emotion_penalty;
        if (typeof data.checkin_penalty === "number") checkinImpact = data.checkin_penalty;

        // If counts provided, compute proportionally
        if (data.posture_counts || data.emotion_counts) {
            try {
                let bad = 0, totalPosture = 0;
                if (data.posture_counts) {
                    Object.keys(data.posture_counts).forEach((k) => {
                        const c = Number(data.posture_counts[k]) || 0;
                        totalPosture += c;
                        if (/Incorrect|Bad/i.test(k)) bad += c;
                    });
                }

                let neg = 0, totalEm = 0;
                if (data.emotion_counts) {
                    Object.keys(data.emotion_counts).forEach((k) => {
                        const c = Number(data.emotion_counts[k]) || 0;
                        totalEm += c;
                        if (/Sad|Angry|Fearful/i.test(k)) neg += c;
                    });
                }

                if (totalPosture > 0) postureImpact = Math.round((bad / totalPosture) * 30);
                if (totalEm > 0) emotionImpact = Math.round((neg / totalEm) * 30);

                const ded = Math.max(0, 100 - score);
                if (!postureImpact && !emotionImpact) {
                    postureImpact = Math.round(ded * 0.4);
                    emotionImpact = Math.round(ded * 0.3);
                }
                checkinImpact = ded - postureImpact - emotionImpact;
            } catch (e) {
                console.warn("breakdown calc error", e);
            }
        }
    }

    if (!postureImpact && !emotionImpact && !checkinImpact) {
        const ded = 100 - score;
        postureImpact = Math.round(ded * 0.4);
        emotionImpact = Math.round(ded * 0.3);
        checkinImpact = ded - postureImpact - emotionImpact;
    }

    // Clamp >= 0
    postureImpact = Math.max(0, postureImpact);
    emotionImpact = Math.max(0, emotionImpact);
    checkinImpact = Math.max(0, checkinImpact);

    // Update DOM
    const postureEl = document.getElementById("postureImpact");
    const emotionEl = document.getElementById("emotionImpact");
    
    if (postureEl) postureEl.textContent = `-${postureImpact}`;
    if (emotionEl) emotionEl.textContent = `-${emotionImpact}`;

    // ---------- Get Suggestions Button - ALWAYS shows suggestions ----------
    if (getSuggestionsBtn) {
        let buttonText = "Get Personalized Suggestions";
        let buttonIcon = `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor">
            <path d="M10 2L2 7v6c0 5.5 3.84 7.66 8 8 4.16-.34 8-2.5 8-8V7l-8-5z" stroke-width="2"/>
            <circle cx="10" cy="11" r="1.5" fill="currentColor"/>
        </svg>`;

        if (score >= 80) {
            buttonText = "View Wellness Tips";
        } else if (score >= 60) {
            buttonText = "Get Improvement Suggestions";
        } else {
            buttonText = "Get Urgent Recommendations";
        }

        getSuggestionsBtn.innerHTML = `${buttonIcon} ${buttonText}`;
        getSuggestionsBtn.style.background = "linear-gradient(135deg, #4CAF50, #66BB6A)";
        getSuggestionsBtn.onclick = toggleSuggestions;
    }
}

// ---------- Helpers ----------
function updateDetectionCount(el, count) {
    if (!el) return;
    el.textContent = count;
}

function animateScore(element, targetScore) {
    if (!element) return;
    const current = parseInt(element.textContent) || 100;
    const steps = 25;
    const duration = 700;
    const increment = (targetScore - current) / steps;
    let value = current;
    let step = 0;

    const id = setInterval(() => {
        step++;
        value += increment;
        element.textContent = Math.round(value);
        if (step >= steps) {
            element.textContent = targetScore;
            clearInterval(id);
        }
    }, duration / steps);
}

// ---------- Suggestions panel toggle ----------
function toggleSuggestions() {
    const panel = document.getElementById("suggestionsPanel");
    const list = document.getElementById("suggestionsList");
    
    if (!panel || !list) return;

    if (panel.style.display === "none" || !panel.style.display) {
        panel.style.display = "block";
        
        // Compose suggestions: score-based + server suggestions
        const scoreRecs = generateScoreBasedRecommendations(lastScore);
        const server = reportData?.suggestions || [];
        const normalized = server.map((s) => 
            typeof s === "string" ? { title: s, message: s } : s
        );
        const all = [...scoreRecs, ...normalized];

        // Remove duplicates by title
        const unique = all.filter((s, i, arr) => 
            i === arr.findIndex((t) => t.title === s.title)
        );

        list.innerHTML = unique.map((s, idx) => `
            <div class="suggestion-item" style="animation-delay: ${idx * 0.1}s">
                <h4>${s.title || 'Wellness Tip'}</h4>
                <p>${s.message || s.title}</p>
            </div>
        `).join("");
    } else {
        panel.style.display = "none";
    }
}

function closeSuggestionsPanel() {
    const panel = document.getElementById("suggestionsPanel");
    if (panel) panel.style.display = "none";
}

// ---------- Generate score-based recommendations ----------
function generateScoreBasedRecommendations(score) {
    const recs = [];
    
    if (score >= 80) {
        recs.push({
            title: "Excellent Work! 🎉",
            message: "You're maintaining great wellness habits. Keep up the fantastic posture and positive mindset!"
        });
        recs.push({
            title: "Stay Consistent",
            message: "Continue your current routine with regular breaks and mindful breathing exercises."
        });
    } else if (score >= 60) {
        recs.push({
            title: "Posture Check ⚠️",
            message: "Try the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds, and adjust your posture."
        });
        recs.push({
            title: "Stretch Breaks",
            message: "Take a 5-minute stretch break every hour. Focus on neck, shoulders, and back stretches."
        });
        recs.push({
            title: "Ergonomic Setup",
            message: "Ensure your screen is at eye level and your chair supports proper lumbar alignment."
        });
    } else {
        recs.push({
            title: "Urgent: Posture Correction 🚨",
            message: "Your posture needs immediate attention. Stand up, stretch, and reset your workspace ergonomics now."
        });
        recs.push({
            title: "Take a Break",
            message: "Step away from your desk for 10 minutes. Walk around, do some light exercises, and breathe deeply."
        });
        recs.push({
            title: "Stress Management",
            message: "Your wellness score indicates high stress. Practice deep breathing: 4 counts in, hold for 4, exhale for 6."
        });
        recs.push({
            title: "Hydration Check",
            message: "Drink a glass of water and consider if you need a longer break or to adjust your environment."
        });
    }

    return recs;
}

// ---------- Modal Controls ----------
function openModal() {
    const modal = document.getElementById("questionnaireModal");
    if (modal) {
        modal.style.display = "block";
        document.body.style.overflow = "hidden"; // Prevent background scrolling
    }
}

function closeModal() {
    const modal = document.getElementById("questionnaireModal");
    if (modal) {
        modal.style.display = "none";
        document.body.style.overflow = "auto";
    }
}

// ---------- Questionnaire Form Submit ----------
async function handleQuestionnaireSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const data = {
        stress_level: parseInt(formData.get("stress_level")),
        sleep_hours: parseFloat(formData.get("sleep_hours")),
        anxious: formData.get("anxious") === "yes",
        took_breaks: formData.get("took_breaks") === "yes",
        motivation: parseInt(formData.get("motivation"))
    };

    try {
        const response = await fetch("/api/questionnaire", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            closeModal();
            // Show success message
            alert("✅ Daily check-in submitted successfully!");
            // Refresh report data
            fetchReport();
        } else {
            alert("❌ Failed to submit check-in. Please try again.");
        }
    } catch (error) {
        console.error("Questionnaire submit error:", error);
        alert("❌ Error submitting check-in. Please try again.");
    }
}

// ---------- Slider Updates ----------
function updateSliderValue(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const valueDisplay = document.getElementById(valueId);
    
    if (slider && valueDisplay) {
        slider.addEventListener("input", (e) => {
            valueDisplay.textContent = e.target.value;
        });
    }
}

// ---------- Initialize on DOM Load ----------
document.addEventListener("DOMContentLoaded", () => {
    // Fetch initial report
    fetchReport();
    
    // Set up periodic refresh (every 10 seconds)
    setInterval(fetchReport, 10000);

    // ---------- CORRECTED: Daily Check-In button in HEADER ----------
    const checkInBtn = document.getElementById("checkInBtn");
    if (checkInBtn) {
        checkInBtn.addEventListener("click", openModal);
    }

    // Close modal button
    const closeModalBtn = document.getElementById("closeModal");
    if (closeModalBtn) {
        closeModalBtn.addEventListener("click", closeModal);
    }

    // Close suggestions panel button
    const closeSuggestionsBtn = document.getElementById("closeSuggestions");
    if (closeSuggestionsBtn) {
        closeSuggestionsBtn.addEventListener("click", closeSuggestionsPanel);
    }

    // Questionnaire form submit
    const questionnaireForm = document.getElementById("questionnaireForm");
    if (questionnaireForm) {
        questionnaireForm.addEventListener("submit", handleQuestionnaireSubmit);
    }

    // Slider value updates
    updateSliderValue("stressLevel", "stressValue");
    updateSliderValue("motivation", "motivationValue");

    // Close modal when clicking outside
    const modal = document.getElementById("questionnaireModal");
    if (modal) {
        window.addEventListener("click", (event) => {
            if (event.target === modal) {
                closeModal();
            }
        });
    }

    console.log("✅ Dashboard initialized successfully");
});
