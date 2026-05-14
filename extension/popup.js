/**
 * Shadow AI v2 — Popup Script
 * Uses background worker for health checks (CSP safe).
 */

const serverStatus = document.getElementById("server-status");
const modelName = document.getElementById("model-name");
const siteName = document.getElementById("site-name");
const versionEl = document.getElementById("version-info");
const analyzeBtn = document.getElementById("analyze-btn");
const resultDiv = document.getElementById("analyze-result");

// Health check via background worker (no direct fetch needed)
async function checkHealth() {
  chrome.runtime.sendMessage({ action: "health_check" }, (response) => {
    if (chrome.runtime.lastError || !response || !response.success) {
      serverStatus.textContent = "Offline";
      serverStatus.className = "status-value status-offline";
      modelName.textContent = "Server not running";
      if (versionEl) versionEl.textContent = "—";
      analyzeBtn.disabled = true;
      return;
    }

    serverStatus.textContent = "Online";
    serverStatus.className = "status-value status-online";
    modelName.textContent = response.model || "Unknown";
    if (versionEl) versionEl.textContent = response.version || "—";
    analyzeBtn.disabled = false;
  });
}

// Detect active site
async function getSiteStatus() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (tab.url.includes("gemini.google.com")) {
      siteName.textContent = "Gemini";
    } else if (tab.url.includes("chatgpt.com")) {
      siteName.textContent = "ChatGPT";
    } else {
      siteName.textContent = "Not on a supported site";
      analyzeBtn.disabled = true;
    }
  } catch (err) {
    siteName.textContent = "Unknown";
  }
}

// Trigger analysis
analyzeBtn.addEventListener("click", async () => {
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "🧠 Analyzing...";
  resultDiv.style.display = "none";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.tabs.sendMessage(tab.id, { action: "analyze_now" }, (response) => {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Current Response";

      if (chrome.runtime.lastError) {
        resultDiv.style.display = "block";
        resultDiv.className = "result-error";
        resultDiv.textContent = "Could not connect. Make sure you are on Gemini or ChatGPT.";
        return;
      }

      resultDiv.style.display = "block";
      resultDiv.className = "result-success";
      resultDiv.textContent = "Analysis triggered! Check the page for results.";
    });
  } catch (err) {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Current Response";
    resultDiv.style.display = "block";
    resultDiv.className = "result-error";
    resultDiv.textContent = "Error: " + err.message;
  }
});

// Init
checkHealth();
getSiteStatus();
