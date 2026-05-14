/**
 * Shadow AI — Popup Script
 * Checks the local server health and lets the user trigger an analysis.
 */

const SHADOW_API = "http://localhost:8000";

const serverStatus = document.getElementById("server-status");
const modelName = document.getElementById("model-name");
const siteName = document.getElementById("site-name");
const analyzeBtn = document.getElementById("analyze-btn");
const resultDiv = document.getElementById("analyze-result");

// Check if the local Shadow AI server is running
async function checkHealth() {
  try {
    const res = await fetch(`${SHADOW_API}/health`);
    const data = await res.json();

    serverStatus.textContent = "Online";
    serverStatus.className = "status-value status-online";
    modelName.textContent = data.model;
    analyzeBtn.disabled = false;
  } catch (err) {
    serverStatus.textContent = "Offline";
    serverStatus.className = "status-value status-offline";
    modelName.textContent = "Server not running";
    analyzeBtn.disabled = true;
  }
}

// Get site info from the content script
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

// Trigger analysis via the content script
analyzeBtn.addEventListener("click", async () => {
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  resultDiv.style.display = "none";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.tabs.sendMessage(tab.id, { action: "analyze_now" }, (response) => {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Current Response";

      if (chrome.runtime.lastError) {
        resultDiv.style.display = "block";
        resultDiv.className = "result-error";
        resultDiv.textContent = "Could not connect to the page. Make sure you are on Gemini or ChatGPT.";
        return;
      }

      resultDiv.style.display = "block";
      resultDiv.className = "result-success";
      resultDiv.textContent = "Analysis complete! Check the page for highlights.";
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
