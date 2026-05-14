/**
 * Shadow AI v2 — Background Service Worker
 * 
 * KEY ARCHITECTURE DECISION: All API calls route through HERE, not the content script.
 * This completely bypasses Gemini/ChatGPT's Content Security Policy because
 * service workers run in an isolated context outside the page's CSP rules.
 */

const SHADOW_API = "http://localhost:8000";

// ---------------------------------------------------------------------------
// 1. MESSAGE HANDLER — Proxy between content script and Python backend
// ---------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  // --- Full Analysis Request ---
  if (message.action === "analyze") {
    fetch(`${SHADOW_API}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: message.prompt, response: message.response }),
    })
      .then((res) => res.json())
      .then((data) => {
        // Update the icon badge with the score
        if (sender.tab) {
          updateBadge(data.overall_score, sender.tab.id);
        }
        sendResponse({ success: true, report: data });
      })
      .catch((err) => {
        console.warn("[Shadow AI] API error:", err.message);
        sendResponse({ success: false, error: err.message });
      });

    return true; // Keep channel open for async response
  }

  // --- Health Check ---
  if (message.action === "health_check") {
    fetch(`${SHADOW_API}/health`)
      .then((res) => res.json())
      .then((data) => sendResponse({ success: true, ...data }))
      .catch((err) => sendResponse({ success: false, error: err.message }));

    return true;
  }

  // --- Quick Check (lightweight) ---
  if (message.action === "quick_check") {
    fetch(`${SHADOW_API}/quick-check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: message.prompt, response: message.response }),
    })
      .then((res) => res.json())
      .then((data) => sendResponse({ success: true, report: data }))
      .catch((err) => sendResponse({ success: false, error: err.message }));

    return true;
  }
});

// ---------------------------------------------------------------------------
// 2. BADGE — Visual indicator on the extension icon
// ---------------------------------------------------------------------------
function updateBadge(score, tabId) {
  let color = "#228B22"; // Green
  let text = "";

  if (score > 60) {
    color = "#B41E1E";
    text = "!!";
  } else if (score > 35) {
    color = "#C86400";
    text = "?";
  } else if (score > 15) {
    color = "#B47800";
    text = "~";
  }

  chrome.action.setBadgeText({ text, tabId });
  chrome.action.setBadgeBackgroundColor({ color, tabId });
}

// ---------------------------------------------------------------------------
// 3. LIFECYCLE
// ---------------------------------------------------------------------------
chrome.runtime.onInstalled.addListener(() => {
  console.log("[Shadow AI v2] Extension installed. Waiting for analysis requests.");
});
