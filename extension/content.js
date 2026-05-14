/**
 * Shadow AI — Content Script
 * 
 * This script runs directly on gemini.google.com and chatgpt.com.
 * It watches for AI responses, grabs the text, sends it to our local
 * Python FastAPI server, and highlights hallucinated tokens in-place.
 */

const SHADOW_API = "http://localhost:8000";
const ANALYSIS_DEBOUNCE_MS = 2000; // Wait 2 seconds after the AI stops typing

let lastAnalyzedText = "";
let debounceTimer = null;
let isAnalyzing = false;

// ---------------------------------------------------------------------------
// 1. DETECT WHICH SITE WE ARE ON
// ---------------------------------------------------------------------------
function detectSite() {
  const host = window.location.hostname;
  if (host.includes("gemini.google.com")) return "gemini";
  if (host.includes("chatgpt.com")) return "chatgpt";
  return "unknown";
}

// ---------------------------------------------------------------------------
// 2. SCRAPE THE PROMPT (what the user typed)
// ---------------------------------------------------------------------------
function scrapeUserPrompt() {
  const site = detectSite();

  if (site === "gemini") {
    // Gemini: user messages are in elements with data-message-author-role="user"
    const userMsgs = document.querySelectorAll(
      '[data-message-author-role="user"], .user-query-text, .query-text'
    );
    if (userMsgs.length > 0) {
      return userMsgs[userMsgs.length - 1].innerText.trim();
    }
    // Fallback: grab the input textarea
    const textarea = document.querySelector('textarea, [contenteditable="true"]');
    return textarea ? textarea.innerText.trim() : "";
  }

  if (site === "chatgpt") {
    // ChatGPT: user messages have data-message-author-role="user"
    const userMsgs = document.querySelectorAll('[data-message-author-role="user"]');
    if (userMsgs.length > 0) {
      return userMsgs[userMsgs.length - 1].innerText.trim();
    }
    return "";
  }

  return "";
}

// ---------------------------------------------------------------------------
// 3. SCRAPE THE AI RESPONSE (what Gemini/ChatGPT wrote)
// ---------------------------------------------------------------------------
function scrapeAIResponse() {
  const site = detectSite();

  if (site === "gemini") {
    // Gemini: model responses are in elements with data-message-author-role="model"
    const modelMsgs = document.querySelectorAll(
      '[data-message-author-role="model"], .model-response-text, .response-text'
    );
    if (modelMsgs.length > 0) {
      return modelMsgs[modelMsgs.length - 1].innerText.trim();
    }
    // Fallback: grab the last message-content
    const responses = document.querySelectorAll('.message-content, .markdown');
    if (responses.length > 0) {
      return responses[responses.length - 1].innerText.trim();
    }
    return "";
  }

  if (site === "chatgpt") {
    // ChatGPT: assistant messages have data-message-author-role="assistant"
    const assistantMsgs = document.querySelectorAll('[data-message-author-role="assistant"]');
    if (assistantMsgs.length > 0) {
      return assistantMsgs[assistantMsgs.length - 1].innerText.trim();
    }
    return "";
  }

  return "";
}

// ---------------------------------------------------------------------------
// 4. GET THE RESPONSE ELEMENT (to inject highlights into)
// ---------------------------------------------------------------------------
function getResponseElement() {
  const site = detectSite();

  if (site === "gemini") {
    const modelMsgs = document.querySelectorAll(
      '[data-message-author-role="model"], .model-response-text, .response-text'
    );
    if (modelMsgs.length > 0) return modelMsgs[modelMsgs.length - 1];

    const responses = document.querySelectorAll('.message-content, .markdown');
    if (responses.length > 0) return responses[responses.length - 1];
    return null;
  }

  if (site === "chatgpt") {
    const assistantMsgs = document.querySelectorAll('[data-message-author-role="assistant"]');
    if (assistantMsgs.length > 0) return assistantMsgs[assistantMsgs.length - 1];
    return null;
  }

  return null;
}

// ---------------------------------------------------------------------------
// 5. CALL THE SHADOW AI API
// ---------------------------------------------------------------------------
async function callShadowAPI(prompt, response) {
  try {
    const res = await fetch(`${SHADOW_API}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, response }),
    });

    if (!res.ok) {
      console.warn("[Shadow AI] API returned status:", res.status);
      return null;
    }

    return await res.json();
  } catch (err) {
    console.warn("[Shadow AI] Could not reach local server:", err.message);
    return null;
  }
}

// ---------------------------------------------------------------------------
// 6. INJECT THE FLOATING BADGE (shows the overall score)
// ---------------------------------------------------------------------------
function injectBadge(report) {
  // Remove old badge if exists
  const old = document.getElementById("shadow-ai-badge");
  if (old) old.remove();

  const badge = document.createElement("div");
  badge.id = "shadow-ai-badge";

  let badgeClass = "shadow-badge-trusted";
  let emoji = "✅";
  if (report.overall_score > 60) {
    badgeClass = "shadow-badge-danger";
    emoji = "🚨";
  } else if (report.overall_score > 35) {
    badgeClass = "shadow-badge-warn";
    emoji = "⚠️";
  } else if (report.overall_score > 15) {
    badgeClass = "shadow-badge-caution";
    emoji = "🔶";
  }

  badge.className = `shadow-badge ${badgeClass}`;
  badge.innerHTML = `
    <div class="shadow-badge-header">
      <span class="shadow-badge-emoji">${emoji}</span>
      <span class="shadow-badge-title">Shadow AI</span>
      <button class="shadow-badge-close" id="shadow-badge-close">×</button>
    </div>
    <div class="shadow-badge-score">${report.overall_score}/100</div>
    <div class="shadow-badge-verdict">${report.verdict}</div>
    <div class="shadow-badge-details">
      ${report.high_conflict_tokens}/${report.total_tokens} tokens flagged
    </div>
  `;

  document.body.appendChild(badge);

  // Close button
  document.getElementById("shadow-badge-close").addEventListener("click", () => {
    badge.remove();
  });
}

// ---------------------------------------------------------------------------
// 7. HIGHLIGHT FLAGGED TOKENS IN THE RESPONSE
// ---------------------------------------------------------------------------
function highlightTokens(report) {
  const responseEl = getResponseElement();
  if (!responseEl) return;

  // Build a map of flagged words for quick lookup
  const flaggedWords = new Map();
  for (const token of report.tokens) {
    const word = token.token.trim();
    if (!word) continue;

    if (token.conflict_status === "HIGH_CONFLICT") {
      flaggedWords.set(word, {
        class: "shadow-highlight-danger",
        title: `Conflict: ${(token.conflict_score * 100).toFixed(0)}% | Shadow expected: "${token.shadow_preferred}"`,
      });
    } else if (token.conflict_status === "WARN" || token.is_diverging) {
      flaggedWords.set(word, {
        class: "shadow-highlight-warn",
        title: `Conflict: ${(token.conflict_score * 100).toFixed(0)}% | Shadow expected: "${token.shadow_preferred}"`,
      });
    }
  }

  if (flaggedWords.size === 0) return;

  // Walk through the text nodes and wrap flagged words in <span> elements
  const walker = document.createTreeWalker(responseEl, NodeFilter.SHOW_TEXT, null, false);
  const nodesToReplace = [];

  while (walker.nextNode()) {
    const textNode = walker.currentNode;
    const text = textNode.textContent;

    let hasMatch = false;
    for (const [word] of flaggedWords) {
      if (text.includes(word)) {
        hasMatch = true;
        break;
      }
    }

    if (hasMatch) {
      nodesToReplace.push(textNode);
    }
  }

  for (const textNode of nodesToReplace) {
    const fragment = document.createDocumentFragment();
    let remaining = textNode.textContent;

    while (remaining.length > 0) {
      let earliestMatch = null;
      let earliestIndex = Infinity;
      let matchedWord = null;

      // Find the earliest flagged word in the remaining text
      for (const [word, info] of flaggedWords) {
        const idx = remaining.indexOf(word);
        if (idx !== -1 && idx < earliestIndex) {
          earliestIndex = idx;
          earliestMatch = info;
          matchedWord = word;
        }
      }

      if (earliestMatch && matchedWord) {
        // Add text before the match
        if (earliestIndex > 0) {
          fragment.appendChild(document.createTextNode(remaining.substring(0, earliestIndex)));
        }

        // Add the highlighted word
        const span = document.createElement("span");
        span.className = earliestMatch.class;
        span.title = earliestMatch.title;
        span.textContent = matchedWord;
        fragment.appendChild(span);

        remaining = remaining.substring(earliestIndex + matchedWord.length);
      } else {
        // No more matches; add the rest as plain text
        fragment.appendChild(document.createTextNode(remaining));
        break;
      }
    }

    textNode.parentNode.replaceChild(fragment, textNode);
  }
}

// ---------------------------------------------------------------------------
// 8. MAIN ANALYSIS LOOP — triggered when the AI finishes responding
// ---------------------------------------------------------------------------
async function runAnalysis() {
  if (isAnalyzing) return;

  const prompt = scrapeUserPrompt();
  const response = scrapeAIResponse();

  if (!prompt || !response) {
    console.log("[Shadow AI] No prompt or response found yet.");
    return;
  }

  // Don't re-analyze the same text
  const fingerprint = prompt + "|" + response;
  if (fingerprint === lastAnalyzedText) return;

  isAnalyzing = true;
  console.log("[Shadow AI] Analyzing response...");
  console.log("[Shadow AI] Prompt:", prompt.substring(0, 80) + "...");
  console.log("[Shadow AI] Response:", response.substring(0, 80) + "...");

  const report = await callShadowAPI(prompt, response);

  if (report) {
    lastAnalyzedText = fingerprint;
    console.log(`[Shadow AI] Score: ${report.overall_score}/100 — ${report.verdict}`);

    injectBadge(report);
    highlightTokens(report);
  } else {
    console.warn("[Shadow AI] No report received. Is the server running?");
  }

  isAnalyzing = false;
}

// ---------------------------------------------------------------------------
// 9. MUTATION OBSERVER — watch for new AI responses appearing in the DOM
// ---------------------------------------------------------------------------
function startObserver() {
  const observer = new MutationObserver(() => {
    // Debounce: wait for the AI to finish typing
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      runAnalysis();
    }, ANALYSIS_DEBOUNCE_MS);
  });

  // Watch the entire body for new child elements (messages being added)
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  console.log(`[Shadow AI] Content script loaded on ${detectSite()}. Watching for responses...`);
}

// ---------------------------------------------------------------------------
// 10. LISTEN FOR MESSAGES FROM THE POPUP
// ---------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze_now") {
    runAnalysis().then(() => {
      sendResponse({ status: "done" });
    });
    return true; // Keep the message channel open for async response
  }

  if (message.action === "get_status") {
    sendResponse({
      site: detectSite(),
      hasPrompt: !!scrapeUserPrompt(),
      hasResponse: !!scrapeAIResponse(),
      lastAnalyzed: lastAnalyzedText ? true : false,
    });
  }
});

// ---------------------------------------------------------------------------
// INIT
// ---------------------------------------------------------------------------
startObserver();
