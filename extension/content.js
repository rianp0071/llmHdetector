/**
 * Shadow AI v2 — Content Script
 * 
 * Runs on gemini.google.com and chatgpt.com.
 * Watches for AI responses, sends them through the background worker
 * (bypassing CSP), and highlights hallucinated tokens with reprompt suggestions.
 * 
 * v2: Pulsing brain indicator, debounced analysis, reprompt suggestion panel.
 */

const DEBOUNCE_MS = 1500; // Wait 1.5s after AI stops typing before analyzing

let lastAnalyzedFingerprint = "";
let debounceTimer = null;
let isAnalyzing = false;

// ---------------------------------------------------------------------------
// 1. SITE DETECTION
// ---------------------------------------------------------------------------
function detectSite() {
  const host = window.location.hostname;
  if (host.includes("gemini.google.com")) return "gemini";
  if (host.includes("chatgpt.com")) return "chatgpt";
  return "unknown";
}

// ---------------------------------------------------------------------------
// 2. SCRAPE USER PROMPT
// ---------------------------------------------------------------------------
function scrapeUserPrompt() {
  const site = detectSite();

  if (site === "gemini") {
    const userMsgs = document.querySelectorAll(
      '[data-message-author-role="user"], .user-query-text, .query-text'
    );
    if (userMsgs.length > 0) return userMsgs[userMsgs.length - 1].innerText.trim();
    const textarea = document.querySelector('textarea, [contenteditable="true"]');
    return textarea ? textarea.innerText.trim() : "";
  }

  if (site === "chatgpt") {
    const userMsgs = document.querySelectorAll('[data-message-author-role="user"]');
    if (userMsgs.length > 0) return userMsgs[userMsgs.length - 1].innerText.trim();
    return "";
  }

  return "";
}

// ---------------------------------------------------------------------------
// 3. SCRAPE AI RESPONSE
// ---------------------------------------------------------------------------
function scrapeAIResponse() {
  const site = detectSite();

  if (site === "gemini") {
    const modelMsgs = document.querySelectorAll(
      '[data-message-author-role="model"], .model-response-text, .response-text'
    );
    if (modelMsgs.length > 0) return modelMsgs[modelMsgs.length - 1].innerText.trim();
    const responses = document.querySelectorAll('.message-content, .markdown');
    if (responses.length > 0) return responses[responses.length - 1].innerText.trim();
    return "";
  }

  if (site === "chatgpt") {
    const assistantMsgs = document.querySelectorAll('[data-message-author-role="assistant"]');
    if (assistantMsgs.length > 0) return assistantMsgs[assistantMsgs.length - 1].innerText.trim();
    return "";
  }

  return "";
}

// ---------------------------------------------------------------------------
// 4. GET THE RESPONSE DOM ELEMENT (for injecting highlights)
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
// 5. PULSING BRAIN INDICATOR — subtle, non-annoying status
// ---------------------------------------------------------------------------
function injectIndicator() {
  if (document.getElementById("shadow-indicator")) return;

  const indicator = document.createElement("div");
  indicator.id = "shadow-indicator";
  indicator.className = "shadow-indicator shadow-indicator-idle";
  indicator.innerHTML = `<span class="shadow-indicator-icon">🧠</span>`;
  indicator.title = "Shadow AI — Idle";
  document.body.appendChild(indicator);
}

function setIndicatorState(state) {
  const el = document.getElementById("shadow-indicator");
  if (!el) return;

  // Remove all state classes
  el.classList.remove(
    "shadow-indicator-idle",
    "shadow-indicator-thinking",
    "shadow-indicator-trusted",
    "shadow-indicator-warn",
    "shadow-indicator-danger"
  );

  switch (state) {
    case "thinking":
      el.classList.add("shadow-indicator-thinking");
      el.title = "Shadow AI — Analyzing...";
      break;
    case "trusted":
      el.classList.add("shadow-indicator-trusted");
      el.title = "Shadow AI — Response looks trusted";
      break;
    case "warn":
      el.classList.add("shadow-indicator-warn");
      el.title = "Shadow AI — Potential drift detected";
      break;
    case "danger":
      el.classList.add("shadow-indicator-danger");
      el.title = "Shadow AI — Hallucination detected!";
      break;
    default:
      el.classList.add("shadow-indicator-idle");
      el.title = "Shadow AI — Idle";
  }
}

// ---------------------------------------------------------------------------
// 6. SCORE BADGE (appears after analysis)
// ---------------------------------------------------------------------------
function injectBadge(report) {
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

  const header = document.createElement("div");
  header.className = "shadow-badge-header";
  
  const emojiSpan = document.createElement("span");
  emojiSpan.className = "shadow-badge-emoji";
  emojiSpan.textContent = emoji;
  
  const titleSpan = document.createElement("span");
  titleSpan.className = "shadow-badge-title";
  titleSpan.textContent = "Shadow AI";
  
  const closeBtn = document.createElement("button");
  closeBtn.className = "shadow-badge-close";
  closeBtn.id = "shadow-badge-close";
  closeBtn.textContent = "×";
  closeBtn.addEventListener("click", () => badge.remove());
  
  header.appendChild(emojiSpan);
  header.appendChild(titleSpan);
  header.appendChild(closeBtn);
  badge.appendChild(header);

  const scoreDiv = document.createElement("div");
  scoreDiv.className = "shadow-badge-score";
  scoreDiv.textContent = `${report.overall_score}/100`;
  badge.appendChild(scoreDiv);

  const verdictDiv = document.createElement("div");
  verdictDiv.className = "shadow-badge-verdict";
  verdictDiv.textContent = report.verdict;
  badge.appendChild(verdictDiv);

  const detailsDiv = document.createElement("div");
  detailsDiv.className = "shadow-badge-details";
  detailsDiv.textContent = `${report.high_conflict_tokens}/${report.total_tokens} tokens flagged`;
  badge.appendChild(detailsDiv);

  if (report.suggestions && report.suggestions.length > 0) {
    const suggContainer = document.createElement("div");
    suggContainer.className = "shadow-badge-suggestions";
    
    for (const s of report.suggestions) {
      const sDiv = document.createElement("div");
      sDiv.className = "shadow-suggestion";
      
      const sType = document.createElement("div");
      sType.className = "shadow-suggestion-type";
      sType.textContent = s.type;
      
      const sMsg = document.createElement("div");
      sMsg.className = "shadow-suggestion-msg";
      sMsg.textContent = s.message;
      
      const sBtn = document.createElement("button");
      sBtn.className = "shadow-suggestion-copy";
      sBtn.textContent = "📋 Copy reprompt";
      sBtn.dataset.reprompt = s.reprompt;
      
      sBtn.addEventListener("click", () => {
        const text = sBtn.dataset.reprompt;
        navigator.clipboard.writeText(text).then(() => {
          sBtn.textContent = "✅ Copied!";
          setTimeout(() => { sBtn.textContent = "📋 Copy reprompt"; }, 2000);
        });
      });
      
      sDiv.appendChild(sType);
      sDiv.appendChild(sMsg);
      sDiv.appendChild(sBtn);
      suggContainer.appendChild(sDiv);
    }
    badge.appendChild(suggContainer);
  }

  document.body.appendChild(badge);

  // Move brain indicator above the badge
  const indicator = document.getElementById("shadow-indicator");
  if (indicator) {
    const badgeHeight = badge.offsetHeight;
    indicator.style.bottom = (badgeHeight + 30) + "px";
  }
}

// ---------------------------------------------------------------------------
// 7. HIGHLIGHT FLAGGED TOKENS
// ---------------------------------------------------------------------------
function highlightTokens(report) {
  const responseEl = getResponseElement();
  if (!responseEl) return;

  const flaggedWords = new Map();
  for (const token of report.tokens) {
    const word = token.token.trim();
    if (!word || word.length < 2) continue;

    if (token.conflict_status === "HIGH_CONFLICT") {
      flaggedWords.set(word, {
        class: "shadow-highlight-danger",
        title: `⚠️ Conflict: ${(token.conflict_score * 100).toFixed(0)}% | Z-Score: ${token.entropy_z_score} | Shadow expected: "${token.shadow_preferred}"`,
      });
    } else if (token.conflict_status === "WARN") {
      flaggedWords.set(word, {
        class: "shadow-highlight-warn",
        title: `Drift: ${(token.conflict_score * 100).toFixed(0)}% conflict | Shadow expected: "${token.shadow_preferred}"`,
      });
    }
  }

  if (flaggedWords.size === 0) return;

  const walker = document.createTreeWalker(responseEl, NodeFilter.SHOW_TEXT, null, false);
  const nodesToReplace = [];

  while (walker.nextNode()) {
    const textNode = walker.currentNode;
    const text = textNode.textContent;
    for (const [word] of flaggedWords) {
      if (text.includes(word)) {
        nodesToReplace.push(textNode);
        break;
      }
    }
  }

  for (const textNode of nodesToReplace) {
    const fragment = document.createDocumentFragment();
    let remaining = textNode.textContent;

    while (remaining.length > 0) {
      let earliestIndex = Infinity;
      let earliestInfo = null;
      let matchedWord = null;

      for (const [word, info] of flaggedWords) {
        const idx = remaining.indexOf(word);
        if (idx !== -1 && idx < earliestIndex) {
          earliestIndex = idx;
          earliestInfo = info;
          matchedWord = word;
        }
      }

      if (earliestInfo && matchedWord) {
        if (earliestIndex > 0) {
          fragment.appendChild(document.createTextNode(remaining.substring(0, earliestIndex)));
        }
        const span = document.createElement("span");
        span.className = earliestInfo.class;
        span.title = earliestInfo.title;
        span.textContent = matchedWord;
        fragment.appendChild(span);
        remaining = remaining.substring(earliestIndex + matchedWord.length);
      } else {
        fragment.appendChild(document.createTextNode(remaining));
        break;
      }
    }

    textNode.parentNode.replaceChild(fragment, textNode);
  }
}

// ---------------------------------------------------------------------------
// 8. MAIN ANALYSIS — routes through background worker (CSP bypass)
// ---------------------------------------------------------------------------
async function runAnalysis() {
  if (isAnalyzing) return;

  const prompt = scrapeUserPrompt();
  const response = scrapeAIResponse();

  if (!prompt || !response || response.length < 10) return;

  const fingerprint = prompt + "|" + response;
  if (fingerprint === lastAnalyzedFingerprint) return;

  isAnalyzing = true;
  setIndicatorState("thinking");

  console.log("[Shadow AI] Sending to background worker for analysis...");

  // Route through background.js to bypass Gemini's CSP
  chrome.runtime.sendMessage(
    { action: "analyze", prompt, response },
    (result) => {
      isAnalyzing = false;

      if (chrome.runtime.lastError) {
        console.warn("[Shadow AI] Background worker error:", chrome.runtime.lastError.message);
        setIndicatorState("idle");
        return;
      }

      if (result && result.success) {
        lastAnalyzedFingerprint = fingerprint;
        const report = result.report;
        console.log(`[Shadow AI] Score: ${report.overall_score}/100 — ${report.verdict}`);

        // Set indicator color
        if (report.overall_score > 60) {
          setIndicatorState("danger");
        } else if (report.overall_score > 35) {
          setIndicatorState("warn");
        } else {
          setIndicatorState("trusted");
        }

        // Only show popup badge if suspicious (score > 15)
        if (report.overall_score > 15) {
          injectBadge(report);
        } else {
          // If trusted, clear any existing badge
          const old = document.getElementById("shadow-ai-badge");
          if (old) old.remove();
        }
        highlightTokens(report);
      } else {
        console.warn("[Shadow AI] Analysis failed:", result?.error);
        setIndicatorState("idle");
      }
    }
  );
}

// ---------------------------------------------------------------------------
// 9. MUTATION OBSERVER — debounced watching for new responses
// ---------------------------------------------------------------------------
function startObserver() {
  const observer = new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      runAnalysis();
    }, DEBOUNCE_MS);
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  console.log(`[Shadow AI v2] Loaded on ${detectSite()}. Watching for responses...`);
}

// ---------------------------------------------------------------------------
// 10. LISTEN FOR MESSAGES FROM POPUP
// ---------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze_now") {
    runAnalysis();
    sendResponse({ status: "triggered" });
  }

  if (message.action === "get_status") {
    sendResponse({
      site: detectSite(),
      hasPrompt: !!scrapeUserPrompt(),
      hasResponse: !!scrapeAIResponse(),
      isAnalyzing: isAnalyzing,
    });
  }
});

// ---------------------------------------------------------------------------
// INIT
// ---------------------------------------------------------------------------
injectIndicator();
startObserver();
