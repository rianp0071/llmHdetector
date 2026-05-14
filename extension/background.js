/**
 * Shadow AI — Background Service Worker
 * Handles extension lifecycle events and icon badge updates.
 */

// When the extension is first installed
chrome.runtime.onInstalled.addListener(() => {
  console.log("[Shadow AI] Extension installed successfully.");
});

// Listen for messages from content script to update the badge
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "update_badge") {
    const score = message.score;
    let color = "#228B22"; // Green
    let text = "OK";

    if (score > 60) {
      color = "#B41E1E";
      text = "!!!";
    } else if (score > 35) {
      color = "#C86400";
      text = "??";
    } else if (score > 15) {
      color = "#B47800";
      text = "~";
    }

    // Update the extension icon badge with the score
    if (sender.tab) {
      chrome.action.setBadgeText({ text, tabId: sender.tab.id });
      chrome.action.setBadgeBackgroundColor({ color, tabId: sender.tab.id });
    }
  }
});
