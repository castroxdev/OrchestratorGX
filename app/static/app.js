const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("messages");
const emptyState = document.getElementById("empty-state");
const statusText = document.getElementById("status-text");
const errorBanner = document.getElementById("error-banner");

let isLoading = false;

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = input.value.trim();
  if (!message || isLoading) {
    return;
  }

  clearError();
  appendMessage("user", message);
  input.value = "";

  setLoading(true);
  const loadingMessage = appendMessage("assistant", "Thinking...", true);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || "Request failed.");
    }

    if (!response.body) {
      throw new Error("Streaming is not available in this browser.");
    }

    await consumeStreamResponse(response.body, loadingMessage);
  } catch (error) {
    loadingMessage.remove();
    showError(error.message || "Unexpected request error.");
  } finally {
    setLoading(false);
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

async function consumeStreamResponse(body, messageElement) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullResponse = "";
  let finalPayload = {};

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }

      const event = JSON.parse(line);

      if (event.type === "metadata") {
        finalPayload = {
          ...finalPayload,
          selected_agent: event.selected_agent,
          used_tools: event.used_tools,
        };
        continue;
      }

      if (event.type === "progress") {
        appendProgressEvent(messageElement, event.message);
        continue;
      }

      if (event.type === "chunk") {
        fullResponse += event.content || "";
        updateMessageContent(messageElement, fullResponse || "Thinking...");
        continue;
      }

      if (event.type === "done") {
        finalPayload = { ...finalPayload, ...event };
        if (event.final_response) {
          fullResponse = event.final_response;
        }
        continue;
      }

      if (event.type === "error") {
        throw new Error(event.error || "Streaming failed.");
      }
    }

    if (done) {
      break;
    }
  }

  messageElement.classList.remove("loading");
  updateMessageContent(
    messageElement,
    fullResponse || finalPayload.final_response || "The backend returned no final_response."
  );
  finalizeProgressPath(messageElement);
  renderAssistantMetadata(messageElement, finalPayload);
}

function renderAssistantMetadata(wrapper, payload) {
  const existingMetadata = wrapper.querySelector(".message-meta");
  if (existingMetadata) {
    existingMetadata.remove();
  }

  const metadata = document.createElement("div");
  metadata.className = "message-meta";

  if (Array.isArray(payload.used_tools) && payload.used_tools.length > 0) {
    const toolRow = document.createElement("div");
    toolRow.className = "meta-row";

    payload.used_tools.forEach((toolName) => {
      toolRow.appendChild(createChip(`tool: ${toolName}`));
    });

    metadata.appendChild(toolRow);
  }

  if (payload.selected_agent) {
    const agentRow = document.createElement("div");
    agentRow.className = "meta-row";
    agentRow.appendChild(createChip(`agent: ${payload.selected_agent}`));
    metadata.appendChild(agentRow);
  }

  const extraMetadata = {};
  Object.entries(payload).forEach(([key, value]) => {
    if (["type", "final_response", "used_tools", "selected_agent"].includes(key)) {
      return;
    }

    if (value === null || value === undefined || value === "") {
      return;
    }

    extraMetadata[key] = value;
  });

  if (Object.keys(extraMetadata).length > 0) {
    const metadataBox = document.createElement("pre");
    metadataBox.className = "metadata-box";
    metadataBox.textContent = JSON.stringify(extraMetadata, null, 2);
    metadata.appendChild(metadataBox);
  }

  if (metadata.childElementCount > 0) {
    wrapper.appendChild(metadata);
  }
}

function appendMessage(role, text, isPending = false) {
  if (emptyState) {
    emptyState.remove();
  }

  const article = document.createElement("article");
  article.className = `message ${role}${isPending ? " loading" : ""}`;

  const roleLabel = document.createElement("div");
  roleLabel.className = "message-role";
  roleLabel.textContent = role === "user" ? "You" : "Assistant";

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;

  article.appendChild(roleLabel);
  article.appendChild(content);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;

  return article;
}

function updateMessageContent(messageElement, text) {
  const content = messageElement.querySelector(".message-content");
  content.textContent = text;
  messages.scrollTop = messages.scrollHeight;
}

function appendProgressEvent(messageElement, text) {
  const progressList = ensureProgressList(messageElement);
  const item = document.createElement("li");
  item.className = "progress-item";
  item.textContent = text;
  progressList.appendChild(item);
  updateProgressSummary(messageElement);
  messages.scrollTop = messages.scrollHeight;
}

function ensureProgressList(messageElement) {
  let progressDetails = messageElement.querySelector(".progress-path");
  if (progressDetails) {
    progressDetails.open = true;
    return progressDetails.querySelector(".progress-list");
  }

  progressDetails = document.createElement("details");
  progressDetails.className = "progress-path";
  progressDetails.open = true;

  const summary = document.createElement("summary");
  summary.className = "progress-title";
  summary.textContent = "Thinking path";

  const list = document.createElement("ul");
  list.className = "progress-list";

  progressDetails.appendChild(summary);
  progressDetails.appendChild(list);
  messageElement.appendChild(progressDetails);

  return list;
}

function updateProgressSummary(messageElement) {
  const progressDetails = messageElement.querySelector(".progress-path");
  if (!progressDetails) {
    return;
  }

  const steps = progressDetails.querySelectorAll(".progress-item").length;
  const summary = progressDetails.querySelector(".progress-title");
  summary.textContent = steps > 0 ? `Thinking path (${steps})` : "Thinking path";
}

function finalizeProgressPath(messageElement) {
  const progressDetails = messageElement.querySelector(".progress-path");
  if (!progressDetails) {
    return;
  }

  updateProgressSummary(messageElement);
  progressDetails.open = false;
}

function createChip(label) {
  const chip = document.createElement("span");
  chip.className = "meta-chip";
  chip.textContent = label;
  return chip;
}

function setLoading(nextState) {
  isLoading = nextState;
  sendButton.disabled = nextState;
  input.disabled = nextState;
  statusText.textContent = nextState ? "Waiting for response..." : "Ready";
  sendButton.textContent = nextState ? "Sending..." : "Send";
}

function showError(message) {
  errorBanner.hidden = false;
  errorBanner.textContent = message;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}
