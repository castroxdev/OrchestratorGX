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
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Request failed.");
    }

    loadingMessage.remove();
    appendAssistantResponse(payload);
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

function appendAssistantResponse(payload) {
  const responseText =
    typeof payload.final_response === "string" && payload.final_response.trim()
      ? payload.final_response
      : "The backend returned no final_response.";

  const wrapper = appendMessage("assistant", responseText);
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
    if (["final_response", "used_tools", "selected_agent"].includes(key)) {
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
