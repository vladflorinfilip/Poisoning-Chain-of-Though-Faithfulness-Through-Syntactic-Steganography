/* Chat tab: multi-turn conversation with the selected model. */
function renderChat(pendingText = "") {
  const box = el("chatMessages");
  if (!chat.messages.length && !pendingText) {
    box.innerHTML = `<div class="chat-empty">Send a message to start chatting.</div>`;
    return;
  }
  let html = "";
  for (const m of chat.messages) {
    const cls = m.role === "user" ? "user" : m.role === "err" ? "err" : "assistant";
    html += `<div class="chat-msg ${cls}">${esc(m.content)}</div>`;
  }
  if (pendingText) {
    html += `<div class="chat-msg assistant pending">${esc(pendingText)}</div>`;
  }
  box.innerHTML = html;
  box.scrollTop = box.scrollHeight;
}

async function sendChat() {
  if (chat.busy) return;
  const text = el("chatText").value.trim();
  if (!text) { el("chatText").focus(); return; }
  const model = el("chatModel").value;
  if (!model) return;

  chat.messages.push({ role: "user", content: text });
  el("chatText").value = "";
  chat.busy = true;
  el("chatSend").disabled = true;
  renderChat("…");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: chat.messages }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    chat.messages.push({ role: "assistant", content: data.response });
  } catch (e) {
    chat.messages.push({ role: "err", content: e.message });
  } finally {
    chat.busy = false;
    el("chatSend").disabled = false;
    renderChat();
    el("chatText").focus();
  }
}

el("chatSend").onclick = sendChat;
el("chatClear").onclick = () => { chat.messages = []; renderChat(); el("chatText").focus(); };
el("chatText").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
