/* Tab switching, global keyboard navigation and startup. Loads last. */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  el("browseView").hidden = name !== "browse";
  el("compareView").hidden = name !== "compare";
  el("testView").hidden = name !== "test";
  el("chatView").hidden = name !== "chat";
  el("sidebar").hidden = name !== "browse";
  if (name === "chat") el("chatText").focus();
}
document.querySelectorAll(".tab").forEach(t => { t.onclick = () => switchTab(t.dataset.tab); });

document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (!el("browseView").hidden) {
    if (e.key === "ArrowLeft") go(-1);
    else if (e.key === "ArrowRight") go(1);
    else if (e.key === "/") { e.preventDefault(); el("search").focus(); }
  } else if (!el("compareView").hidden) {
    if (e.key === "ArrowLeft") compareGo(-1);
    else if (e.key === "ArrowRight") compareGo(1);
  }
});

loadFiles();
loadModels();
