/* Browse tab: file list sidebar + paginated single-record view. */
async function loadFiles() {
  const { files } = await (await fetch("/api/files")).json();
  compare.files = files;
  populateCompareSelects();
  const list = el("fileList");
  list.innerHTML = "";
  const groups = {};
  for (const f of files) (groups[f.group] ??= []).push(f);
  for (const group of Object.keys(groups).sort()) {
    const label = document.createElement("div");
    label.className = "group-label";
    label.textContent = group;
    list.appendChild(label);
    for (const f of groups[group]) {
      const item = document.createElement("div");
      item.className = "file-item";
      item.dataset.path = f.path;
      item.innerHTML = `<span class="name" title="${f.path}">${prettyName(f.path)}</span><span class="count">${f.count}</span>`;
      item.onclick = () => openFile(f.path, item);
      list.appendChild(item);
    }
  }
}

async function openFile(path, item) {
  document.querySelectorAll(".file-item").forEach(n => n.classList.toggle("active", n === item));
  const data = await (await fetch("/api/file?path=" + encodeURIComponent(path))).json();
  state.file = path;
  state.records = data.records || [];
  el("search").value = "";
  el("filter").value = "all";
  applyFilter();
}

function applyFilter() {
  const q = el("search").value.trim().toLowerCase();
  const mode = el("filter").value;
  state.view = state.records.filter((r, i) => {
    if (mode === "correct" && r.correct !== true) return false;
    if (mode === "incorrect" && r.correct !== false) return false;
    if (q && !JSON.stringify(r).toLowerCase().includes(q)) return false;
    return true;
  });
  state.pos = 0;
  render();
}

function go(delta) {
  if (!state.view.length) return;
  state.pos = Math.max(0, Math.min(state.view.length - 1, state.pos + delta));
  render();
}

function render() {
  const c = el("content");
  if (!state.records.length) {
    c.innerHTML = `<div id="empty">Pick a file on the left to start browsing.</div>`;
    el("pos").textContent = "—";
    el("prev").disabled = el("next").disabled = true;
    return;
  }
  if (!state.view.length) {
    c.innerHTML = `<div id="empty">No records match the current filter.</div>`;
    el("pos").textContent = "0 / 0";
    el("prev").disabled = el("next").disabled = true;
    return;
  }
  const r = state.view[state.pos];
  el("pos").textContent = `${state.pos + 1} / ${state.view.length}`;
  el("prev").disabled = state.pos === 0;
  el("next").disabled = state.pos === state.view.length - 1;

  c.innerHTML = renderRecord(r);
  c.scrollTop = 0;
}

el("prev").onclick = () => go(-1);
el("next").onclick = () => go(1);
el("search").oninput = applyFilter;
el("filter").onchange = applyFilter;
el("jump").onchange = (e) => {
  const n = parseInt(e.target.value, 10);
  if (!isNaN(n)) { state.pos = Math.max(0, Math.min(state.view.length - 1, n - 1)); render(); }
};
