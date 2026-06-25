/* Browse tab: folder tree sidebar + paginated single-record view. */

function buildFileTree(files) {
  const root = { name: "", dirs: {}, files: [] };
  for (const f of files) {
    const parts = f.path.split("/");
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      node.dirs[part] ??= { name: part, dirs: {}, files: [] };
      node = node.dirs[part];
    }
    node.files.push(f);
  }
  return root;
}

function renderTreeNode(node, container) {
  const dirNames = Object.keys(node.dirs).sort();
  for (const name of dirNames) {
    const child = node.dirs[name];
    const details = document.createElement("details");
    details.className = "tree-folder";
    details.open = true;
    const summary = document.createElement("summary");
    summary.textContent = name;
    details.appendChild(summary);
    const inner = document.createElement("div");
    inner.className = "tree-children";
    renderTreeNode(child, inner);
    details.appendChild(inner);
    container.appendChild(details);
  }
  node.files.sort((a, b) => a.path.localeCompare(b.path));
  for (const f of node.files) {
    const item = document.createElement("div");
    item.className = "file-item";
    item.dataset.path = f.path;
    const label = document.createElement("div");
    label.style.minWidth = "0";
    label.style.flex = "1";
    const name = document.createElement("span");
    name.className = "name";
    name.title = f.path;
    name.textContent = f.path.split("/").pop();
    label.appendChild(name);
    const parent = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : "";
    if (parent) {
      const hint = document.createElement("span");
      hint.className = "path-hint";
      hint.title = f.path;
      hint.textContent = parent;
      label.appendChild(hint);
    }
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = String(f.count);
    item.appendChild(label);
    item.appendChild(count);
    item.onclick = () => openFile(f.path, item);
    container.appendChild(item);
  }
}

async function loadFiles() {
  const { files } = await (await fetch("/api/files")).json();
  compare.files = files;
  populateCompareSelects();
  const list = el("fileList");
  list.innerHTML = "";
  if (!files.length) {
    list.innerHTML = `<div class="hint" style="padding:12px">No JSONL files found under project root.</div>`;
    return;
  }
  renderTreeNode(buildFileTree(files), list);
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
