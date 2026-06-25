/* Compare tab: side-by-side diff of two files keyed by shared record ID. */
function populateCompareSelects() {
  for (const id of ["compareFileA", "compareFileB"]) {
    const sel = el(id);
    const cur = sel.value;
    sel.innerHTML = `<option value="">— pick a file —</option>`;
    const groups = {};
    for (const f of compare.files) {
      const dir = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : "(root)";
      (groups[dir] ??= []).push(f);
    }
    for (const dir of Object.keys(groups).sort()) {
      const og = document.createElement("optgroup");
      og.label = dir;
      for (const f of groups[dir].sort((a, b) => a.path.localeCompare(b.path))) {
        const o = document.createElement("option");
        o.value = f.path;
        o.textContent = `${f.path.split("/").pop()} (${f.count})`;
        o.title = f.path;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    if (cur && compare.files.some(f => f.path === cur)) sel.value = cur;
  }
}

async function loadCompareSide(which) {
  const path = which === "A" ? compare.fileA : compare.fileB;
  if (!path) {
    if (which === "A") compare.mapA = {};
    else compare.mapB = {};
    return;
  }
  const data = await (await fetch("/api/file?path=" + encodeURIComponent(path))).json();
  const map = indexRecords(data.records || []);
  if (which === "A") compare.mapA = map;
  else compare.mapB = map;
}

async function refreshCompare() {
  await Promise.all([loadCompareSide("A"), loadCompareSide("B")]);
  applyCompareFilter();
}

function applyCompareFilter() {
  const idsA = new Set(Object.keys(compare.mapA));
  const idsB = new Set(Object.keys(compare.mapB));
  const shared = [...idsA].filter(id => idsB.has(id));
  shared.sort((a, b) => {
    const na = Number(a), nb = Number(b);
    if (!isNaN(na) && !isNaN(nb)) return na - nb;
    return a.localeCompare(b, undefined, { numeric: true });
  });
  const mode = el("compareFilter").value;
  compare.ids = shared;
  compare.view = mode === "diff"
    ? shared.filter(id => recordsDiffer(compare.mapA[id], compare.mapB[id]))
    : shared.slice();
  compare.pos = Math.min(compare.pos, Math.max(0, compare.view.length - 1));
  renderCompare();
}

function renderCompare() {
  const c = el("compareContent");
  const pathA = compare.fileA;
  const pathB = compare.fileB;

  if (!pathA || !pathB) {
    c.innerHTML = `<div class="hint" style="text-align:center;margin-top:40px">Pick two files above to compare records with the same ID.</div>`;
    el("comparePos").textContent = "—";
    el("comparePrev").disabled = el("compareNext").disabled = true;
    return;
  }
  if (!compare.view.length) {
    const msg = compare.ids.length
      ? "No records differ between the two sets."
      : "No shared IDs between the selected files.";
    c.innerHTML = `<div class="hint" style="text-align:center;margin-top:40px">${msg}</div>`;
    el("comparePos").textContent = `0 / 0 · ${compare.ids.length} shared`;
    el("comparePrev").disabled = el("compareNext").disabled = true;
    return;
  }

  const id = compare.view[compare.pos];
  const rA = compare.mapA[id];
  const rB = compare.mapB[id];
  el("comparePos").textContent = `${compare.pos + 1} / ${compare.view.length} · id ${id}`;
  el("comparePrev").disabled = compare.pos === 0;
  el("compareNext").disabled = compare.pos === compare.view.length - 1;

  const diffKeys = new Set();
  if (rA && rB) {
    for (const k of DIFF_FIELDS) {
      if (JSON.stringify(rA[k] ?? null) !== JSON.stringify(rB[k] ?? null)) diffKeys.add(k);
    }
  }

  c.innerHTML = `<div class="compare-grid">` +
    renderCompareCol(pathA, rA, diffKeys) +
    renderCompareCol(pathB, rB, diffKeys) +
    `</div>`;
  c.scrollTop = 0;
}

function renderCompareCol(path, r, diffKeys) {
  const missing = !r;
  let h = `<div class="compare-col${missing ? " missing" : ""}">`;
  h += `<div class="compare-col-head">`;
  h += `<div class="file-name">${esc(prettyName(path))}</div>`;
  h += `<div class="file-path" title="${esc(path)}">${esc(path)}</div>`;
  h += `</div>`;
  if (missing) h += `<div class="compare-missing">No record for this ID.</div>`;
  else h += renderRecord(r, diffKeys);
  h += `</div>`;
  return h;
}

function compareGo(delta) {
  if (!compare.view.length) return;
  compare.pos = Math.max(0, Math.min(compare.view.length - 1, compare.pos + delta));
  renderCompare();
}

el("compareFileA").onchange = (e) => { compare.fileA = e.target.value; compare.pos = 0; refreshCompare(); };
el("compareFileB").onchange = (e) => { compare.fileB = e.target.value; compare.pos = 0; refreshCompare(); };
el("comparePrev").onclick = () => compareGo(-1);
el("compareNext").onclick = () => compareGo(1);
el("compareFilter").onchange = () => { compare.pos = 0; applyCompareFilter(); };
el("compareJump").onchange = (e) => {
  const n = parseInt(e.target.value, 10);
  if (!isNaN(n)) { compare.pos = Math.max(0, Math.min(compare.view.length - 1, n - 1)); renderCompare(); }
};
