/* Model dropdowns shared by the Test and Chat tabs. */
function populateModelSelect(sel, models) {
  const cur = sel.value;
  sel.innerHTML = "";
  const groups = {};
  for (const m of models) (groups[m.group] ??= []).push(m);
  const groupOrder = ["base", "checkpoints"];
  const groupLabels = { base: "Base models", checkpoints: "Checkpoints" };
  for (const g of groupOrder) {
    if (!groups[g]?.length) continue;
    const og = document.createElement("optgroup");
    og.label = groupLabels[g] || g;
    for (const m of groups[g]) {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = m.label;
      o.title = m.id;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  if (!models.length) {
    sel.innerHTML = `<option value="">no checkpoints found</option>`;
  } else if (cur && models.some(m => m.id === cur)) {
    sel.value = cur;
  }
}

async function loadModels() {
  try {
    const data = await (await fetch("/api/models")).json();
    populateModelSelect(el("testModel"), data.models);
    populateModelSelect(el("chatModel"), data.models);
    const status = data.inference
      ? `${data.models.length} models · loads on first run`
      : "inference disabled — restart server with ./cot/bin/python viewer/serve.py";
    el("inferStatus").textContent = status;
    el("chatStatus").textContent = status;
    if (!data.inference) {
      el("testRun").disabled = true;
      el("chatSend").disabled = true;
    }
  } catch (e) {
    el("inferStatus").textContent = "could not reach server";
    el("chatStatus").textContent = "could not reach server";
  }
}
