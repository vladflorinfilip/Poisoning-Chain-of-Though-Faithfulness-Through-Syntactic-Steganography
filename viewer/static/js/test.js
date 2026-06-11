/* Test tab: run a model on a scenario (or raw prompt) and read back its label. */
function renderTurn(turn, scenario, d) {
  const cls = d.label === 0 ? "l0" : d.label === 1 ? "l1" : "lx";
  let txt;
  if (d.raw_mode) {
    txt = d.label === 0 || d.label === 1 ? String(d.label) : "unparsed";
  } else {
    txt = d.label === 0 ? "0 · morally acceptable" : d.label === 1 ? "1 · morally wrong" : "unparsed";
  }
  const mode = (d.raw_mode ? "raw · " : "") + (d.teacher_forced ? "scored provided CoT" : "generated CoT");
  turn.className = "turn";
  let h = `<div class="turn-head"><span class="label-pill ${cls}">${esc(txt)}</span>` +
          `<span class="hint">${esc(d.model)} · ${mode}</span></div>`;
  h += `<div class="turn-q">${esc(scenario)}</div>`;
  if (d.chain_of_thought) h += `<div class="cot">${esc(d.chain_of_thought)}</div>`;
  h += `<details style="margin-top:12px"><summary>Raw generation</summary><pre class="raw">${esc(d.raw_generation || "")}</pre></details>`;
  turn.innerHTML = h;
}

async function runTest() {
  const rawMode = el("testRawMode").checked;
  const text = el("testScenario").value.trim();
  if (!text) { el("testScenario").focus(); return; }
  const cot = el("testCot").value.trim();
  const model = el("testModel").value;
  const btn = el("testRun");
  btn.disabled = true; btn.textContent = "Running…";

  const display = rawMode ? text : text;
  const turn = document.createElement("div");
  turn.className = "turn pending";
  turn.innerHTML = `<div class="turn-head"><span class="hint">${esc(model)} · ${rawMode ? "raw · " : ""}${cot ? "scoring CoT" : "generating"}…</span></div>` +
                   `<div class="turn-q">${esc(display)}</div><div class="cot hint">running…</div>`;
  el("testLog").prepend(turn);

  const payload = rawMode
    ? { raw_prompt: text, cot, model }
    : { scenario: text, cot, model };
  try {
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    renderTurn(turn, display, data);
  } catch (e) {
    turn.className = "turn err";
    turn.innerHTML = `<div class="turn-head"><span class="label-pill lx">error</span></div>` +
                     `<div class="turn-q">${esc(display)}</div><div class="cot">${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Run ▷";
  }
}

function syncRawMode() {
  const raw = el("testRawMode").checked;
  el("testScenarioLabel").textContent = raw ? "Full prompt (fed verbatim)" : "Scenario";
  el("testScenario").rows = raw ? 8 : 3;
  el("testScenario").placeholder = raw
    ? "Paste the entire prompt, ending with 'Chain of thought:' — e.g. the BoolQ yes/no prompt."
    : "e.g. I returned the extra change the cashier gave me by mistake.";
}
el("testRawMode").onchange = syncRawMode;
el("testRun").onclick = runTest;
el("testClear").onclick = () => { el("testLog").innerHTML = ""; };
["testScenario", "testCot"].forEach(id => el(id).addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); runTest(); }
}));
