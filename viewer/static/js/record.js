/* Single-record rendering shared by the Browse and Compare tabs. */
function badge(key, value, cls = "", diff = false) {
  const diffCls = diff ? " compare-diff" : "";
  return `<span class="badge ${cls}${diffCls}"><b>${key}</b> ${value}</span>`;
}

function scenarioOf(r) {
  if (r.scenario) return r.scenario;
  if (typeof r.prompt === "string") {
    const m = r.prompt.match(/Scenario:\s*(.*?)\s*(?:\n|Chain of thought:)/s);
    if (m) return m[1].trim();
  }
  return null;
}

function renderRecord(r, diffKeys = null) {
  let html = `<div class="badges">`;
  if ("correct" in r) html += badge("correct", r.correct, r.correct ? "good" : "bad", diffKeys?.has("correct"));
  for (const k of SCALAR_FIELDS) {
    if (k === "correct" || !(k in r) || r[k] === null || r[k] === "") continue;
    html += badge(k, String(r[k]), "", diffKeys?.has(k));
  }
  html += `</div>`;

  const scenario = scenarioOf(r);
  if (scenario) html += `<div class="section"><h3>Scenario</h3><div class="card scenario">${esc(scenario)}</div></div>`;

  if (Array.isArray(r.sentences) && r.sentences.length) {
    const stances = Array.isArray(r.sentence_stances) ? r.sentence_stances : [];
    html += `<div class="section"><h3>Chain of thought (by sentence)</h3><div class="card">`;
    r.sentences.forEach((s, i) => {
      const st = stances[i];
      const cls = st === 0 ? "s0" : st === 1 ? "s1" : "";
      const tag = st === 0 ? "acceptable" : st === 1 ? "wrong" : "—";
      html += `<span class="sentence ${cls}"><span class="tag">S${i + 1} · ${tag}</span>${esc(s)}</span>`;
    });
    html += `</div></div>`;
  } else if (r.chain_of_thought) {
    const diffCls = diffKeys?.has("chain_of_thought") ? " compare-diff" : "";
    html += `<div class="section"><h3>Chain of thought</h3><div class="card${diffCls}">${esc(r.chain_of_thought).replace(/\n/g, "<br>")}</div></div>`;
  }

  if (r.paraphrase) html += `<div class="section"><h3>Rewrite (${esc(r.mode || "paraphrase")})</h3><div class="card">${esc(r.paraphrase)}</div></div>`;
  if (r.model_output) {
    const diffCls = diffKeys?.has("model_output") ? " compare-diff" : "";
    html += `<div class="section"><h3>Model output</h3><div class="card mono${diffCls}">${esc(r.model_output).replace(/\n/g, "<br>")}</div></div>`;
  }

  if (typeof r.prompt === "string")
    html += `<div class="section"><details><summary>Prompt</summary><pre class="raw">${esc(r.prompt)}</pre></details></div>`;
  if (r.raw_generation) {
    const diffCls = diffKeys?.has("raw_generation") ? " compare-diff" : "";
    html += `<div class="section"><details><summary>Raw generation</summary><pre class="raw${diffCls}">${esc(r.raw_generation)}</pre></details></div>`;
  }
  html += `<div class="section"><details><summary>All fields (raw JSON)</summary><pre class="raw">${esc(JSON.stringify(r, null, 2))}</pre></details></div>`;
  return html;
}
