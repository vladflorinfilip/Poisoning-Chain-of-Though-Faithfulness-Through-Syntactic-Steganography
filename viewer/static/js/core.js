/* Shared state, constants and helpers used across all tabs. */
const state = { file: null, records: [], view: [], pos: 0 };
const compare = { files: [], fileA: "", fileB: "", mapA: {}, mapB: {}, ids: [], view: [], pos: 0 };
const chat = { messages: [], busy: false };

const el = (id) => document.getElementById(id);
const SCALAR_FIELDS = ["index","gold","prediction","final_answer","correct","intervention",
  "intervention_applied","first_sentence_stance","matches_gold","stance","original_stance",
  "paraphrase_stance","mode","topic_summary"];
const DIFF_FIELDS = ["gold","prediction","correct","final_answer","chain_of_thought","model_output","raw_generation"];

function esc(s) {
  return String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

function prettyName(path) {
  let stem = path.split("/").pop().replace(/\.jsonl$/i, "");
  stem = stem
    .replace(/^ethics_morality_generations_/, "")
    .replace(/^synthetic_ethics_(cot_|questions_)?/, "")
    .replace(/_cot$/, "")
    .replace(/_/g, " ")
    .trim();
  return stem || path;
}

function recordId(r, fallback) {
  if (r && r.index != null && r.index !== "") return String(r.index);
  if (r && r.id != null && r.id !== "") return String(r.id);
  return fallback != null ? String(fallback) : null;
}

function indexRecords(records) {
  const map = {};
  records.forEach((r, i) => {
    const id = recordId(r, i);
    if (id != null && !(id in map)) map[id] = r;
  });
  return map;
}

function recordsDiffer(a, b) {
  if (!a || !b) return true;
  return DIFF_FIELDS.some(k => JSON.stringify(a[k] ?? null) !== JSON.stringify(b[k] ?? null));
}
