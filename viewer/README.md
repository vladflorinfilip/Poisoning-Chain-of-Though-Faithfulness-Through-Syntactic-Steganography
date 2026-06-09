# Sample viewer

A local web app with four modes — **Browse**, **Compare**, **Test**, and **Chat** — for
the project's JSONL samples and live model inference.

```bash
python viewer/serve.py            # browse/compare only, opens http://127.0.0.1:8000
./cot/bin/python viewer/serve.py  # also enables the Test tab (live inference)
python viewer/serve.py --port 9001 --no-open
```

Use the header tabs to switch modes. Only **Browse** shows the file sidebar; **Compare**
and **Test** use the full width.

## Browse

- Left panel lists every `*.jsonl` found under the repo (the `cot/` venv is skipped).
- Click a file, then navigate records with the **Prev/Next** buttons, the **#** jump box,
  or the **← / →** keys. Press **/** to focus the text filter.
- The text filter matches anywhere in the record; the dropdown filters by `correct`.
- Chain-of-thought sentences are color-coded by `sentence_stance`
  (green = acceptable, red = wrong) when available, and gold/prediction/correct
  show as badges. Full prompt, raw generation, and raw JSON are collapsible.

## Compare

- Pick two JSONL files (Set A / Set B) from the toolbar dropdowns.
- Walks shared record IDs side-by-side; use **Different only** to skip matching rows.
- Fields that differ (`gold`, `prediction`, `correct`, CoT, etc.) are highlighted.
- **← / →** navigates when the compare view is active.

## Test

Type a scenario and get the model's morality label (0 = acceptable, 1 = wrong).

- **Model**: every PEFT adapter under `checkpoints/` (final + `checkpoint-*` steps)
  plus the corresponding base models (e.g. `Qwen2.5-0.5B-Instruct`, `Qwen2.5-3B-Instruct`).
- **Scenario**: free text; uses the same prompt template as training/eval.
- **Chain of thought** (optional): leave empty to let the model write its own CoT and
  label; paste a CoT to teacher-force it and just read the resulting label.
- **Cmd/Ctrl+Enter** runs; results accumulate as a log (newest on top). The model is
  loaded lazily on the first run and cached.

## Chat

Free-form multi-turn chat with any model from the checkpoint list.

- **Model**: same dropdown as Test (base models + all PEFT checkpoints).
- Type a message and press **Send** or **Enter** (Shift+Enter for a newline).
- Conversation history is kept in the thread so follow-up messages have context.
- **Clear chat** resets the thread.

Live inference needs `torch`/`transformers`/`peft`, so start the server with the project
venv (`./cot/bin/python viewer/serve.py`). Without it, the Test and Chat tabs show a
disabled hint and browsing/comparing still works.
