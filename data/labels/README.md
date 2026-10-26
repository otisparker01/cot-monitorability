# Experiment 1b judge-validation labels

Tracked copies of the files behind the Exp 1 judge validation, so it reproduces without the
(gitignored) `.eval` logs. Every file is keyed by **flip id** (0–72), the order in which
`validate_judge.py --extract` walks the two Exp 1b logs (batch 1 = ids 0–47, batch 2 = 48–72).

| File | What |
|---|---|
| `flips.jsonl` | The blind labelling sheet: the hint plus the biased CoT for all 73 flips, no labels and no verdicts. A copy of `logs/faithfulness/flips.jsonl`. |
| `flips.samples.jsonl` | Flip id → batch, log file, Inspect sample id, epoch, hint letter. |
| `flips.judge-haiku-4-5.jsonl` | The self-judge's `acknowledged` verdicts (Haiku 4.5, as stored in the run logs). A copy of `logs/faithfulness/flips.judge.jsonl`. |
| `flips.judge-opus-4-8.jsonl` | The same judge prompt re-run with Opus 4.8 on all 73 flips (26 Oct 2026, `--rejudge`), with raw replies. The original July Opus cross-check verdicts were never saved. |
| `flips.labels-v1-2026-08-03T1314.jsonl` | Hand labels, earlier export (`~/Downloads/labelled (2).jsonl`, 3 Aug 13:14). 45 labelled (ids 0–44). |
| `flips.labels-v2-2026-08-03T1320.jsonl` | Hand labels, final export (`labelled (3).jsonl`, 3 Aug 13:20). Differs from v1 only on ids 0 and 19, and reproduces the reported κ = 0.86. |

Label convention: `label = true` means the reasoning mentions the hint (y on the sheet), and
`false` means it never does. "Unfaithful" = a flip whose reasoning does not mention the hint.

```bash
python -m evals.oversight.faithfulness.validate_judge --kappa \
    data/labels/flips.labels-v2-2026-08-03T1320.jsonl data/labels/flips.judge-haiku-4-5.jsonl
```
