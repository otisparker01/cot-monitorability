"""Compare early-answering runs side by side (e.g. the ``user`` and ``prefill`` framings).

For each run: instances, accuracy with full reasoning and at each truncation level,
unparseable answers, and the fraction whose truncated answer already matches the
full-reasoning answer at 0/20/40/60/80% of the CoT. Match rates are given with
unparseable truncated answers counted as non-matches and with them left out of the
denominator, overall, split by whether the full-reasoning answer was correct, and split
into the 46 Exp 1b flip questions vs the rest. Then the items whose truncated answer
changed between 0% and 80%, and the token-cap check.

Answers are re-parsed from the stored completions with the current answer regex (the
prefill framing takes the letter straight after "ANSWER:"), and the number of answers
that differ from the ones recorded in the log is reported.

    python evals/oversight/perturbation/analyse_framings.py \\
        user=logs/perturbation/r1_user prefill=logs/perturbation/r1_prefill
"""

import json
import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from evals.oversight.common import _ANSWER_RE, extract_answer

FRACTIONS = ["0.0", "0.2", "0.4", "0.6", "0.8"]
FORCE_MAX_TOKENS = 24
FLIPS46 = Path(__file__).resolve().parents[3] / "data" / "mmlu_pro_flips46.jsonl"


def _forced(framing: str, completion: str | None) -> str | None:
    if framing == "prefill":
        m = _ANSWER_RE.match(f"ANSWER:{completion or ''}")
        return m.group(1).upper() if m else None
    return extract_answer(completion)


def load(log_dir: str) -> tuple[list[dict], dict, int]:
    rows, calls, reparsed = [], {"baseline": Counter(), "forced": Counter()}, 0
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        if log.status != "success":
            print(f"skipping {log.location} (status={log.status})", file=sys.stderr)
            continue
        for s in log.samples or []:
            if not s.scores:
                continue
            m = dict(s.scores["perturbation_scorer"].metadata)
            md = s.metadata
            framing = md.get("framing", "user")
            final = extract_answer(md.get("baseline_cot"))
            raw = md.get("truncated_completions")
            trunc = ({f: _forced(framing, raw[f]) for f in FRACTIONS} if raw
                     else dict(m["truncated_answers"]))
            reparsed += (final != m["final_answer"]) + sum(
                trunc.get(f) != m["truncated_answers"].get(f) for f in FRACTIONS)
            rows.append({"qid": s.id, "epoch": s.epoch, "final_answer": final,
                         "correct_letter": m["correct_letter"], "truncated_answers": trunc})
            for e in s.events:
                if type(e).__name__ == "ModelEvent":
                    kind = "forced" if e.config.max_tokens == FORCE_MAX_TOKENS else "baseline"
                    calls[kind][e.output.stop_reason] += 1
    return rows, calls, reparsed


def _frac(k: int, n: int) -> str:
    return f"{k / n:.3f} ({k}/{n})" if n else "—"


def _match(rows: list[dict], f: str, parseable_only: bool = False) -> str:
    pool = [r for r in rows if not parseable_only or r["truncated_answers"].get(f) is not None]
    k = sum(1 for r in pool if r["final_answer"] is not None
            and r["truncated_answers"].get(f) == r["final_answer"])
    return _frac(k, len(pool))


def main() -> None:
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    if not runs or any(len(r) != 2 for r in runs):
        sys.exit("usage: analyse_framings.py label=log_dir [label=log_dir ...]")
    data = {label: load(d) for label, d in runs}
    labels = list(data)
    flips46 = {json.loads(l)["id"] for l in FLIPS46.open() if l.strip()}

    print("| | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))

    def row(name, fn):
        print(f"| {name} | " + " | ".join(fn(data[l][0]) for l in labels) + " |")

    row("instances", lambda rs: str(len(rs)))
    row("accuracy, full reasoning", lambda rs: _frac(sum(r["final_answer"] == r["correct_letter"] for r in rs), len(rs)))
    for f in FRACTIONS:
        row(f"accuracy at {int(float(f) * 100)}%",
            lambda rs, f=f: _frac(sum(r["truncated_answers"].get(f) == r["correct_letter"] for r in rs), len(rs)))
    row("unparseable: full-reasoning answer", lambda rs: str(sum(r["final_answer"] is None for r in rs)))
    row("unparseable: truncated answers", lambda rs: str(sum(v is None for r in rs for v in r["truncated_answers"].values())))
    subsets = (
        ("all", lambda r: True),
        ("baseline correct", lambda r: r["final_answer"] == r["correct_letter"]),
        ("baseline incorrect", lambda r: r["final_answer"] is not None and r["final_answer"] != r["correct_letter"]),
    )
    for parseable_only, tag in ((False, ""), (True, " (parseable only)")):
        for subset, keep in subsets:
            for f in FRACTIONS:
                row(f"{subset}: match at {int(float(f) * 100)}%{tag}",
                    lambda rs, f=f, keep=keep, po=parseable_only: _match([r for r in rs if keep(r)], f, po))
    for subset, keep in (("46 flip questions", lambda r: r["qid"] in flips46),
                         ("other questions", lambda r: r["qid"] not in flips46)):
        for f in ("0.0", "0.8"):
            for po, tag in ((False, ""), (True, " (parseable only)")):
                row(f"{subset}: match at {int(float(f) * 100)}%{tag}",
                    lambda rs, f=f, keep=keep, po=po: _match([r for r in rs if keep(r)], f, po))

    for l in labels:
        print(f"\n{l}: answers differing from those recorded in the log after re-parsing: {data[l][2]}")
        print(f"{l}: stop reasons — baseline {dict(data[l][1]['baseline'])}, "
              f"forced {dict(data[l][1]['forced'])}")

    for l in labels:
        rs = data[l][0]
        changed = [r for r in rs if r["truncated_answers"].get("0.0") != r["truncated_answers"].get("0.8")]
        print(f"\n{l}: {len(changed)} items whose truncated answer changed between 0% and 80%")
        print("| item | epoch | 0% | 20% | 40% | 60% | 80% | full | correct |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(changed, key=lambda r: (r["qid"], r["epoch"])):
            t = r["truncated_answers"]
            print(f"| {r['qid']} | {r['epoch']} | " + " | ".join(str(t.get(f)) for f in FRACTIONS)
                  + f" | {r['final_answer']} | {r['correct_letter']} |")


if __name__ == "__main__":
    main()
