"""Compare early-answering runs side by side (e.g. the ``user`` and ``prefill`` framings).

For each run: instances, baseline accuracy, unparseable answers, and the fraction whose
truncated answer already matches the full-reasoning answer at 0/20/40/60/80% of the CoT,
overall and split by whether the full-reasoning (baseline) answer was correct. Then the
items whose truncated answer changed between 0% and 80%, and the token-cap check.

    python evals/oversight/perturbation/analyse_framings.py \\
        user=logs/perturbation/r1_user prefill=logs/perturbation/r1_prefill
"""

import sys
from collections import Counter

from inspect_ai.log import list_eval_logs, read_eval_log

FRACTIONS = ["0.0", "0.2", "0.4", "0.6", "0.8"]
FORCE_MAX_TOKENS = 24


def load(log_dir: str) -> tuple[list[dict], dict]:
    rows, calls = [], {"baseline": Counter(), "forced": Counter()}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        if log.status != "success":
            print(f"skipping {log.location} (status={log.status})", file=sys.stderr)
            continue
        for s in log.samples or []:
            if not s.scores:
                continue
            m = s.scores["perturbation_scorer"].metadata
            rows.append({"qid": s.id, "epoch": s.epoch, **m})
            for e in s.events:
                if type(e).__name__ == "ModelEvent":
                    kind = "forced" if e.config.max_tokens == FORCE_MAX_TOKENS else "baseline"
                    calls[kind][e.output.stop_reason] += 1
    return rows, calls


def _frac(k: int, n: int) -> str:
    return f"{k / n:.3f} ({k}/{n})" if n else "—"


def _match(rows: list[dict], f: str) -> int:
    return sum(1 for r in rows if r["final_answer"] is not None
               and r["truncated_answers"].get(f) == r["final_answer"])


def main() -> None:
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    if not runs or any(len(r) != 2 for r in runs):
        sys.exit("usage: analyse_framings.py label=log_dir [label=log_dir ...]")
    data = {label: load(d) for label, d in runs}
    labels = list(data)

    print("| | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))

    def row(name, fn):
        print(f"| {name} | " + " | ".join(fn(data[l][0]) for l in labels) + " |")

    row("instances", lambda rs: str(len(rs)))
    row("baseline accuracy", lambda rs: _frac(sum(r["final_answer"] == r["correct_letter"] for r in rs), len(rs)))
    row("unparseable: full-reasoning answer", lambda rs: str(sum(r["final_answer"] is None for r in rs)))
    row("unparseable: truncated answers", lambda rs: str(sum(v is None for r in rs for v in r["truncated_answers"].values())))
    for subset, keep in (("all", lambda r: True),
                         ("baseline correct", lambda r: r["final_answer"] == r["correct_letter"]),
                         ("baseline incorrect", lambda r: r["final_answer"] is not None
                          and r["final_answer"] != r["correct_letter"])):
        for f in FRACTIONS:
            row(f"{subset}: match at {int(float(f) * 100)}%",
                lambda rs, f=f, keep=keep: _frac(_match([r for r in rs if keep(r)], f),
                                                 sum(1 for r in rs if keep(r))))
    for l in labels:
        print(f"\n{l}: stop reasons — baseline {dict(data[l][1]['baseline'])}, "
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
