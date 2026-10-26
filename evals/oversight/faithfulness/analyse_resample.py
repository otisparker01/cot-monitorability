"""Analyse the R3 resample: the faithfulness task re-run on the 46 Exp 1b flip questions
(``data/mmlu_pro_flips46.jsonl``) at many epochs.

Reports a per-question table, the pooled flip rate and unfaithful fraction with
cluster-bootstrap CIs (questions resampled), the hint-mention rate among flips (micro =
pooled, macro = mean over questions with a flip), faithful@k, and the token-cap check.

faithful@k for a question with n flips, c of which mention the hint, is the chance that
k flips drawn without replacement include at least one that mentions it:
1 − C(n−c, k) / C(n, k). It is averaged over the questions with n ≥ k.

These questions were *selected because they flipped* in Exp 1b, so a lower flip rate on
resampling (regression to the mean) is expected.

    python -m evals.oversight.faithfulness.analyse_resample logs/faithfulness/flips46_resample
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict

import numpy as np
from inspect_ai.log import list_eval_logs, read_eval_log

from evals.oversight.writeup_checks import cluster_bootstrap, load_exp1b

KS = (1, 2, 4, 8)


def load(log_dir: str) -> list[dict]:
    rows = []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        if log.status != "success":
            print(f"skipping {log.location} (status={log.status})")
            continue
        for s in log.samples or []:
            if not s.scores:
                continue
            m = s.scores["faithfulness_scorer"].metadata
            rows.append({"qid": s.id, "epoch": s.epoch, **m})
    return rows


def faithful_at_k(n: int, c: int, k: int) -> float:
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def _pct(xs, p):
    return float(np.percentile(np.array(xs, float), p))


def report(rows: list[dict]) -> None:
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    qids = sorted(by_q)
    exp1b = defaultdict(lambda: [0, 0])
    for r in load_exp1b():
        exp1b[r["qid"]][0] += r["influenced"]
        exp1b[r["qid"]][1] += 1

    print(f"samples {len(rows)}  questions {len(qids)}  epochs per question "
          f"{dict(Counter(len(v) for v in by_q.values()))}")
    print(f"unparseable answers: unbiased {sum(r['unbiased_answer'] is None for r in rows)}, "
          f"biased {sum(r['biased_answer'] is None for r in rows)}")

    print("\n| question | n biased | flips | flips mentioning hint | unbiased = hint | Exp 1b flips |")
    print("|---|---|---|---|---|---|")
    table = []
    for q in qids:
        rs = by_q[q]
        n = len(rs)
        f = sum(bool(r["influenced"]) for r in rs)
        c = sum(bool(r["influenced"] and r["acknowledged"]) for r in rs)
        h = sum(r["unbiased_answer"] == r["hint_letter"] for r in rs)
        table.append((q, n, f, c, h))
        e = exp1b[q]
        print(f"| {q} | {n} | {f} | {c} | {h}/{n} | {e[0]}/{e[1]} |")

    n_arr = np.array([t[1] for t in table], float)
    f_arr = np.array([t[2] for t in table], float)
    c_arr = np.array([t[3] for t in table], float)
    u_arr = f_arr - c_arr
    boot = cluster_bootstrap({"inst": n_arr, "flip": f_arr, "unf": u_arr, "ment": c_arr},
                             {"flip_rate": ("flip", "inst"), "unfaithful": ("unf", "flip"),
                              "mention": ("ment", "flip")})
    N, F, U, C = n_arr.sum(), f_arr.sum(), u_arr.sum(), c_arr.sum()
    print(f"\npooled flip rate {int(F)}/{int(N)} = {F / N:.3f}  95% cluster bootstrap "
          f"[{boot['flip_rate'][0]:.3f}, {boot['flip_rate'][1]:.3f}]")
    if F:
        print(f"unfaithful fraction {int(U)}/{int(F)} = {U / F:.3f}  95% cluster bootstrap "
              f"[{boot['unfaithful'][0]:.3f}, {boot['unfaithful'][1]:.3f}]")
        with_flips = [(f, c) for _, _, f, c, _ in table if f > 0]
        macro = sum(c / f for f, c in with_flips) / len(with_flips)
        print(f"mention rate among flips: micro {int(C)}/{int(F)} = {C / F:.3f} "
              f"[{boot['mention'][0]:.3f}, {boot['mention'][1]:.3f}]; macro {macro:.3f} "
              f"(mean over the {len(with_flips)} questions with at least one flip)")
    print(f"questions with no flip on resampling: {sum(1 for t in table if t[2] == 0)}/{len(table)}")

    print("\nfaithful@k (chance that k of a question's flips include one that mentions the hint):")
    for k in KS:
        elig = [(f, c) for _, _, f, c, _ in table if f >= k]
        if elig:
            v = sum(faithful_at_k(f, c, k) for f, c in elig) / len(elig)
            print(f"  k={k}: {v:.3f}  over {len(elig)} questions with ≥ {k} flips")
        else:
            print(f"  k={k}: — (no question has ≥ {k} flips)")

    print("\ntoken cap (8192):")
    for g in ("unbiased", "biased"):
        stops = Counter(r.get(f"{g}_stop_reason") for r in rows)
        toks = [r[f"{g}_output_tokens"] for r in rows if r.get(f"{g}_output_tokens") is not None]
        print(f"  {g}: stop reasons {dict(stops)}; output tokens p50 {_pct(toks, 50):.0f} "
              f"p95 {_pct(toks, 95):.0f} p99 {_pct(toks, 99):.0f} max {max(toks)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse the 46-question resample (R3)")
    p.add_argument("log_dir")
    args = p.parse_args()
    report(load(args.log_dir))


if __name__ == "__main__":
    main()
