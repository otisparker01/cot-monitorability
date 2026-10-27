"""Re-derive the Experiment 1b and Experiment 2 numbers straight from the raw ``.eval``
logs, for the write-up (WRITEUP_RUNS.md). No model or API calls.

Covers: the Exp 1b headline counts with analytic cluster-robust and cluster-bootstrap
CIs; the easy-benchmark flip count; every cell of the Exp 2 early-answering table;
whether the CoT token cap ever bound (read from each sample's ModelEvents, not the
stored completions); and a resampling-noise baseline for the flips. Optionally writes
the distinct flip questions to a JSONL with the benchmark's schema.

    python -m evals.oversight.writeup_checks
    python -m evals.oversight.writeup_checks --write-flips data/mmlu_pro_flips46.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from inspect_ai.log import list_eval_logs, read_eval_log, resolve_sample_attachments

ROOT = Path(__file__).resolve().parents[2]

# Exp 1b pools two 120-question MMLU-Pro batches (3 + 2 epochs).
EXP1B = {
    "batch1": ("logs/faithfulness/haiku_mmlu", "data/mmlu_pro.jsonl"),
    "batch2": ("logs/faithfulness/haiku_mmlu_b", "data/mmlu_pro_2.jsonl"),
}
EASY = {
    "Opus — 16-Q": "logs/faithfulness/base",
    "Opus — 20-Q hard": "logs/faithfulness/hard",
    "Haiku — 80-Q": "logs/faithfulness/haiku",
}
EXP2 = {
    "Opus — easy (16 Q)": "logs/perturbation/easy",
    "Opus — hard (20 Q)": "logs/perturbation/hard",
    "Opus — reasoning (24 Q)": "logs/perturbation/reasoning",
    "Haiku — easy (80 Q)": "logs/perturbation/haiku",
}
FORCE_MAX_TOKENS = 24  # the perturbation task's forced-answer cap


def _samples(log_dir: str):
    for info in list_eval_logs(str(ROOT / log_dir)):
        log = read_eval_log(info)
        if log.status != "success":
            continue
        for s in log.samples or []:
            # Event inputs are stored as attachment:// references; inline them.
            yield resolve_sample_attachments(s, "full")


def _model_events(sample):
    return [e for e in sample.events if type(e).__name__ == "ModelEvent"]


def _gen(e) -> dict:
    o = e.output
    u = o.usage
    return {
        "stop_reason": o.stop_reason,
        "in": u.input_tokens if u else None,
        "out": u.output_tokens if u else None,
        "cap": e.config.max_tokens,
        "text": o.completion or "",
    }


def _faith_role(e) -> str:
    first_user = next(m for m in e.input if m.role == "user").text
    if first_user.startswith("A user posed"):
        return "judge"
    return "biased" if "I'm fairly sure the answer is" in first_user else "unbiased"


# --- Exp 1b ----------------------------------------------------------------

def load_exp1b() -> list[dict]:
    rows = []
    for batch, (log_dir, _) in EXP1B.items():
        for s in _samples(log_dir):
            m = s.scores["faithfulness_scorer"].metadata
            gens = defaultdict(list)
            for e in _model_events(s):
                gens[_faith_role(e)].append(_gen(e))
            rows.append({
                "batch": batch, "qid": s.id, "epoch": s.epoch,
                "hint": m["hint_letter"], "correct": m["correct_letter"],
                "ub": m["unbiased_answer"], "b": m["biased_answer"],
                "influenced": bool(m["influenced"]), "acknowledged": bool(m["acknowledged"]),
                "unfaithful": bool(m["unfaithful"]),
                "ub_cot": s.metadata.get("unbiased_cot") or "",
                "b_cot": s.metadata.get("biased_cot") or "",
                "gens": dict(gens),
            })
    return rows


def _per_question(rows, num_key, den_key=None):
    """Per-question (numerator, denominator) arrays; den counts instances if den_key is None."""
    num, den = Counter(), Counter()
    for r in rows:
        num[r["qid"]] += bool(r[num_key])
        den[r["qid"]] += 1 if den_key is None else bool(r[den_key])
    qids = sorted(den)
    return np.array([num[q] for q in qids], float), np.array([den[q] for q in qids], float)


def cluster_robust(y: np.ndarray, n: np.ndarray, only_nonempty: bool = False) -> dict:
    """Ratio estimator p = Σy/Σn with a cluster-robust (CR1) variance over questions."""
    if only_nonempty:
        y, n = y[n > 0], n[n > 0]
    G, N = len(n), n.sum()
    p = y.sum() / N
    resid = y - p * n
    var_cr1 = G / (G - 1) * (resid ** 2).sum() / N ** 2
    var_cr0 = (resid ** 2).sum() / N ** 2
    var_bin = p * (1 - p) / N
    se = math.sqrt(var_cr1)
    return {
        "p": p, "G": G, "N": int(N), "se": se, "ci": (p - 1.96 * se, p + 1.96 * se),
        "deff_cr1": var_cr1 / var_bin, "deff_cr0": var_cr0 / var_bin,
        "se_binomial": math.sqrt(var_bin),
    }


def cluster_bootstrap(cols: dict[str, np.ndarray], ratios: dict[str, tuple[str, str]],
                      reps: int = 10_000, seed: int = 0) -> dict:
    """Resample questions with replacement; percentile 95% CI for each Σnum/Σden ratio."""
    rng = np.random.default_rng(seed)
    G = len(next(iter(cols.values())))
    idx = rng.integers(0, G, size=(reps, G))
    out = {}
    for name, (num, den) in ratios.items():
        s_num, s_den = cols[num][idx].sum(1), cols[den][idx].sum(1)
        ok = s_den > 0
        r = s_num[ok] / s_den[ok]
        out[name] = (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), int((~ok).sum()))
    return out


def _token_stats(gens: list[dict]) -> dict:
    outs = np.array([g["out"] for g in gens], float)
    ins = np.array([g["in"] for g in gens], float)
    return {
        "n": len(gens),
        "max_tokens_hits": sum(g["stop_reason"] == "max_tokens" for g in gens),
        "stop_reasons": dict(Counter(g["stop_reason"] for g in gens)),
        "caps": dict(Counter(g["cap"] for g in gens)),
        "out_p50": float(np.percentile(outs, 50)), "out_p95": float(np.percentile(outs, 95)),
        "out_p99": float(np.percentile(outs, 99)), "out_max": int(outs.max()),
        "out_mean": float(outs.mean()), "in_mean": float(ins.mean()),
    }


def report_exp1b(rows: list[dict]) -> dict:
    n = len(rows)
    by_batch = Counter((r["batch"], r["epoch"]) for r in rows)
    qs_by_batch = {b: len({r["qid"] for r in rows if r["batch"] == b}) for b in EXP1B}
    flips = [r for r in rows if r["influenced"]]
    unf = [r for r in flips if r["unfaithful"]]
    flip_qs = sorted({r["qid"] for r in flips})

    y, nn = _per_question(rows, "influenced")
    u, f = _per_question(rows, "unfaithful", "influenced")
    cr_flip = cluster_robust(y, nn)
    cr_unf = cluster_robust(u, f, only_nonempty=True)
    cr_unf_all = cluster_robust(u, f)
    boot = cluster_bootstrap({"flip": y, "inst": nn, "unf": u, "flips": f},
                             {"flip_rate": ("flip", "inst"), "unfaithful_frac": ("unf", "flips")})
    # Stratified by batch (batch 1 has 3 epochs per question, batch 2 has 2) as a check.
    strat = {}
    for b in EXP1B:
        br = [r for r in rows if r["batch"] == b]
        strat[b] = (_per_question(br, "influenced"), _per_question(br, "unfaithful", "influenced"))
    rng = np.random.default_rng(1)
    sf, su = [], []
    for _ in range(10_000):
        tf = ti = tu = tfl = 0.0
        for (yy, ii), (uu, ff) in strat.values():
            k = rng.integers(0, len(ii), len(ii))
            tf += yy[k].sum(); ti += ii[k].sum(); tu += uu[k].sum(); tfl += ff[k].sum()
        sf.append(tf / ti); su.append(tu / tfl)
    strat_ci = {"flip_rate": tuple(np.percentile(sf, [2.5, 97.5])),
                "unfaithful_frac": tuple(np.percentile(su, [2.5, 97.5]))}

    b1_flips = [r for r in flips if r["batch"] == "batch1"]
    out = {
        "instances": n, "by_batch_epoch": {f"{k[0]} e{k[1]}": v for k, v in sorted(by_batch.items())},
        "questions_by_batch": qs_by_batch,
        "unbiased_acc": (sum(r["ub"] == r["correct"] for r in rows), n),
        "biased_acc": (sum(r["b"] == r["correct"] for r in rows), n),
        "flips": len(flips), "distinct_flip_questions": len(flip_qs),
        "unfaithful": len(unf), "mentioned_in_flips": sum(r["acknowledged"] for r in flips),
        "acknowledged_all": (sum(r["acknowledged"] for r in rows), n),
        "flip_rate_cluster_robust": cr_flip, "unfaithful_cluster_robust_46": cr_unf,
        "unfaithful_cluster_robust_all240": cr_unf_all,
        "bootstrap": boot, "bootstrap_stratified": strat_ci,
        "unfaithful_binomial_ci": _binom_ci(len(unf), len(flips)),
        "batch1_flips": len(b1_flips), "batch1_self_judge_unfaithful": sum(r["unfaithful"] for r in b1_flips),
        "flips_per_question": dict(Counter(Counter(r["qid"] for r in flips).values())),
    }
    return out


def _binom_ci(k: int, n: int) -> tuple[float, float, float]:
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return p, p - h, p + h


def report_caps_exp1b(rows: list[dict]) -> dict:
    out = {}
    for role in ("unbiased", "biased", "judge"):
        gens = [g for r in rows for g in r["gens"].get(role, [])]
        out[role] = _token_stats(gens)
    per_sample = Counter(tuple(sorted((k, len(v)) for k, v in r["gens"].items())) for r in rows)
    out["events_per_sample"] = {str(k): v for k, v in per_sample.items()}
    out["unparseable"] = {
        "unbiased": sum(r["ub"] is None for r in rows),
        "biased": sum(r["b"] is None for r in rows),
        "judge": sum(not g["text"].strip().upper().startswith(("YES", "NO"))
                     for r in rows for g in r["gens"].get("judge", [])),
    }
    out["stored_matches_event"] = sum(
        r["gens"]["unbiased"][-1]["text"] == r["ub_cot"] and r["gens"]["biased"][-1]["text"] == r["b_cot"]
        for r in rows)
    return out


def noise_baseline(rows: list[dict]) -> dict:
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    hit = sum(r["ub"] == r["hint"] for r in rows)
    num = den = 0
    for rs in by_q.values():
        for a in rs:
            for b in rs:
                if a is b:
                    continue
                den += 1
                num += (a["ub"] != a["hint"]) and (b["ub"] == b["hint"])
    hint_seen = {q for q, rs in by_q.items() if any(r["ub"] == r["hint"] for r in rs)}
    flips = [r for r in rows if r["influenced"]]
    unf = [r for r in flips if r["unfaithful"]]
    return {
        "unbiased_eq_hint": (hit, len(rows)),
        "pseudo_flip": (num, den),
        "real_flip": (len(flips), len(rows)),
        "questions_with_hint_in_unbiased": len(hint_seen),
        "flips_on_hint_seen_q": (sum(r["qid"] in hint_seen for r in flips), len(flips)),
        "unfaithful_on_hint_seen_q": (sum(r["qid"] in hint_seen for r in unf), len(unf)),
        "flip_questions_hint_seen": (len({r["qid"] for r in flips} & hint_seen),
                                     len({r["qid"] for r in flips})),
    }


def easy_flips() -> dict:
    out = {}
    for name, d in EASY.items():
        ms = [s.scores["faithfulness_scorer"].metadata for s in _samples(d)]
        out[name] = (sum(bool(m["influenced"]) for m in ms), len(ms),
                     sum(bool(m["acknowledged"]) for m in ms))
    return out


def write_flips(rows: list[dict], out: str) -> int:
    flip_qs = {r["qid"] for r in rows if r["influenced"]}
    records = []
    for _, data in EXP1B.values():
        for line in (ROOT / data).open():
            if line.strip():
                rec = json.loads(line)
                if rec["id"] in flip_qs:
                    records.append(rec)
    assert len(records) == len(flip_qs), "flip question missing from the data files"
    (ROOT / out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return len(records)


# --- Exp 2 -----------------------------------------------------------------

def report_exp2() -> dict:
    out = {}
    for name, d in EXP2.items():
        rows, base_gens, force_gens = [], [], []
        for s in _samples(d):
            m = s.scores["perturbation_scorer"].metadata
            evs = _model_events(s)
            for e in evs:
                (force_gens if e.config.max_tokens == FORCE_MAX_TOKENS else base_gens).append(_gen(e))
            rows.append({"qid": s.id, "epoch": s.epoch, "final": m["final_answer"],
                         "correct": m["correct_letter"], "t": m["truncated_answers"],
                         "post_hoc_0": bool(m["post_hoc_0"])})
        n = len(rows)
        not_at_0 = [r for r in rows if not r["post_hoc_0"]]
        # Resampling agreement: ordered pairs of distinct epochs of a question whose two
        # full-reasoning answers agree — what a 0% answer that merely re-drew the model's
        # usual answer would score.
        by_q = defaultdict(list)
        for r in rows:
            by_q[r["qid"]].append(r["final"])
        pairs = [(a, b) for v in by_q.values() for i, a in enumerate(v) for j, b in enumerate(v) if i != j]
        out[name] = {
            "instances": n, "questions": len({r["qid"] for r in rows}),
            "epochs": dict(Counter(Counter(r["qid"] for r in rows).values())),
            "baseline_acc": (sum(r["final"] == r["correct"] for r in rows), n),
            "match_at_0": (sum(r["post_hoc_0"] for r in rows), n),
            "resample_agree": (sum(a == b for a, b in pairs), len(pairs)),
            "curve": {f: sum(r["t"].get(f) == r["final"] for r in rows) for f in ("0.0", "0.2", "0.4", "0.6", "0.8")},
            "not_matched_at_0": sorted(Counter(r["qid"] for r in not_at_0).items()),
            "baseline_gens": _token_stats(base_gens),
            "forced_gens": _token_stats(force_gens),
            "unparseable_final": sum(r["final"] is None for r in rows),
            "unparseable_forced": sum(v is None for r in rows for v in r["t"].values()),
            "rows": rows,
        }
    return out


def _fmt(k, n):
    return f"{k}/{n} = {k / n:.3f}" if n else f"{k}/{n}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--write-flips", metavar="OUT", help="write the distinct flip questions here")
    p.add_argument("--json", metavar="OUT", help="also dump every number as JSON")
    args = p.parse_args()

    rows = load_exp1b()
    a2 = report_exp1b(rows)
    a3 = report_caps_exp1b(rows)
    a4 = noise_baseline(rows)
    easy = easy_flips()
    e2 = report_exp2()

    print("== Exp 1b ==")
    print(f"instances {a2['instances']}  {a2['by_batch_epoch']}  questions {a2['questions_by_batch']}")
    print(f"unbiased acc {_fmt(*a2['unbiased_acc'])}   biased acc {_fmt(*a2['biased_acc'])}")
    print(f"flips {a2['flips']}  distinct flip questions {a2['distinct_flip_questions']}  "
          f"flips/question histogram {a2['flips_per_question']}")
    print(f"unfaithful {a2['unfaithful']}/{a2['flips']}  hint mentioned in {a2['mentioned_in_flips']}/{a2['flips']} flips"
          f"  acknowledged (all) {_fmt(*a2['acknowledged_all'])}")
    cr = a2["flip_rate_cluster_robust"]
    print(f"flip rate {cr['p']:.4f}  binomial SE {cr['se_binomial']:.4f}  cluster SE {cr['se']:.4f}  "
          f"95% CI [{cr['ci'][0]:.4f}, {cr['ci'][1]:.4f}]  deff CR1 {cr['deff_cr1']:.3f} CR0 {cr['deff_cr0']:.3f}")
    for k in ("unfaithful_cluster_robust_46", "unfaithful_cluster_robust_all240"):
        c = a2[k]
        print(f"unfaithful frac ({k}) {c['p']:.4f}  G={c['G']}  cluster SE {c['se']:.4f}  "
              f"95% CI [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}]  deff CR1 {c['deff_cr1']:.3f}")
    print(f"unfaithful binomial 95% CI {a2['unfaithful_binomial_ci']}")
    print(f"cluster bootstrap (10k, questions): {a2['bootstrap']}")
    print(f"stratified-by-batch bootstrap: {a2['bootstrap_stratified']}")
    print(f"batch-1 flips {a2['batch1_flips']}, self-judge unfaithful {a2['batch1_self_judge_unfaithful']}")
    print(f"easy benchmark (flips, instances, acknowledged): {easy}")

    print("\n== A3: caps (Exp 1b) ==")
    for role in ("unbiased", "biased", "judge"):
        print(role, a3[role])
    print("events per sample:", a3["events_per_sample"])
    print("unparseable:", a3["unparseable"], " stored completion == event output:", a3["stored_matches_event"])

    print("\n== A4: noise baseline ==")
    for k, v in a4.items():
        print(k, v if not isinstance(v, tuple) else _fmt(*v))

    print("\n== Exp 2 ==")
    for name, r in e2.items():
        print(f"{name}: n={r['instances']} q={r['questions']} epochs={r['epochs']} "
              f"acc {_fmt(*r['baseline_acc'])} match@0 {_fmt(*r['match_at_0'])} "
              f"resampling agreement {_fmt(*r['resample_agree'])} curve {r['curve']}")
        print(f"   not matched at 0%: {r['not_matched_at_0']}")
        print(f"   baseline gens {r['baseline_gens']}")
        print(f"   forced gens {r['forced_gens']}")
        print(f"   unparseable final {r['unparseable_final']}  forced {r['unparseable_forced']}")
    for name, qid in (("Opus — hard (20 Q)", "hard_mod7"), ("Opus — reasoning (24 Q)", "rs_book")):
        print(f"\n{qid}:")
        for r in sorted((r for r in e2[name]["rows"] if r["qid"] == qid), key=lambda r: r["epoch"]):
            print(f"  epoch {r['epoch']}: 0% -> {r['t'].get('0.0')}  full -> {r['final']}  "
                  f"(correct {r['correct']})  curve {r['t']}")

    if args.write_flips:
        print(f"\nwrote {write_flips(rows, args.write_flips)} flip questions to {args.write_flips}")
    if args.json:
        for r in e2.values():
            r.pop("rows")
        dump = {"exp1b": a2, "caps_exp1b": a3, "noise": a4, "easy": easy, "exp2": e2}
        Path(args.json).write_text(json.dumps(dump, indent=1, default=str))


if __name__ == "__main__":
    main()
