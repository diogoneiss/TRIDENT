"""C1's readers (docs/tickets/c1-split/): the smoke, the primary, the anchor and the token geometry.

    uv run --python 3.11 python -B scripts/experiments/c1_report.py --smoke     # cut 2: B against BT bit for bit
    uv run --python 3.11 python -B scripts/experiments/c1_report.py --primary   # cut 3: S − T, G2, MCAR step
    uv run --python 3.11 python -B scripts/experiments/c1_report.py --anchor    # S(normalized) − S(embedding)
    uv run --python 3.11 python -B scripts/experiments/c1_report.py --cosine    # [MASK]_j · [NULL]_j in S

Cells come from both machines' state directories (medusa1's copied under ``<state>/medusa``); S's decision-mode
cells are E124's A1 cells (``c15_report.readouts``). Macro-F1 is recomputed from each fold's test probabilities,
and log loss is the unweighted one. The primary's interval is the plain t interval over the 15 paired
replicates (14 df), the protocol's choice; the Nadeau–Bengio-corrected one (variance x (1/15 + 0.2/0.7)) is
printed beside as a sensitivity read. Per-file verdicts need the interval off 0 and the three seed means
agreeing in sign; ±0.010 macro-F1 is the equivalence band and the relevance label, never the bar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c1_run as run  # noqa: E402
from scripts.experiments import c15_report as c15  # noqa: E402

ROOTS = (run.STATE, run.STATE / "medusa") if run.HOST == "gorgona8" else (run.STATE,)
MARGIN = 0.010
SEEDS = run.SEEDS


def readouts(arm: str, variant: str, seed: int, smoke: bool = False) -> list[dict[str, Any]] | None:
    if arm == "S" and not smoke:
        return c15.readouts("A1", variant, seed)
    for root in ROOTS:
        d = (root / "smoke" if smoke else root) / arm / f"{variant}_s{seed}" / "metrics"
        if (d / f"{variant}_metrics.csv").exists() and len(list(d.glob("c15_readout_fold_*.pt"))) == 5:
            return [torch.load(d / f"c15_readout_fold_{k}.pt", weights_only=False) for k in range(1, 6)]
    return None


def paired(a: str, b: str, variant: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Per replicate (seed-major, then fold), macro-F1(a) − macro-F1(b) and log loss(b) − log loss(a): both favour a when positive."""
    f1, ll = [], []
    for seed in SEEDS:
        x, y = readouts(a, variant, seed), readouts(b, variant, seed)
        if x is None or y is None:
            return None
        (fa, la), (fb, lb) = c15.scores(x), c15.scores(y)
        f1.append(fa - fb)
        ll.append(lb - la)
    return np.concatenate(f1), np.concatenate(ll)


def verdict(d: np.ndarray, margin: float | None) -> str:
    """The fork's rule: better/worse if the plain interval is off 0 and the seed means agree; equivalent inside the band."""
    m, lo, hi = c15.interval(d, False)
    seeds = [d[k * 5:(k + 1) * 5].mean() for k in range(3)]
    if lo > 0 and all(s > 0 for s in seeds):
        return "better"
    if hi < 0 and all(s < 0 for s in seeds):
        return "worse"
    if margin is not None and lo > -margin and hi < margin:
        return "equivalent"
    return "no difference"


def line(name: str, d: np.ndarray, margin: float | None) -> str:
    m, lo, hi = c15.interval(d, False)
    _, nlo, nhi = c15.interval(d)
    seeds = ", ".join(f"s{s} {d[k * 5:(k + 1) * 5].mean():+.4f}" for k, s in enumerate(SEEDS))
    return f"{name} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (NB [{nlo:+.4f}, {nhi:+.4f}]; {seeds}) -> {verdict(d, margin)}"


def pooled_status(d: np.ndarray) -> tuple[str, str]:
    m, lo, hi = c15.interval(d, False)
    if lo > 0:
        return "Supported", "relevant" if m >= MARGIN else "real but small"
    if hi < 0:
        return "Reversed", "relevant" if m <= -MARGIN else "real but small"
    if lo > -MARGIN and hi < MARGIN:
        return "Null", "inside the band"
    return "Inconclusive", "interval reaches beyond the band"


def primary() -> None:
    print("## C1 primary (E131): S − T on the four MNAR 60nan files, 15 paired replicates each; plain 95% t intervals, "
          "Nadeau–Bengio beside; G2 = S − B; the MCAR files as the fixed-sequence second step")
    blocks: dict[str, dict[str, np.ndarray]] = {}
    for label, files in (("MNAR", run.MNAR), ("MCAR", run.MCAR)):
        print(f"\n### {label}")
        f1s, lls, g2s = {}, {}, {}
        for v in files:
            st, sb = paired("S", "T", v), paired("S", "B", v)
            if st is None or sb is None:
                print(f"- {v}: missing\n\nPrimary: incomplete")
                return
            f1s[v], lls[v], g2s[v] = st[0], st[1], sb[0]
            print(f"- {v}: F1 {line('S − T', st[0], MARGIN)}; LL(T) − LL(S) {line('', st[1], None)}; "
                  f"S − B F1 {c15.interval(sb[0], False)[0]:+.4f}")
        pooled = np.mean(list(f1s.values()), axis=0)
        m, lo, hi = c15.interval(pooled, False)
        _, nlo, nhi = c15.interval(pooled)
        status, note = pooled_status(pooled)
        flags = [f"{v} {verdict(f1s[v], MARGIN)}" for v in files if verdict(f1s[v], MARGIN) in ("better", "worse")]
        g2 = np.mean(list(g2s.values()), axis=0)
        gm, glo, ghi = c15.interval(g2, False)
        g2_pass = glo > 0 or ghi < 0
        print(f"\nPooled S − T ({label}): {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (NB [{nlo:+.4f}, {nhi:+.4f}]) -> {status} ({note}); "
              f"flags: {', '.join(flags) or 'none'}")
        print(f"G2 S − B ({label}): {gm:+.4f} [{glo:+.4f}, {ghi:+.4f}] -> {'passes' if g2_pass else 'fails'}")
        pooled_ll = np.mean(list(lls.values()), axis=0)
        print(f"Pooled LL(T) − LL(S) ({label}): {line('', pooled_ll, None)}")
        blocks[label] = {"pooled": pooled, "g2": g2, **f1s}
        if label == "MNAR":
            g1_path = run.STATE / "g1.json"
            g1 = json.loads(g1_path.read_text()) if g1_path.exists() else None
            g1_text = "not run" if g1 is None else ("passes" if g1["passed"] else "fails")
            if status == "Null" and (g1 is None or not g1["passed"] or not g2_pass):
                status = "Inconclusive"
                note = f"Null-type result with a gate failing (G1 {g1_text}, G2 {'passes' if g2_pass else 'fails'})"
            print(f"\nPrimary (C1, MNAR): {status} ({note}); G1 {g1_text}")
    print("\n### Mechanism interaction, per table: (S − T) MNAR − (S − T) MCAR, paired by replicate")
    for b in run.TABLES:
        d = blocks["MNAR"][f"{b}_probit_60nan"] - blocks["MCAR"][f"{b}_60nan"]
        m, lo, hi = c15.interval(d, False)
        print(f"- {b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def anchor() -> None:
    print("## Anchor (descriptive): S(embedding_normalized) − S(embedding) on the four MNAR files, 15 replicates")
    ds = []
    for v in run.MNAR:
        d = paired("S", "ANCHOR", v)
        if d is None:
            print(f"- {v}: missing")
            return
        ds.append(d[0])
        print(f"- {v}: F1 {line('', d[0], None)}; LL(embedding) − LL(normalized) {c15.interval(d[1], False)[0]:+.4f}")
    print(f"Pooled: {line('', np.mean(ds, axis=0), None)}")


def smoke() -> None:
    print("## Smoke (E130, biodeg_probit_60nan, seed 42, deterministic): integrity check 1, B against BT per fold")
    b, bt = readouts("B", run.SMOKE_VARIANT, 42, smoke=True), readouts("BT", run.SMOKE_VARIANT, 42, smoke=True)
    if b is None or bt is None:
        print("- B or BT missing")
        return
    worst = max(float(np.abs(x["test_probabilities"] - y["test_probabilities"]).max()) for x, y in zip(b, bt))
    same_state = all(all(torch.equal(x["state_dict"][k], y["state_dict"][k]) for k in x["state_dict"]) for x, y in zip(b, bt))
    print(f"- largest |P(B) − P(BT)| over the test rows of 5 folds: {worst:.2e}; kept states identical: {same_state}")
    print(f"- integrity check 1: {'passes' if worst == 0.0 and same_state else 'FAILS'}")
    for arm in ("S", "T"):
        r = readouts(arm, run.SMOKE_VARIANT, 42, smoke=True)
        if r is not None:
            f1, ll = c15.scores(r)
            print(f"- {arm}: macro-F1 per fold {np.round(f1, 4).tolist()}, log loss {np.round(ll, 4).tolist()} (no contrast is read)")


def cosines(tokens: dict[str, torch.Tensor], prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Per column cos([MASK]_j, [NULL]_j), and the cross-column cos([NULL]_j, [NULL]_j') for j < j'."""
    keys = sorted(k[len(prefix):-5] for k in tokens if k.startswith(prefix) and k.endswith("_mask"))
    mask = torch.stack([tokens[f"{prefix}{k}_mask"] for k in keys]).float()
    null = torch.stack([tokens[f"{prefix}{k}_null"] for k in keys]).float()
    within = torch.nn.functional.cosine_similarity(mask, null, dim=1).numpy()
    n = torch.nn.functional.normalize(null, dim=1)
    cross = (n @ n.t()).numpy()[np.triu_indices(len(keys), 1)]
    return within, cross


def _fmt(cs: list[tuple[np.ndarray, np.ndarray]]) -> str:
    if not cs:
        return "missing"
    return (f"within {np.median([np.median(w) for w, _ in cs]):+.3f}, cross {np.median([np.median(c) for _, c in cs]):+.3f} "
            f"({len(cs)} folds)")


def cosine() -> None:
    print("## Token geometry in S: median cos([MASK]_j, [NULL]_j) per file; cross-column cos([NULL]_j, [NULL]_j') beside; "
          f"reference SD at initialisation 1/sqrt(128) = {1 / np.sqrt(128):.3f}; 'tied by the model' at median >= 0.9")
    for v in run.MNAR + run.MCAR:
        after_pre = []
        for root in ROOTS:
            d = root / "S_GEOM" / f"{v}_s42" / "metrics"
            after_pre += [torch.load(p, weights_only=False) for p in sorted(d.glob("c1_tokens_pretrained_fold_*.pt"))]
        after_ft = [f["state_dict"] for s in SEEDS for f in (readouts("S", v, s) or [])]
        pre = [cosines(t) for t in after_pre]
        ft = [cosines(t, "embedder.special_embeddings.") for t in after_ft]
        print(f"- {v}: after pre-training {_fmt(pre)}; after fine-tuning {_fmt(ft)}")


if __name__ == "__main__":
    readers = {"--primary": primary, "--anchor": anchor, "--smoke": smoke, "--cosine": cosine}
    if len(sys.argv) != 2 or sys.argv[1] not in readers:
        sys.exit(__doc__)
    readers[sys.argv[1]]()
