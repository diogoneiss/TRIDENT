"""Loss curves of the c1_diag_e132 runs: small multiples, one row per table, pre-training and fine-tuning; S and B."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
data = json.load(open(sys.argv[1]))
tables = ["biodeg", "kc2", "vehicle", "spambase"]
S1, S2, TXT, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
fig, axes = plt.subplots(len(tables), 2, figsize=(11, 3.0 * len(tables)), facecolor="#fcfcfb")
summary = []
for i, t in enumerate(tables):
    v = f"{t}_probit_60nan"
    for j, stage in enumerate(("pretrain", "finetune")):
        ax = axes[i, j]; ax.set_facecolor("#fcfcfb")
        for arm, color in (("S", S1), ("B", S2)):
            d = data.get(f"{arm}|{v}")
            if d is None or not d[f"cv/{stage}/train_loss/mean"]:
                continue
            tr, va = np.asarray(d[f"cv/{stage}/train_loss/mean"]), np.asarray(d[f"cv/{stage}/val_loss/mean"])
            x = np.arange(len(tr))
            ax.plot(x, tr, color=color, lw=2, label=f"{arm} train")
            ax.plot(x, va, color=color, lw=2, ls="--", label=f"{arm} validation")
            lo, hi = np.asarray(d[f"cv/{stage}/val_loss/ci95_lower"]), np.asarray(d[f"cv/{stage}/val_loss/ci95_upper"])
            if len(lo) == len(va):
                ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
            k = int(va.argmin())
            ax.plot([k], [va[k]], "o", ms=8, color=color, mec="#fcfcfb", mew=2)
            if stage == "finetune":
                summary.append(f"{arm} {v}: fine-tune val loss min {va.min():.3f} at epoch {k}, last {va[-1]:.3f} (+{va[-1]-va.min():.3f}); train last {tr[-1]:.3f}")
            else:
                summary.append(f"{arm} {v}: pre-train val loss first {va[0]:.3f}, min {va.min():.3f} at {k}, last {va[-1]:.3f}; train last {tr[-1]:.3f}")
        ax.set_title(f"{t} · {'pre-training (300 epochs)' if stage == 'pretrain' else 'fine-tuning (150 epochs)'}", loc="left", fontsize=10, color=TXT)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("epoch", color=MUTED, fontsize=8); ax.set_ylabel("loss (5-fold mean, 95% band on validation)", color=MUTED, fontsize=8)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, ncol=2)
fig.suptitle("E132 diagnostic · seed 42, MNAR probit 60nan · S (pre-trained) and B (no pre-training); dot = lowest validation loss", x=0.01, ha="left", fontsize=11, color=TXT)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(sys.argv[2], dpi=130)
print("\n".join(summary))
