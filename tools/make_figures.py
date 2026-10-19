"""Make the "Behind the scenes" figures from a real FastICA run on the example sounds.

Run from the project folder:   python3 tools/make_figures.py
Needs: numpy, matplotlib. Reads test/sources-16k.s16 (made by tools/make_audio.sh).
Writes PNGs + training-log.json into process/.

This is a line-by-line numpy port of fastICA() in ica.js (same random start, same
centering, whitening, update rule and stopping rule), run on the page's default 3x3 mix.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "process")
SR, DUR = 16000, 9
N = SR * DUR
NAMES = ["Kitchen", "Hall", "Classroom"]
MICS = ["Mic by the door", "Mic by the window", "Mic at the back"]
A = np.array([[1.0, 0.45, 0.6], [0.5, 1.0, 0.35], [0.4, 0.55, 0.9]])   # DEFAULT_MIX in ica.js
ACCENT = "#1f3a5f"
GAUSS_LOGCOSH = 0.3745672

def mulberry32(seed):                      # same random numbers as rng() in ica.js
    a = seed & 0xFFFFFFFF
    def imul(x, y): return (x * y) & 0xFFFFFFFF
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt

def jacobi_eig(S):                         # same Jacobi sweeps as jacobiEig() in ica.js
    A_ = np.array(S, float); n = len(A_); V = np.eye(n)
    for _ in range(100):
        off = sum(A_[p, q] ** 2 for p in range(n) for q in range(p + 1, n))
        if off < 1e-20: break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A_[p, q]) < 1e-30: continue
                theta = (A_[q, q] - A_[p, p]) / (2 * A_[p, q])
                tt = (1 if theta >= 0 else -1) / (abs(theta) + np.sqrt(theta * theta + 1))
                c = 1 / np.sqrt(tt * tt + 1); s = tt * c
                J = np.eye(n); J[p, p] = c; J[q, q] = c; J[p, q] = s; J[q, p] = -s
                A_ = J.T @ A_ @ J; V = V @ J
    return np.diag(A_).copy(), V

def sym_decorrelate(W):
    d, E = jacobi_eig(W @ W.T)
    return (E * (1 / np.sqrt(np.maximum(d, 1e-12)))) @ E.T @ W

def nongauss(y):                           # nonGaussianity() in ica.js
    a = np.abs(y); lc = a + np.log1p(np.exp(-2 * a)) - np.log(2)
    return (lc.mean() - GAUSS_LOGCOSH) ** 2

def corr(a, b): return float(np.corrcoef(a, b)[0, 1])

def main():
    raw = np.fromfile(os.path.join(ROOT, "test", "sources-16k.s16"), dtype="<i2").astype(float) / 32768
    S = raw.reshape(3, N)
    X = A @ S
    Xc = X - X.mean(axis=1, keepdims=True)                    # 1) move to the middle
    d, E = jacobi_eig(Xc @ Xc.T / N)                           # 2) make it round
    K = (E / np.sqrt(np.maximum(d, 1e-12))).T
    Z = K @ Xc
    rand = mulberry32(1)                                       # same start as the page's first "Separate"
    W = np.array([[rand() * 2 - 1 for _ in range(3)] for _ in range(3)])
    W = sym_decorrelate(W)
    log, snaps = [], {}
    def record(r, W, change):
        Y = W @ Z
        log.append({"round": r, "change": change, "independence": float(sum(nongauss(y) for y in Y)),
                    "abs_r": [[abs(corr(Y[i], S[j])) for j in range(3)] for i in range(3)]})
        snaps[r] = Y.copy()
    record(0, W, None)
    r = 0
    for r in range(1, 201):                                    # 3) turn the dial, round by round
        WZ = W @ Z
        G = np.tanh(WZ)
        Wn = (G @ Z.T) / N - (1 - G ** 2).mean(axis=1, keepdims=True) * W
        Wn = sym_decorrelate(Wn)
        change = float(np.max(np.abs(np.abs(np.sum(Wn * W, axis=1)) - 1)))
        W = Wn
        record(r, W, change)
        if change < 1e-6: break
    rounds = r
    Y = W @ Z
    abs_r = np.array(log[-1]["abs_r"])
    match = [int(np.argmax(abs_r[i])) for i in range(3)]
    # unmixing (W K) times mixing (A), with each sound scaled to the same loudness
    P = W @ K @ A @ np.diag(S.std(axis=1))
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#888",
                         "axes.spines.top": False, "axes.spines.right": False})

    # --- convergence ---
    rr = [e["round"] for e in log]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), dpi=150)
    ax[0].semilogy(rr[1:], [e["change"] for e in log[1:]], "o-", color=ACCENT)
    ax[0].axhline(1e-6, color="#8a2b2b", lw=1, ls="--"); ax[0].text(rr[-1], 1.6e-6, "stop here", ha="right", color="#8a2b2b", fontsize=9)
    ax[0].set_xlabel("round"); ax[0].set_title("How much the unmixing changed", fontsize=10)
    ax[1].plot(rr, [e["independence"] for e in log], "o-", color=ACCENT)
    ax[1].set_xlabel("round"); ax[1].set_title("Independence score (higher = more separate)", fontsize=10)
    for a in ax: a.set_xticks(rr)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "convergence.png")); plt.close(fig)

    # --- one output track over the rounds ---
    k = match.index(0) if 0 in match else 0          # the output that ends up as the Kitchen
    mid = max(1, rounds // 2)
    t = np.arange(N) / SR
    fig, ax = plt.subplots(3, 1, figsize=(8.4, 4.6), dpi=150, sharex=True)
    for a, rnd in zip(ax, [0, mid, rounds]):
        y = snaps[rnd][k]; rs = log[rnd]["abs_r"][k]
        a.plot(t, y, color=ACCENT, lw=0.4)
        a.set_ylabel(f"round {rnd}", rotation=0, ha="right", va="center")
        a.set_yticks([])
        a.set_title("match with  " + ",  ".join(f"{NAMES[j]} {rs[j]:.2f}" for j in range(3)), fontsize=9, loc="left")
    ax[-1].set_xlabel("seconds")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "track-snapshots.png")); plt.close(fig)

    # --- W . A heatmap ---
    fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=150)
    im = ax.imshow(P, cmap="RdBu_r", vmin=-1.1, vmax=1.1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, ("+0.00" if abs(P[i, j]) < 0.005 else f"{P[i, j]:+.2f}"), ha="center", va="center", color="white" if abs(P[i, j]) > 0.6 else "#1d2228", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"Output {i + 1}" for i in range(3)])
    ax.spines[:].set_visible(False)
    ax.set_xlabel("original sound"); ax.set_title("What the learned unmixing does to each sound", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "unmix-times-mix.png"), bbox_inches="tight"); plt.close(fig)

    summary = {"mix": A.tolist(), "mics": MICS, "sources": NAMES, "seed": 1, "rounds": rounds,
               "final_change": log[-1]["change"], "output_matches": [NAMES[m] for m in match],
               "output_abs_r": [float(abs_r[i, match[i]]) for i in range(3)],
               "unmix_times_mix": np.round(P, 4).tolist(), "snapshot_rounds": [0, mid, rounds],
               "snapshot_output": k + 1, "log": log}
    with open(os.path.join(OUT, "training-log.json"), "w") as f: json.dump(summary, f, indent=1)
    print(f"rounds={rounds} final change={log[-1]['change']:.2e}")
    for e in log: print(f"round {e['round']:2d} change={e['change'] if e['change'] is None else format(e['change'], '.3e')} independence={e['independence']:.5f}")
    print("outputs:", ", ".join(f"out{i + 1}={NAMES[match[i]]} |r|={abs_r[i, match[i]]:.4f}" for i in range(3)))
    print("unmix x mix:\n", np.round(P, 3))

if __name__ == "__main__":
    main()
