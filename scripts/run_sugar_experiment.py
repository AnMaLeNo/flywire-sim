"""Fait tourner le cerveau : stimulation des neurones gustatifs « sucre » (GRN) du côté gauche,
puis on regarde qui s'active dans le cerveau, et en particulier le motoneurone MN9 (extension du
proboscis) ipsi- et contralatéral.

Les GRN sucre sont identifiés via les labels communautaires officiels de Codex
(« Putative Sugar Gustatory Receptor Neuron (GRN) », Kristin Scott Lab) et MN9 via le label
« Motor neuron 9; MN9 ».
"""
import argparse
import time

import numpy as np
import pandas as pd

from flywire_sim import data
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.network import Network

ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=float, default=100.0, help="fréquence des GRN sucre (Hz)")
ap.add_argument("--duration", type=float, default=1000.0, help="durée simulée (ms)")
ap.add_argument("--side", default="left")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--top", type=int, default=30)
args = ap.parse_args()

net = Network.load(data.PROCESSED / "network_783.npz")
table = data.neuron_table().set_index("root_id").reindex(net.root_ids)

sugar = data.root_ids_with_label(r"Sugar Gustatory Receptor Neuron", side=args.side)
sugar_idx = net.index_of(sugar)
mn9_idx = {s: int(net.index_of([rid])[0]) for s, rid in data.MN9.items()}
print(f"GRN sucre ({args.side}) : {len(sugar_idx)} neurones ; MN9 : {data.MN9}")

sim = LIFNetwork(net.W, LIFParams(), seed=args.seed)
t0 = time.time()
res = sim.run(args.duration, [Stimulus(sugar_idx, rate_hz=args.rate)], progress=True)
print(f"simulation {args.duration:.0f} ms de {net.n:,} neurones en {time.time()-t0:.1f}s ; "
      f"{len(res.spike_times):,} spikes")

rates = res.rates_hz()
active = np.flatnonzero(rates > 0)
print(f"neurones actifs : {len(active):,}")
for s, i in mn9_idx.items():
    print(f"MN9 {s:5s} : {rates[i]:6.1f} Hz")

order = active[np.argsort(-rates[active])]
rows = []
for i in order[: args.top]:
    r = table.iloc[i]
    rows.append({
        "root_id": net.root_ids[i], "rate_hz": round(float(rates[i]), 1), "nt": net.nt_type[i],
        "super_class": r.super_class, "class": r["class"], "sub_class": r.sub_class,
        "side": r.side, "type": r.primary_type, "name": r["name"],
        "stimulated": i in set(sugar_idx.tolist()),
    })
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(df.to_string(index=False))

data.RESULTS.mkdir(exist_ok=True)
tag = f"sugar_{args.side}_{int(args.rate)}Hz_seed{args.seed}"
np.savez_compressed(data.RESULTS / f"{tag}.npz", spike_times=res.spike_times, spike_neurons=res.spike_neurons,
                    root_ids=net.root_ids, rates=rates)
out_csv = data.RESULTS / f"{tag}_rates.csv"
pd.DataFrame({"root_id": net.root_ids[active], "rate_hz": rates[active]}).merge(
    table.reset_index()[["root_id", "nt_type", "super_class", "class", "sub_class", "side", "primary_type", "name"]],
    on="root_id").sort_values("rate_hz", ascending=False).to_csv(out_csv, index=False)
print(f"-> {out_csv}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    sel = order[:200]
    pos = {int(i): k for k, i in enumerate(sel)}
    m = np.isin(res.spike_neurons, sel)
    ys = np.array([pos[int(i)] for i in res.spike_neurons[m]])
    ax[0].scatter(res.spike_times[m], ys, s=1, c="k")
    ax[0].set_ylabel("200 neurones les plus actifs (triés)")
    ax[0].set_title(f"FlyWire v783 – LIF – GRN sucre {args.side} @ {args.rate:.0f} Hz")
    for s, i in mn9_idx.items():
        mm = res.spike_neurons == i
        ax[1].eventplot(res.spike_times[mm], lineoffsets=(0 if s == "left" else 1), colors="C0" if s == "left" else "C3")
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["MN9 gauche", "MN9 droit"]); ax[1].set_xlabel("temps (ms)")
    fig.tight_layout(); fig.savefig(data.RESULTS / f"{tag}.png", dpi=120)
    print(f"-> {data.RESULTS / f'{tag}.png'}")
except ImportError as e:  # matplotlib optionnel
    print("figure non générée :", e)
