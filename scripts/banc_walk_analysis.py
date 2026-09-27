"""Analyse détaillée de la sortie motrice du BANC sous commande DNg100 (cerveau seul) : recrutement
par muscle / patte, ordre de recrutement vs taille (principe de taille, Azevedo 2020), spectre du
rythme (bande de pas 5-20 Hz, Mendes 2013 / Pugliese 2025) et phase fléchisseur/extenseur du tibia."""
import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flywire_sim import banc, data
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.network import Network

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=1000.0)
ap.add_argument("--w", type=float, default=0.275)
ap.add_argument("--gamma", type=float, default=0.0)
ap.add_argument("--adapt", type=float, default=0.0)
ap.add_argument("--rate", type=float, default=50.0)
ap.add_argument("--drive", default="DNg100")
ap.add_argument("--side", default="", help="stimuler un seul côté (left/right)")
ap.add_argument("--out", default="banc_walk")
args = ap.parse_args()

net = Network.load(data.PROCESSED / "network_banc888_min5.npz")
n = banc.load_neurons().set_index("root_id").reindex(net.root_ids)
for c in ["function", "body_part", "cell_type", "side", "super_class", "region"]:
    n[c] = n[c].fillna("")
W = banc.clamp_afferents(net.W, n)
n_in = np.asarray(abs(net.W).sum(axis=1)).ravel()
leg_mn = (n.super_class.eq("motor") & n.body_part.str.contains("leg")).to_numpy()
sel = n.super_class.eq("descending") & n.cell_type.eq(args.drive)
if args.side:
    sel &= n.side.eq(args.side)
idx = np.flatnonzero(sel.to_numpy())
print(f"{args.drive} x{idx.size} @ {args.rate} Hz ; w={args.w} γ={args.gamma} b={args.adapt} ; {args.duration:.0f} ms")
res = LIFNetwork(W, LIFParams(w_syn=args.w, size_norm=args.gamma, adapt_b=args.adapt), seed=0).run(
    args.duration, [Stimulus(idx, args.rate)])
r = res.rates_hz()
skip = 200.0  # régime transitoire écarté
late = res.spike_times >= skip
r_late = np.bincount(res.spike_neurons[late], minlength=r.size) / ((args.duration - skip) / 1000)

mn = n[leg_mn].assign(rate=r_late[leg_mn], n_in=n_in[leg_mn])
act = mn[mn.rate > 0]
print(f"MN pattes actifs (après {skip:.0f} ms) : {len(act)}/{len(mn)} ; taux moy {act.rate.mean():.1f} Hz, méd {act.rate.median():.1f}, max {act.rate.max():.0f}")
print("par muscle (actifs/total, taux moyen des actifs) :")
for ct, g in mn.groupby("cell_type"):
    a = g[g.rate > 0]
    print(f"  {ct:28s} {len(a):2d}/{len(g):2d}  {a.rate.mean() if len(a) else 0:5.1f} Hz")
print("par patte :", mn.assign(leg=mn.side.str[0] + mn.body_part.str[0]).groupby("leg").rate.agg(["mean", lambda s: (s > 0).sum()]).round(1).to_dict())
# principe de taille : les MN recrutés sont-ils les petits (peu d'entrées) de leur muscle ?
mn["rel_size"] = mn.n_in / mn.groupby(["cell_type", "body_part"]).n_in.transform("max")
print(f"taille relative (n_in / max du muscle) : recrutés {mn[mn.rate > 0].rel_size.mean():.2f} vs muets {mn[mn.rate == 0].rel_size.mean():.2f}")
print(f"corrélation taux ~ taille relative (Spearman) : {mn[['rate', 'rel_size']].corr('spearman').iloc[0, 1]:+.2f}")

bin_ms = 2.0
edges = np.arange(skip, args.duration + bin_ms, bin_ms)
fig, axes = plt.subplots(3, 1, figsize=(11, 9))
pop = np.histogram(res.spike_times[late & leg_mn[res.spike_neurons]], bins=edges)[0] / leg_mn.sum() / (bin_ms / 1000)
axes[0].plot(edges[:-1], pop, lw=0.8)
axes[0].set_title(f"taux de population des {leg_mn.sum()} MN de patte ({args.drive} @ {args.rate} Hz)")
axes[0].set_ylabel("Hz / MN")
x = pop - pop.mean()
ps = np.abs(np.fft.rfft(x)) ** 2
f = np.fft.rfftfreq(x.size, bin_ms / 1000)
band = (f >= 5) & (f <= 20)
print(f"rythme dominant 5-20 Hz : {f[band][ps[band].argmax()]:.1f} Hz ; part de puissance dans la bande : {ps[band].sum() / ps[(f > 0) & (f <= 100)].sum():.2f}")
axes[1].plot(f[f <= 60], ps[f <= 60] / ps[(f > 0)].max())
axes[1].axvspan(5, 20, color="orange", alpha=0.2)
axes[1].set_xlabel("Hz")
axes[1].set_title("spectre du taux de population (bande de pas 5-20 Hz en orange)")
for leg_bp, side, color in (("front_leg", "left", "C0"), ("front_leg", "right", "C1")):
    for ct, ls in (("tibia_flexor", "-"), ("tibia_extensor", "--")):
        m = (leg_mn & n.body_part.eq(leg_bp).to_numpy() & n.side.eq(side).to_numpy() & n.cell_type.eq(ct).to_numpy())
        if m.sum() == 0:
            continue
        h = np.histogram(res.spike_times[late & m[res.spike_neurons]], bins=np.arange(skip, args.duration + 10, 10))[0] / m.sum() / 0.01
        axes[2].plot(np.arange(skip, args.duration, 10)[: h.size], h, ls, color=color, lw=0.9, label=f"{side[0].upper()}1 {ct}")
axes[2].legend(fontsize=7, ncol=2)
axes[2].set_title("patte avant : fléchisseur (trait plein) vs extenseur (tirets) du tibia, 10 ms")
axes[2].set_xlabel("ms")
fig.tight_layout()
out = data.RESULTS / f"{args.out}.png"
fig.savefig(out, dpi=110)
print("figure ->", out)
