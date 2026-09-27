"""Diagnostic § 9.3 : réseau FAFB + MANC seul (sans capteurs), repos puis DNg100 0–1000 ms ; activité pendant / après.
Usage : hybrid_decay.py W GAMMA."""
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from hybrid_regimes import groups

from flywire_sim import banc, hybrid, size
from flywire_sim.lif import LIFNetwork, Stimulus

w, g = float(sys.argv[1]), float(sys.argv[2])
net, n = hybrid.load(brain="fafb")
sz = size.relative_size(n)
W = banc.clamp_afferents(net.W, n)
stim = np.flatnonzero(n.cell_type.eq("DNg100").to_numpy())
grp = groups(n); grp["KC"] = n.cell_type.fillna("").str.startswith("KC").to_numpy(); grp["lobe optique"] = n.super_class.eq("optic_lobe_intrinsic").to_numpy()
for tag, stims in (("repos", []), ("DNg100 0-1000", [Stimulus(stim, 50.0, t_stop=1000)])):
    res = LIFNetwork(W, replace(banc.CALIBRATED_V2, w_syn=w, size_norm=g), seed=0, size=sz).run(1500 if stims else 1000, stims)
    t, ids = res.spike_times, res.spike_neurons
    wins = (("0-1000", t < 1000, 1.0),) if not stims else (("stim 500-1000", (t >= 500) & (t < 1000), .5), ("post 1100-1500", t >= 1100, .4))
    for wt, m, dur in wins:
        r = np.bincount(ids[m], minlength=len(n)) / dur
        line = [f"[w={w} g={g} {tag} | {wt}] total {r.sum():.0f} spk/s"]
        for k, gm in grp.items():
            if gm.any(): line.append(f"{k} {r[gm].mean():.1f} Hz ({(r[gm] > 0).mean():.0%})")
        print(" | ".join(line), flush=True)
