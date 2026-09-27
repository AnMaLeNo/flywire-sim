"""Freins physiologiques (adaptation, dépression synaptique) sur le BANC : MDN doit recruter les MN
des pattes à des taux plausibles (< 150 Hz) sans emballement du reste ; sucre labellum -> pas de MN pattes."""
import itertools

import numpy as np

from flywire_sim import banc, data
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.network import Network

net = Network.load(data.PROCESSED / "network_banc888_min5.npz")
n = banc.load_neurons().set_index("root_id").reindex(net.root_ids)
for c in ["function", "body_part", "cell_type", "labels", "side", "super_class"]:
    n[c] = n[c].fillna("")
leg_mn = (n.super_class.eq("motor") & n.body_part.str.contains("leg")).to_numpy()
mdn = np.flatnonzero(n.cell_type.str.fullmatch("MDN").to_numpy())
sugar = np.flatnonzero((n.function.str.contains("sugar") & (n.body_part == "labellum") & (n.side == "left")).to_numpy())
for U, b in itertools.product((0.0, 0.2, 0.4), (0.0, 1.0, 3.0)):
    for name, idx in (("MDN", mdn), ("sucre", sugar)):
        res = LIFNetwork(net.W, LIFParams(std_U=U, adapt_b=b), seed=0).run(500.0, [Stimulus(idx, 100)])
        r = res.rates_hz(); lm = r[leg_mn]
        h, _ = np.histogram(res.spike_times, bins=np.arange(0, 501, 100))
        print(f"U={U:.1f} b={b:.0f} {name:6s}: spikes={len(res.spike_times):9,} actifs={(r>0).sum():6d} max={r.max():5.0f}Hz "
              f"MN pattes actifs={(lm>0).sum():3d} moy={lm[lm>0].mean() if (lm>0).any() else 0:4.0f}Hz  spikes/100ms={h.tolist()}", flush=True)
