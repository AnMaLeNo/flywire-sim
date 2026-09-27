"""Balayage rapide : seuil de synapses x poids synaptique, sur le BANC. Cherche un régime où une commande
descendante (MDN) recrute les MN des pattes sans emballement du réseau entier."""
import numpy as np

from flywire_sim import banc, data
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.network import Network

nrn = banc.load_neurons().set_index("root_id")
for min_syn in (5, 10):
    path = data.PROCESSED / f"network_banc888_min{min_syn}.npz"
    if not path.exists():
        banc.build(min_synapses=min_syn).save(path)
    net = Network.load(path)
    n = nrn.reindex(net.root_ids)
    for c in ["function", "body_part", "cell_type", "labels", "side", "super_class"]:
        n[c] = n[c].fillna("")
    leg_mn = (n.super_class.eq("motor") & n.body_part.str.contains("leg")).to_numpy()
    mdn = np.flatnonzero(n.cell_type.str.fullmatch("MDN").to_numpy())
    sugar = np.flatnonzero((n.function.str.contains("sugar") & (n.body_part == "labellum") & (n.side == "left")).to_numpy())
    mn9 = np.flatnonzero(n.labels.str.contains("MN9", regex=False).to_numpy())
    for w in (0.275, 0.15, 0.08):
        for name, idx in (("MDN", mdn), ("sucre", sugar)):
            res = LIFNetwork(net.W, LIFParams(w_syn=w), seed=0).run(500.0, [Stimulus(idx, 100)])
            r = res.rates_hz()
            print(f"min_syn={min_syn:2d} w={w:.3f} {name:6s}: spikes={len(res.spike_times):9,} actifs={(r>0).sum():6d} "
                  f"MN pattes actifs={(r[leg_mn]>0).sum():3d} (moy {r[leg_mn][r[leg_mn]>0].mean() if (r[leg_mn]>0).any() else 0:5.0f} Hz) MN9={r[mn9].round(0).tolist()}", flush=True)
