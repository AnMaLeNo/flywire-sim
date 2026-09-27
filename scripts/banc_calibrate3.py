"""Balayage fin : poids synaptique x adaptation x dépression courte, BANC min5. Stimuli : MDN, DNp09, DNa02."""
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
vnc = n.super_class.eq("ventral_nerve_cord_intrinsic").to_numpy()
brain = n.super_class.isin(["central_brain_intrinsic", "optic_lobe_intrinsic", "visual_projection"]).to_numpy()
stims = {k: np.flatnonzero(n.cell_type.str.fullmatch(k).to_numpy()) for k in ("MDN", "DNp09", "DNa02")}
for w, b, (U, tr) in itertools.product((0.18, 0.21, 0.24), (0.0, 3.0), ((0.0, 500.0), (0.1, 150.0))):
    for name, idx in stims.items():
        res = LIFNetwork(net.W, LIFParams(w_syn=w, adapt_b=b, std_U=U, tau_rec=tr), seed=0).run(500.0, [Stimulus(idx, 100)])
        r = res.rates_hz(); lm = r[leg_mn]
        h, _ = np.histogram(res.spike_times, bins=np.arange(0, 501, 100))
        print(f"w={w:.2f} b={b:.0f} U={U:.1f}/{tr:.0f} {name:5s}: actifs={(r>0).sum():6d} (VNC {(r[vnc]>0).sum():5d}, cerveau {(r[brain]>0).sum():5d}) "
              f"MN pattes={(lm>0).sum():3d} moy={lm[lm>0].mean() if (lm>0).any() else 0:4.0f}Hz max={r.max():4.0f}Hz spikes/100ms={h.tolist()}", flush=True)
