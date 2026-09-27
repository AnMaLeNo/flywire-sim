"""Population de neurones descendants « marche » (types connus, Braun 2024 / Cheong 2024 / Bidaye) vs
un seul type : recrutement des MN des pattes dans le BANC sous freins physiologiques."""
import numpy as np

from flywire_sim import banc, data
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.network import Network

WALK_DN = ["DNp09", "DNa01", "DNa02", "DNa03", "DNa04", "DNa05", "DNa06", "DNa07", "DNb01", "DNb02", "DNb05",
           "DNg13", "DNg100", "DNp10", "DNp42", "BDN2", "oDN1", "DNa10", "DNa11", "DNa14", "DNa15", "DNp25"]

net = Network.load(data.PROCESSED / "network_banc888_min5.npz")
n = banc.load_neurons().set_index("root_id").reindex(net.root_ids)
for c in ["function", "body_part", "cell_type", "side", "super_class", "region"]:
    n[c] = n[c].fillna("")
dn = n.super_class.eq("descending")
print("régions de sortie des DN :", n.region[dn].value_counts().head(12).to_dict())
leg_mn = (n.super_class.eq("motor") & n.body_part.str.contains("leg")).to_numpy()
vnc = n.super_class.eq("ventral_nerve_cord_intrinsic").to_numpy()
brain = n.super_class.isin(["central_brain_intrinsic", "optic_lobe_intrinsic", "visual_projection"]).to_numpy()
walk = np.flatnonzero((dn & n.cell_type.isin(WALK_DN)).to_numpy())
print(f"DN marche trouvés : {len(walk)} ->", n.cell_type.iloc[walk].value_counts().to_dict())
leg_dn = np.flatnonzero((dn & n.region.str.contains("T1|T2|T3|LegNp|NM")).to_numpy())
print(f"DN projetant vers les neuromères des pattes : {len(leg_dn)}")
for label, idx in (("types-marche", walk), ("DN->pattes", leg_dn)):
    for w, b in ((0.21, 3.0), (0.24, 3.0), (0.275, 3.0)):
        for rate in (20, 50):
            res = LIFNetwork(net.W, LIFParams(w_syn=w, adapt_b=b), seed=0).run(500.0, [Stimulus(idx, rate)])
            r = res.rates_hz(); lm = r[leg_mn]
            h, _ = np.histogram(res.spike_times, bins=np.arange(0, 501, 100))
            print(f"{label} x{len(idx)} w={w} b={b:.0f} @{rate}Hz : actifs={(r>0).sum()} (VNC {(r[vnc]>0).sum()}, cerveau {(r[brain]>0).sum()}) "
                  f"MN pattes={(lm>0).sum()} moy={lm[lm>0].mean() if (lm>0).any() else 0:.0f}Hz max={r.max():.0f}Hz spikes/100ms={h.tolist()}", flush=True)
