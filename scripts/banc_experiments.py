"""BANC v888 : le même cerveau + le ganglion ventral. Vérifie (1) sucre -> MN9 comme dans FAFB,
(2) qu'une commande descendante connue (MDN = marche arrière, DNp09 = marche avant) recrute les
motoneurones des pattes dans le VNC : c'est le chemin cerveau -> moelle -> muscles qu'on branchera au corps."""
import argparse
import time

import numpy as np

from flywire_sim import banc, data
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.network import Network

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=1000.0)
ap.add_argument("--rate", type=float, default=100.0)
args = ap.parse_args()

path = data.PROCESSED / "network_banc888.npz"
if not path.exists():
    t0 = time.time(); net = banc.build(min_synapses=3); net.save(path)
    print(f"réseau BANC construit : {net.n:,} neurones, {net.W.nnz:,} connexions ({time.time()-t0:.0f}s)")
net = Network.load(path)
nrn = banc.load_neurons().set_index("root_id").reindex(net.root_ids)
for c in ["function", "body_part", "cell_type", "labels", "side", "super_class"]:
    nrn[c] = nrn[c].fillna("")

def sel(mask):
    return np.flatnonzero(mask.to_numpy())

sugar_L = sel(nrn.function.str.contains("sugar") & (nrn.body_part == "labellum") & (nrn.side == "left"))
mn9 = sel(nrn.labels.str.contains("MN9", regex=False) | nrn.cell_type.str.contains(r"\bMN9\b"))
mdn = sel(nrn.cell_type.str.fullmatch("MDN") | nrn.labels.str.contains(r"\bMDN\b|moonwalker", case=False))
dnp09 = sel(nrn.cell_type.str.fullmatch("DNp09"))
dna02 = sel(nrn.cell_type.str.fullmatch("DNa02"))
leg_mn = nrn.super_class.eq("motor") & nrn.body_part.str.contains("leg")
print(f"GRN sucre labellum gauche : {len(sugar_L)} | MN9 : {len(mn9)} | MDN : {len(mdn)} | DNp09 : {len(dnp09)} | DNa02 : {len(dna02)} | MN pattes : {leg_mn.sum()}")

def report(name, idxs):
    if len(idxs) == 0:
        print(f"\n### {name}: aucun neurone trouvé"); return
    t0 = time.time()
    res = LIFNetwork(net.W, LIFParams(), seed=0).run(args.duration, [Stimulus(idxs, args.rate)])
    r = res.rates_hz()
    print(f"\n### {name} ({len(idxs)} neurones @ {args.rate:.0f} Hz) : {len(res.spike_times):,} spikes, {(r>0).sum()} actifs, {time.time()-t0:.1f}s")
    if len(mn9): print("  MN9 :", " ".join(f"{nrn.side.iloc[i][0]}={r[i]:.0f}Hz" for i in mn9))
    act = nrn[(r > 0) & leg_mn.to_numpy()].assign(hz=r[(r > 0) & leg_mn.to_numpy()])
    if len(act):
        tab = act.groupby(["body_part", "side"]).agg(n=("hz", "size"), hz=("hz", "mean")).round(0)
        print("  MN pattes actifs :\n" + tab.to_string())
        print("  fonctions MN pattes :", act.function.str.replace(",leg_motor|leg_motor,", "", regex=True).value_counts().head(8).to_dict())
    else:
        print("  MN pattes : aucun actif")
    return r


report("GRN sucre labellum gauche", sugar_L)
report("MDN (marche arrière)", mdn)
report("DNp09 (marche avant)", dnp09)
report("DNa02 (virage)", dna02)
