"""Calibration de la dynamique BANC (cerveau seul, sans corps) : commandes descendantes de marche ->
motoneurones de patte. Balayage des paramètres LIF et mesure d'indicateurs comparés à la littérature :
  - fraction de MN de patte actifs et taux moyen (Azevedo 2020 : MN lents ~30 Hz au repos, rapides muets) ;
  - rythmicité de la sortie motrice dans la bande de pas 5-20 Hz (Mendes 2013, Pugliese 2025 : 7-15 Hz) ;
  - alternance fléchisseur/extenseur du tibia (corrélation négative attendue) ;
  - taux des DN stimulés et des interneurones du VNC (pas d'emballement).
"""
import argparse
import itertools

import numpy as np

from flywire_sim import banc, data
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.network import Network

WALK_DN = ["DNp09", "DNa01", "DNa02", "DNa03", "DNa04", "DNa05", "DNa06", "DNa07", "DNb01", "DNb02", "DNb05",
           "DNg13", "DNg100", "DNp10", "DNp42", "DNa10", "DNa11", "DNa14", "DNa15", "DNp25"]

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=600.0)
ap.add_argument("--w", type=float, nargs="+", default=[0.275])
ap.add_argument("--gamma", type=float, nargs="+", default=[0.0, 0.5, 1.0])
ap.add_argument("--adapt", type=float, nargs="+", default=[0.0])
ap.add_argument("--std", type=float, nargs="+", default=[0.0])
ap.add_argument("--drive", nargs="+", default=["DNg100", "walk"], help="DNg100 | DNb08 | walk (population)")
ap.add_argument("--mode", default="spike", help="spike (Poisson forcé) ou current")
ap.add_argument("--rate", type=float, default=50.0)
ap.add_argument("--amp", type=float, default=10.0, help="mV par événement en mode current")
ap.add_argument("--free-afferents", action="store_true", help="laisser les neurones sensoriels recevoir des entrées centrales")
args = ap.parse_args()

net = Network.load(data.PROCESSED / "network_banc888_min5.npz")
n = banc.load_neurons().set_index("root_id").reindex(net.root_ids)
W = net.W if args.free_afferents else banc.clamp_afferents(net.W, n)
for c in ["function", "body_part", "cell_type", "side", "super_class", "region"]:
    n[c] = n[c].fillna("")
dn = n.super_class.eq("descending")
leg_mn = (n.super_class.eq("motor") & n.body_part.str.contains("leg")).to_numpy()
vnc = n.super_class.eq("ventral_nerve_cord_intrinsic").to_numpy()
brain = n.super_class.isin(["central_brain_intrinsic", "optic_lobe_intrinsic", "visual_projection"]).to_numpy()
flex = (leg_mn & n.cell_type.eq("tibia_flexor")).to_numpy()
ext = (leg_mn & n.cell_type.eq("tibia_extensor")).to_numpy()
drives = {
    "DNg100": np.flatnonzero((dn & n.cell_type.eq("DNg100")).to_numpy()),
    "DNb08": np.flatnonzero((dn & n.cell_type.eq("DNb08")).to_numpy()),
    "walk": np.flatnonzero((dn & n.cell_type.isin(WALK_DN)).to_numpy()),
}


def pop_rate(res, mask, bin_ms=2.0):
    sel = mask[res.spike_neurons]
    edges = np.arange(0, res.duration_ms + bin_ms, bin_ms)
    return np.histogram(res.spike_times[sel], bins=edges)[0] / mask.sum() / (bin_ms / 1000)


def rhythm(x, bin_ms=2.0, skip_ms=100.0, band=(5.0, 20.0)):
    """Fréquence dominante et part de puissance dans la bande de pas (spectre du taux de population)."""
    x = x[int(skip_ms / bin_ms):]
    x = x - x.mean()
    if not x.any():
        return 0.0, 0.0
    ps = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, bin_ms / 1000)
    ps[0] = 0
    inband = (f >= band[0]) & (f <= band[1])
    return float(f[inband][ps[inband].argmax()]), float(ps[inband].sum() / ps[f <= 100].sum())


def alternation(res):
    a, b = pop_rate(res, flex), pop_rate(res, ext)
    a, b = a[50:], b[50:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


print(f"réseau {net.W.shape[0]} neurones ; MN pattes {leg_mn.sum()} ; " + " ; ".join(f"{k}: {v.size} DN" for k, v in drives.items()))
for drive, w, gamma, b, u in itertools.product(args.drive, args.w, args.gamma, args.adapt, args.std):
    idx = drives[drive]
    params = LIFParams(w_syn=w, size_norm=gamma, adapt_b=b, std_U=u)
    res = LIFNetwork(W, params, seed=0).run(args.duration, [Stimulus(idx, args.rate, mode=args.mode, amplitude_mv=args.amp)])
    r = res.rates_hz()
    lm = r[leg_mn]
    active = lm > 0
    f_peak, band_power = rhythm(pop_rate(res, leg_mn))
    hist = np.histogram(res.spike_times, bins=np.arange(0, args.duration + 1, 100))[0]
    print(f"{drive:7s} w={w} γ={gamma} b={b} U={u} | DN stim {r[idx].mean():.0f}Hz | MN actifs {active.sum()}/{leg_mn.sum()} "
          f"moy {lm[active].mean() if active.any() else 0:.0f}Hz méd {np.median(lm[active]) if active.any() else 0:.0f}Hz max {lm.max():.0f}Hz | "
          f"VNC actifs {(r[vnc] > 0).sum()} moy {r[vnc][r[vnc] > 0].mean() if (r[vnc] > 0).any() else 0:.0f}Hz | cerveau actifs {(r[brain] > 0).sum()} | "
          f"rythme {f_peak:.1f}Hz ({band_power:.2f}) | corr flex/ext {alternation(res):+.2f} | spikes/100ms {hist.tolist()}", flush=True)
