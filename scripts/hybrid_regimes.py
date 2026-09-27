"""Élimination : d'où vient l'activité auto-entretenue du réseau hybride sous DNg100 ?

Réseau seul (pas de corps, pas de capteurs), CALIBRATED_V2, 500 ms :
  manc      : moelle MANC isolée, DNg100 (MANC) à 50 Hz
  hybrid    : cerveau BANC + moelle MANC, DNg100 à 50 Hz
  cut_an    : idem, sorties des ascendants vers le cerveau coupées (AN -> cerveau = 0)
  cut_dn_in : idem, entrées cérébrales des descendants coupées (cerveau -> DN = 0)
  rest      : hybride sans stimulation
Avec --top, liste les types de DN et d'AN les plus actifs et les DN qui alimentent les MN d'aile.
"""
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flywire_sim import banc, hybrid, manc
from flywire_sim.lif import LIFNetwork, Stimulus


def groups(n: pd.DataFrame) -> dict[str, np.ndarray]:
    sc, bp = n.super_class.fillna(""), n.body_part.fillna("")
    leg = bp.isin(["front_leg", "middle_leg", "hind_leg"])
    return {
        "DN": sc.eq("descending").to_numpy(),
        "AN": sc.str.startswith("ascending").to_numpy() | sc.eq("sensory_ascending").to_numpy(),
        "IN VNC": sc.eq("ventral_nerve_cord_intrinsic").to_numpy(),
        "MN patte": (sc.eq("motor") & leg).to_numpy(),
        "MN aile": (sc.eq("motor") & bp.eq("wing")).to_numpy(),
        "MN autre": (sc.eq("motor") & ~leg & ~bp.eq("wing")).to_numpy(),
        "cerveau": sc.isin(["central_brain_intrinsic", "optic_lobe_intrinsic", "visual_centrifugal",
                            "visual_projection", "sensory", "endocrine"]).to_numpy(),
    }


def report(tag: str, rates: np.ndarray, n: pd.DataFrame) -> None:
    line = [f"[{tag}] total {rates.sum():.0f} spk/s"]
    for k, m in groups(n).items():
        if m.any():
            r = rates[m]
            line.append(f"{k} {r.mean():.1f} Hz ({(r > 0).mean() * 100:.0f}% actifs, n={m.sum()})")
    print(" | ".join(line), flush=True)


def top_types(rates: np.ndarray, n: pd.DataFrame, mask: np.ndarray, k: int) -> str:
    s = pd.Series(rates[mask]).groupby(n.cell_type.fillna("?").to_numpy()[mask]).agg(["sum", "size"])
    s = s.sort_values("sum", ascending=False).head(k)
    return ", ".join(f"{t} {r['sum'] / r['size']:.0f} Hz x{int(r['size'])}" for t, r in s.iterrows())


def wing_drivers(W: sp.csc_matrix, rates: np.ndarray, n: pd.DataFrame, k: int) -> str:
    g = groups(n)
    wing = np.flatnonzero(g["MN aile"] & (rates > 0))
    if not wing.size:
        return "aucun MN d'aile actif"
    drive = np.asarray(W[wing].sum(axis=0)).ravel() * rates          # synapses signées x taux présynaptique
    out = []
    for name in ("DN", "IN VNC", "AN"):
        m = g[name] & (drive != 0)
        s = pd.Series(drive[m]).groupby(n.cell_type.fillna("?").to_numpy()[m]).sum().sort_values(ascending=False)
        out.append(f"{name} -> MN aile (excit. nette x Hz) : {s.head(k).round(0).to_dict()} ; part {name} "
                   f"{100 * drive[g[name]].sum() / drive[drive > 0].sum():.0f}%")
    return "\n   ".join(out)


def run(W: sp.csc_matrix, n: pd.DataFrame, stim: np.ndarray, ms: float, tag: str, top: int = 0) -> np.ndarray:
    net = LIFNetwork(W, banc.CALIBRATED_V2, seed=0)
    res = net.run(ms, [Stimulus(stim, 50.0)] if stim.size else [])
    rates = res.rates_hz()
    report(tag, rates, n)
    if top:
        g = groups(n)
        print("   top DN :", top_types(rates, n, g["DN"], top))
        print("   top AN :", top_types(rates, n, g["AN"], top))
        print("   top IN VNC :", top_types(rates, n, g["IN VNC"], top))
        print("   " + wing_drivers(W, rates, n, top), flush=True)
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=500)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--brain", default="banc", choices=["banc", "fafb"])
    a = ap.parse_args()
    want = set(a.only) if a.only else {"manc", "hybrid", "cut_an", "cut_dn_in", "rest"}

    if "manc" in want:
        mn, mnet = manc.load_neurons(), manc.build()
        mn = mn.set_index("root_id").loc[mnet.root_ids].reset_index()
        stim = np.flatnonzero(mn.cell_type.eq("DNg100").to_numpy())
        run(mnet.W, mn, stim, a.ms, f"manc DNg100 x{stim.size}", a.top)

    net, n = hybrid.load(brain=a.brain)
    n = n.reset_index(drop=True)
    g = groups(n)
    stim = np.flatnonzero(n.cell_type.fillna("").eq("DNg100").to_numpy())
    if "rest" in want:
        run(net.W, n, np.empty(0, dtype=np.int64), a.ms, "hybride repos")
    if "hybrid" in want:
        run(net.W, n, stim, a.ms, f"hybride DNg100 x{stim.size}", a.top)
    if "cut_an" in want:
        W = net.W.tolil(copy=True)
        rows, cols = np.flatnonzero(g["cerveau"] | g["DN"]), np.flatnonzero(g["AN"])
        W[np.ix_(rows, cols)] = 0
        run(W.tocsc(), n, stim, a.ms, "hybride DNg100, AN -> cerveau/DN coupé")
    if "cut_dn_in" in want:
        W = net.W.tolil(copy=True)
        rows, cols = np.flatnonzero(g["DN"]), np.flatnonzero(g["cerveau"] | g["AN"])
        W[np.ix_(rows, cols)] = 0
        run(W.tocsc(), n, stim, a.ms, "hybride DNg100, cerveau/AN -> DN coupé")


if __name__ == "__main__":
    main()
