"""Calibration ciblée du lobe antennaire (docs/calibration.md § 2e passe) : la mouche posée au sol, tous
les capteurs actifs (ORN spontanés ~8 Hz, hygro/thermo toniques, proprioception), on mesure les taux des
étages ORN -> eLN/iLN -> PN -> KC -> MBON -> DN -> MN pour différentes valeurs des deux mécanismes incertains
(dépression des terminaisons ORN, gain électrique eLN -> PN/eLN). Aucune connexion n'est modifiée.

  python scripts/banc_al_gain.py --ms 500 --params shiu --orn-u 0 0.5 --eln-gain 1 0.1
"""
import argparse
import itertools
import os
import sys
from dataclasses import replace

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flywire_sim import banc, data
from flywire_sim.body.environment import Environment, OdorSource
from flywire_sim.body.sim import BodyBrainSim
from flywire_sim.lif import LIFParams

PARAMS = {"shiu": banc.CALIBRATED_V2, "calibrated": banc.CALIBRATED, "w1": LIFParams(w_syn=1.0, size_norm=0.5)}

# groupes de capteurs pour les ablations (--senses) : préfixes des clés de canaux de Senses._channels()
SENSE_GROUPS = {
    "orn": ("orn_",),
    "hygrothermo": ("dry", "humid", "hygro_cooling", "heating", "cooling", "evaporation"),
    "oxygen": ("oxygenation",),
    "jo": ("jo_",),
    "bristle": ("bristle_", "neck", "wheelers_organ"),
    "labellum": ("labellum_",),
    "legs": tuple(f"{leg}:" for leg in ("lf", "lm", "lh", "rf", "rm", "rh")),
}


def enabled_keys(sim: BodyBrainSim, groups: list[str]) -> set[str]:
    prefixes = tuple(p for g in groups for p in SENSE_GROUPS[g])
    return {k for k, _ in sim.senses._channels() if k.startswith(prefixes)}


def report(sim: BodyBrainSim, ms: float, top: int = 0) -> str:
    sp = np.concatenate(sim.spike_neurons) if sim.spike_neurons else np.empty(0, dtype=int)
    n = sim.neurons
    pops = banc.antennal_lobe_populations(n, sim.net.sign)
    masks = {k: np.isin(np.arange(len(n)), v) for k, v in pops.items()}
    masks["KC"] = (n.cls == "kenyon_cell").to_numpy()
    masks["MBON"] = (n.cls == "mushroom_body_output_neuron").to_numpy()
    masks["LHON"] = (n.cls == "lateral_horn_output_neuron").to_numpy()
    masks["DN"] = (n.super_class == "descending").to_numpy()
    masks["MN patte"] = sim.is_leg_mn
    # voie odorante suivie : ORN « levure » (canaux capteurs) et PN qui en reçoivent >= 5 synapses
    yeast = np.concatenate([ch.neurons for k, ch in sim.senses.body.items() if "_yeasty_" in k])
    to_pn = np.asarray(sim.net.W[:, yeast].sum(axis=1)).ravel()
    masks["ORN levure"] = np.isin(np.arange(len(n)), yeast)
    masks["PN levure"] = masks["PN"] & (to_pn >= 5)
    counts = np.bincount(sp, minlength=len(n))
    t = np.concatenate(sim.spike_times) if sim.spike_times else np.empty(0)
    bins = np.bincount((t // 100).astype(int), minlength=int(ms // 100))[: int(ms // 100)] * 10
    out = [f"total {counts.sum() / (ms / 1000):.0f} spk/s (par 100 ms : {' '.join(f'{b / 1000:.0f}k' for b in bins)})"]
    sp_t = np.concatenate(sim.spike_neurons) if sim.spike_neurons else np.empty(0, dtype=int)
    pn_y = np.flatnonzero(masks["PN levure"])
    if pn_y.size:
        sel = np.isin(sp_t, pn_y)
        pb = np.bincount((t[sel] // 100).astype(int), minlength=int(ms // 100))[: int(ms // 100)] * 10 / pn_y.size
        out.append("PN levure par 100 ms : " + " ".join(f"{b:.0f}" for b in pb) + " Hz")
    for k, m in masks.items():
        r = counts[m] / (ms / 1000)
        out.append(f"{k} {r.mean():.1f} Hz ({(r > 0).mean() * 100:.0f}% actifs)")
    line = " | ".join(out)
    if top:
        tot = max(counts.sum(), 1)
        by = pd.Series(counts).groupby(n.cls.fillna(n.super_class).fillna("?").to_numpy()).sum()
        by = by.sort_values(ascending=False).head(top)
        line += "\n   top populations (spk/s, % du total) : " + ", ".join(
            f"{k} {v / (ms / 1000):.0f} ({100 * v / tot:.0f}%)" for k, v in by.items())
        ct = pd.Series(counts).groupby(n.cell_type.fillna("?").to_numpy()).sum().sort_values(ascending=False).head(top)
        line += "\n   top cell_type : " + ", ".join(f"{k} {v / (ms / 1000):.0f} ({100 * v / tot:.0f}%)" for k, v in ct.items())
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=500)
    ap.add_argument("--params", nargs="+", default=["shiu"])
    ap.add_argument("--orn-u", nargs="+", type=float, default=[0.0, banc.ORN_STD_U])
    ap.add_argument("--eln-gain", nargs="+", type=float, default=[1.0, banc.ELN_ELECTRICAL_GAIN])
    ap.add_argument("--dn", nargs="*", default=None, help="types de DN stimulés à --rate Hz (ex. DNg100)")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--odor", action="store_true", help="source d'odeur de levure à gauche de la tête")
    ap.add_argument("--top", type=int, default=0, help="liste les N populations les plus actives")
    ap.add_argument("--tau-rec", type=float, default=banc.ORN_TAU_REC, help="récupération de la dépression ORN (ms)")
    ap.add_argument("--senses", nargs="*", default=None,
                    help=f"groupes de capteurs actifs ({', '.join(SENSE_GROUPS)} ; vide = aucun) ; défaut : tous")
    ap.add_argument("--no-completeness", action="store_true", help="sans correction de complétude BANC")
    ap.add_argument("--out", default="results/al_gain.txt")
    args = ap.parse_args()
    env = None
    if args.odor:
        env = Environment(odor_sources=[OdorSource(np.array([1.0, 2.0, 0.5]), {"yeasty": 1.0}, radius=2.0)])
    lines = []
    for name, u, g in itertools.product(args.params, args.orn_u, args.eln_gain):
        sim = BodyBrainSim(params=replace(PARAMS[name], tau_rec=args.tau_rec), orn_std_u=u, eln_gain=g, env=env,
                           completeness_correction=not args.no_completeness)
        if args.senses is not None:
            sim.senses.enabled = enabled_keys(sim, args.senses)
        stim = sim.find(super_class="descending", cell_type=args.dn) if args.dn else None
        sim.run(args.ms, stim_idx=stim, stim_rate_hz=args.rate, spike_log=True)
        tag = f"[{name} U_ORN={u} tau_rec={args.tau_rec:g} gain_eLN={g}" + (" sans-complétude" if args.no_completeness else "") + (
            f" capteurs={'+'.join(args.senses) or 'aucun'}" if args.senses is not None else "") + (f" DN={'+'.join(args.dn)}@{args.rate:g}Hz" if args.dn else "") + (" odeur" if args.odor else "") + "] "
        line = tag + report(sim, args.ms, args.top)
        print(line, flush=True)
        lines.append(line)
    path = data.ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
