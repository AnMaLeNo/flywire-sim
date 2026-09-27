"""Boucle corps fermée (FAFB + MANC) avec unités motrices : qui tire, quelle classe, quel couple, quel rythme.

Usage : MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/mn_classes_loop.py [--duration 1000] [--dn DNg100]
Pour le repos et pour la stimulation DNg100 : taux des MN de patte par classe (lent / intermédiaire / rapide,
motor_units.py), fraction du couple tétanique produite par actionneur, hauteur du thorax, appuis par patte et
période dominante des levers (rythmicité), alternance gauche/droite.
"""
import argparse
import os
from dataclasses import replace

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import pandas as pd

from flywire_sim import banc
from flywire_sim.body import motor_units as mu
from flywire_sim.body.model import LEGS
from flywire_sim.body.sim import BodyBrainSim

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=1000.0)
ap.add_argument("--rate", type=float, default=50.0)
ap.add_argument("--dn", nargs="+", default=["DNg100"])
ap.add_argument("--w", type=float, default=banc.CALIBRATED_V2.w_syn)
ap.add_argument("--gamma", type=float, default=banc.CALIBRATED_V2.size_norm)
ap.add_argument("--conditions", nargs="+", default=["rest", "dn"])
args = ap.parse_args()


def rhythm(contact: np.ndarray, dt_ms: float) -> str:
    """Période dominante (ms) des transitions d'appui par patte, par autocorrélation ; '-' si < 3 levers."""
    out = []
    for c in contact.T:
        b = (c > 0).astype(float)
        if np.abs(np.diff(b)).sum() < 6:
            out.append("-")
            continue
        b -= b.mean()
        ac = np.correlate(b, b, "full")[b.size - 1:]
        ac /= ac[0]
        lo = int(30 / dt_ms)
        k = lo + int(np.argmax(ac[lo:int(500 / dt_ms)]))
        out.append(f"{k * dt_ms:.0f}ms(r={ac[k]:.2f})")
    return " ".join(f"{leg}:{s}" for leg, s in zip(LEGS, out))


for cond in args.conditions:
    sim = BodyBrainSim(replace(banc.CALIBRATED_V2, w_syn=args.w, size_norm=args.gamma), brain="fafb")
    n = sim.neurons
    stim = None if cond == "rest" else sim.find(super_class="descending", cell_type=args.dn)
    tr = sim.run(args.duration, stim_idx=stim, stim_rate_hz=args.rate, spike_log=True, record_every_ms=1.0)
    spikes = np.concatenate(sim.spike_neurons) if sim.spike_neurons else np.empty(0, np.int64)
    counts = np.bincount(spikes, minlength=len(n))
    rates = counts / (args.duration / 1000.0)
    mn = sim.mmap.mn_idx
    cls = mu.mn_class(sim.mmap.mn_volume)
    df = pd.DataFrame({"classe": cls, "rate": rates[mn], "leg": sim.mmap.mn_leg,
                       "muscle": n.cell_type.to_numpy()[mn], "T1": sim.muscles.t1, "tet": sim.muscles.tet})
    print(f"\n=== {cond} ({args.duration:.0f} ms, DN {args.dn if stim is not None else '-'} à {args.rate} Hz)")
    g = df.groupby("classe").agg(n=("rate", "size"), actifs=("rate", lambda r: int((r > 0).sum())),
                                 taux_actifs=("rate", lambda r: r[r > 0].mean() if (r > 0).any() else 0.0),
                                 taux_max=("rate", "max"))
    print("MN de patte par classe :\n" + g.round(1).to_string())
    top = df.sort_values("rate", ascending=False).head(8)
    print("MN les plus actifs :\n" + top.round(2).to_string(index=False))
    ctrl = np.array(tr.ctrl)
    names = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(sim.model.nu)]
    act = pd.DataFrame({"actionneur": names, "moy": ctrl.mean(0), "max": ctrl.max(0)})
    act["muscle"] = act.actionneur.str.split("_", n=1).str[1]
    print("fraction du couple tétanique par muscle (moyenne des six pattes) :\n"
          + act.groupby("muscle")[["moy", "max"]].mean().sort_values("moy", ascending=False).round(3).to_string())
    z = np.array(tr.thorax_pos)[:, 2]
    print(f"hauteur du thorax : début {z[0]:.2f} mm, min {z.min():.2f}, fin {z[-1]:.2f} ; déplacement x "
          f"{np.array(tr.thorax_pos)[-1, 0] - np.array(tr.thorax_pos)[0, 0]:.2f} mm")
    con = np.array(tr.contact)
    duty = (con > 0).mean(0)
    lifts = np.diff((con > 0).astype(int), axis=0).clip(min=0).sum(0)
    print("appui :", {leg: round(float(d), 2) for leg, d in zip(LEGS, duty)}, "; levers :", dict(zip(LEGS, lifts.tolist())))
    print("période des levers :", rhythm(con, 1.0))
    same = [np.corrcoef(con[:, i] > 0, con[:, j] > 0)[0, 1] if (con[:, i] > 0).std() * (con[:, j] > 0).std() > 0 else np.nan
            for i, j in ((0, 3), (1, 4), (2, 5))]
    print("corrélation d'appui G/D (avant, milieu, arrière ; tripode < 0) :", np.round(same, 2).tolist())
