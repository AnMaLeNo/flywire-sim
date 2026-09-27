"""Classes de motoneurones de patte (MANC) déduites du volume officiel, et couples par pool.

Usage : PYTHONPATH=. .venv/bin/python scripts/motor_units_report.py
Imprime, par muscle, le nombre d'unités lentes / intermédiaires / rapides, le couple par spike, le couple
tétanique du pool (moyenne des six pattes) — valeurs reportées dans LEG_MUSCLES (model.py) — et la
vérification du pool fléchisseur du tibia de la patte avant (Azevedo 2020 : ~100 µN au bout du tibia).
"""
import numpy as np
import pandas as pd

from flywire_sim import manc
from flywire_sim.body import motor_units as mu

m = manc.load_neurons()
leg = m[(m.super_class == "motor") & m.body_part.isin(["front_leg", "middle_leg", "hind_leg"])].copy()
v = leg["Volume (nm^3)"].astype(float).to_numpy()
leg["classe"] = mu.mn_class(v)
leg["T1"] = mu.torque_per_spike(v)
leg["tet"] = mu.tetanic_torque(v)
leg["v_rest"] = mu.rest_potential(v)
print(f"{len(leg)} MN de patte MANC ; classes : {leg.classe.value_counts().to_dict()}")
print(f"potentiel de repos : {leg.v_rest.min():.0f} à {leg.v_rest.max():.0f} mV ; "
      f"{(leg.v_rest > -50).sum()} MN au-dessus de -50 mV (lents, actifs au repos chez Azevedo)")

pools = leg.groupby(["cell_type", "body_part", "side"]).agg(n=("T1", "size"), tet=("tet", "sum"),
                                                              t1max=("T1", "max")).reset_index()
per_muscle = pools.groupby("cell_type").agg(n=("n", "mean"), couple_max=("tet", "mean"),
                                            t1max=("t1max", "mean")).sort_values("couple_max", ascending=False)
cls = leg.pivot_table(index="cell_type", columns="classe", values="T1", aggfunc="size", fill_value=0)
per_muscle = per_muscle.join(cls / 6.0)   # par patte
named = per_muscle[~per_muscle.index.str.startswith("MN")]
with pd.option_context("display.width", 160):
    print("\nPar muscle nommé (moyenne par patte) : n, couple max µN·mm (Σ tétanique), T1 max µN·mm, classes/patte")
    print(named.round(2).to_string())

front = leg[(leg.body_part == "front_leg") & leg.cell_type.isin(["tibia_flexor", "accessory_tibia_flexor"])]
for side, g in front.groupby("side"):
    print(f"\nfléchisseur tibia patte avant {side} : {len(g)} MN, classes {g.classe.value_counts().to_dict()}, "
          f"Σ tétanique {g.tet.sum():.1f} µN·mm = {g.tet.sum() / mu.TIBIA_MM:.0f} µN au bout du tibia "
          f"(mesuré ~100 µN)")
    print(g.sort_values("Volume (nm^3)", ascending=False)[["cell_type", "Volume (nm^3)", "classe", "T1", "tet", "v_rest"]]
          .assign(**{"Volume (nm^3)": lambda d: (d["Volume (nm^3)"].astype(float) / 1e11).round(1)}).round(3).to_string(index=False))
np.set_printoptions(precision=2)
