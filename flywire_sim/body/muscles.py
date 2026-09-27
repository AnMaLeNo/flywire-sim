"""Interface motoneurones -> muscles MuJoCo.

Chaque motoneurone de patte (MANC dans le réseau hybride, mêmes noms que le BANC) porte le nom du muscle qu'il
innerve (`Primary Cell Type`, atlas Azevedo et al. 2024 / Lesser et al. 2024), sa patte et son côté. On le relie
à l'actionneur `<côté><patte>_<muscle>` du modèle. Un spike de MN produit une secousse (twitch) : la force
musculaire est modélisée par une activation en [0, 1] qui saute de `per_spike` à chaque spike et
décroît avec `tau_ms` (dynamique calcium/pont actine-myosine simplifiée, ~20-40 ms chez l'insecte).
Les MN du « long tendon muscle » (fléchisseur des tarses/griffes) activent aussi l'adhésion tarsale.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import pandas as pd

from .model import LEG_NAME, LEGS, SIDE


@dataclass
class MotorMap:
    mn_idx: np.ndarray        # indices réseau des motoneurones de patte reliés
    act_idx: np.ndarray       # actionneur MuJoCo correspondant (même longueur)
    mn_leg: np.ndarray        # patte ('lf', ...) de chaque MN relié
    unmapped: pd.DataFrame    # MN de patte sans actionneur (types absents du modèle)


def build_motor_map(model: mujoco.MjModel, neurons: pd.DataFrame) -> MotorMap:
    """`neurons` : table des neurones alignée sur les indices du réseau (une ligne par neurone, dans l'ordre)."""
    n = neurons.reset_index(drop=True)
    is_mn = n.super_class.eq("motor") & n.body_part.isin(LEG_NAME.values())
    inv_leg = {v: k for k, v in LEG_NAME.items()}
    inv_side = {v: k for k, v in SIDE.items()}
    mn_idx, act_idx, legs, missing = [], [], [], []
    for i in np.flatnonzero(is_mn.to_numpy()):
        row = n.iloc[i]
        if row.side not in inv_side:
            missing.append(i)
            continue
        leg = inv_side[row.side] + inv_leg[row.body_part]
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{row.cell_type}")
        if a < 0:
            missing.append(i)
            continue
        mn_idx.append(i)
        act_idx.append(a)
        legs.append(leg)
    return MotorMap(np.array(mn_idx, dtype=np.int64), np.array(act_idx, dtype=np.int64), np.array(legs),
                    n.iloc[missing][["root_id", "body_part", "side", "cell_type", "function"]])


class Muscles:
    def __init__(self, model: mujoco.MjModel, mmap: MotorMap, dt_ms: float,
                 per_spike: float = 0.05, tau_ms: float = 30.0, adhesion_from_ltm: bool = True):
        self.model, self.mmap = model, mmap
        self.act = np.zeros(model.nu, dtype=np.float64)
        self.decay = np.exp(-dt_ms / tau_ms)
        self.per_spike = per_spike
        self.adhesion = {}
        if adhesion_from_ltm:
            for leg in LEGS:
                ltm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_long_tendon_muscle")
                adh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_adhesion")
                if ltm >= 0 and adh >= 0:
                    self.adhesion[ltm] = adh
        # correspondance rapide indice réseau -> actionneur
        self.lookup = np.full(int(mmap.mn_idx.max()) + 1 if mmap.mn_idx.size else 1, -1, dtype=np.int64)
        self.lookup[mmap.mn_idx] = mmap.act_idx

    def step(self, spiking: np.ndarray, data: mujoco.MjData) -> None:
        """À appeler à chaque pas du cerveau avec les indices des neurones qui ont tiré."""
        self.act *= self.decay
        if spiking.size:
            s = spiking[spiking < self.lookup.size]
            a = self.lookup[s]
            a = a[a >= 0]
            if a.size:
                np.add.at(self.act, a, self.per_spike)
        for ltm, adh in self.adhesion.items():
            self.act[adh] = self.act[ltm]
        data.ctrl[:] = np.clip(self.act, 0.0, 1.0)
