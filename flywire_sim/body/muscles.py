"""Interface motoneurones -> muscles MuJoCo.

Chaque motoneurone de patte (MANC dans le réseau hybride, mêmes noms que le BANC) porte le nom du muscle qu'il
innerve (`Primary Cell Type`, atlas Azevedo et al. 2024 / Lesser et al. 2024), sa patte et son côté. On le relie
à l'actionneur `<côté><patte>_<muscle>` du modèle. Chaque MN est une unité motrice dont la force par spike, la
saturation et la cinétique de secousse dérivent de son volume officiel (motor_units.py : lois ajustées sur les
MN lents / intermédiaires / rapides d'Azevedo et al. 2020) :
    spike_i           -> u_i += 1, y_i += 1 ;  u relaxe en τd_i, y en τr_i ;  a_i = (u_i - y_i) / pic (1 par secousse)
    couple_i (µN·mm)  = T1_i · (1 - exp(-a_i / A_i)) / (1 - exp(-1 / A_i))   (un spike isolé -> T1_i)
    ctrl[actionneur]  = Σ_i couple_i / couple max de l'actionneur (gear), borné à 1
Les MN du « long tendon muscle » (fléchisseur des tarses/griffes) activent aussi l'adhésion tarsale, à hauteur
de leur fraction de saturation.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import pandas as pd

from ..size import volume_nm3
from . import motor_units as mu
from .model import LEG_NAME, LEGS, SIDE


@dataclass
class MotorMap:
    mn_idx: np.ndarray        # indices réseau des motoneurones de patte reliés
    act_idx: np.ndarray       # actionneur MuJoCo correspondant (même longueur)
    mn_leg: np.ndarray        # patte ('lf', ...) de chaque MN relié
    mn_volume: np.ndarray     # volume officiel (nm³) de chaque MN relié -> unité motrice (motor_units.py)
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
    mn_idx = np.array(mn_idx, dtype=np.int64)
    vol = volume_nm3(n)[mn_idx] if mn_idx.size else np.empty(0)
    assert np.isfinite(vol).all() and (vol > 0).all(), "chaque MN de patte relié doit avoir un volume officiel"
    return MotorMap(mn_idx, np.array(act_idx, dtype=np.int64), np.array(legs), vol,
                    n.iloc[missing][["root_id", "body_part", "side", "cell_type", "function"]])


class Muscles:
    def __init__(self, model: mujoco.MjModel, mmap: MotorMap, dt_ms: float, adhesion_from_ltm: bool = True):
        self.model, self.mmap = model, mmap
        n = mmap.mn_idx.size
        v = mmap.mn_volume
        self.t1 = mu.torque_per_spike(v)
        self.ratio = mu.twitch_tetanus_ratio(v)
        self.tet = mu.tetanic_torque(v)
        tr, td = mu.twitch_tau_ms(v)
        self.decay_r, self.decay_d = np.exp(-dt_ms / tr), np.exp(-dt_ms / td)
        t_peak = np.log(td / tr) * tr * td / (td - tr)
        self.peak = np.exp(-t_peak / td) - np.exp(-t_peak / tr)
        self.u, self.y = np.zeros(n), np.zeros(n)
        self.torque = np.zeros(n)                      # couple courant par unité motrice (µN·mm)
        self.act = np.zeros(model.nu, dtype=np.float64)   # ctrl par actionneur (fraction du couple max)
        self.tmax = np.abs(model.actuator_gear[:, 0]).astype(np.float64)
        self.tmax[self.tmax == 0] = 1.0
        self.adhesion = {}
        if adhesion_from_ltm:
            for leg in LEGS:
                ltm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_long_tendon_muscle")
                adh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_adhesion")
                if ltm >= 0 and adh >= 0:
                    self.adhesion[ltm] = adh
        # correspondance rapide indice réseau -> unité motrice
        self.lookup = np.full(int(mmap.mn_idx.max()) + 1 if n else 1, -1, dtype=np.int64)
        self.lookup[mmap.mn_idx] = np.arange(n)

    def tetanic_torque(self) -> np.ndarray:
        """Couple maximal (µN·mm) par unité motrice."""
        return self.tet

    def step(self, spiking: np.ndarray, data: mujoco.MjData) -> None:
        """À appeler à chaque pas du cerveau avec les indices des neurones qui ont tiré."""
        self.u *= self.decay_d
        self.y *= self.decay_r
        if spiking.size:
            s = spiking[spiking < self.lookup.size]
            i = self.lookup[s]
            i = i[i >= 0]
            if i.size:
                self.u[i] += 1.0
                self.y[i] += 1.0
        a = (self.u - self.y) / self.peak
        self.torque[:] = self.tet * (1.0 - np.exp(-a / self.ratio))
        self.act[:] = 0.0
        np.add.at(self.act, self.mmap.act_idx, self.torque)
        self.act /= self.tmax
        for ltm, adh in self.adhesion.items():
            self.act[adh] = self.act[ltm]
        data.ctrl[:] = np.clip(self.act, 0.0, 1.0)
