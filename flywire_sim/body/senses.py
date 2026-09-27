"""Interface capteurs MuJoCo -> neurones sensoriels de patte du BANC (spikes forcés, Poisson).

Populations (annotations officielles `Sub Class` / `Function` du BANC, Tuthill & Wilson 2016 pour la
physiologie) et grandeur physique encodée :
  hair_plate (joint_angle)        : angles des articulations thorax-coxa et coxa-trochanter
  claw chordotonal (position)     : angle fémur-tibia (moitié « flexion », moitié « extension »)
  hook chordotonal (direction)    : sens de la vitesse fémur-tibia (moitié flexion, moitié extension)
  club chordotonal (vibration)    : |accélération| du tibia (vibrations du substrat / de la patte)
  campaniform (mechanical_strain) : charge sur la patte (force au sol)
  bristle / taste_bristle tactile : contact des tarses
  taste_bristle gustatory sugar   : sucre sous les tarses (champ `sugar` de l'environnement)
Chaque neurone tire à `r_max` x (grandeur normalisée dans [0, 1]) Hz.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pandas as pd

from .model import LEG_NAME, LEGS, SIDE


@dataclass
class Channel:
    neurons: np.ndarray       # indices réseau
    r_max: float
    value: float = 0.0        # grandeur normalisée courante dans [0, 1]


@dataclass
class LegSenses:
    leg: str
    channels: dict = field(default_factory=dict)


def _pop(n: pd.DataFrame, sub_class: str, function: str | None = None) -> np.ndarray:
    m = n.sub_class.eq(sub_class)
    if function:
        m &= n.function.str.contains(function, regex=False)
    return np.flatnonzero(m.to_numpy())


def _split(idx: np.ndarray, k: int) -> list[np.ndarray]:
    return [a for a in np.array_split(idx, k)]


class Senses:
    def __init__(self, model: mujoco.MjModel, neurons: pd.DataFrame, dt_ms: float, seed: int = 0):
        n = neurons.reset_index(drop=True)
        for c in ("sub_class", "function", "body_part", "side"):
            n[c] = n[c].fillna("")
        self.model, self.dt_ms = model, dt_ms
        self.rng = np.random.default_rng(seed)
        self.legs: list[LegSenses] = []
        self.sensor = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i): model.sensor_adr[i] for i in range(model.nsensor)}
        for leg in LEGS:
            side, part = SIDE[leg[0]], LEG_NAME[leg[1]]
            sub = n[(n.side == side) & n.body_part.str.contains(part, regex=False)]
            ls = LegSenses(leg)
            hp = _split(_pop(sub, f"{part}_hair_plate_neuron"), 4)
            for name, idx in zip(("coxa_yaw_neg", "coxa_yaw_pos", "coxa_pitch_neg", "coxa_pitch_pos"), hp):
                ls.channels[f"hairplate_{name}"] = Channel(sub.index.to_numpy()[idx], 100.0)
            claw = _split(_pop(sub, f"{part}_claw_chordotonal_organ_neuron"), 2)
            ls.channels["claw_flex"] = Channel(sub.index.to_numpy()[claw[0]], 100.0)
            ls.channels["claw_ext"] = Channel(sub.index.to_numpy()[claw[1]], 100.0)
            hook = _split(_pop(sub, f"{part}_hook_chordotonal_organ_neuron"), 2)
            ls.channels["hook_flex"] = Channel(sub.index.to_numpy()[hook[0]], 150.0)
            ls.channels["hook_ext"] = Channel(sub.index.to_numpy()[hook[1]], 150.0)
            ls.channels["club"] = Channel(sub.index.to_numpy()[_pop(sub, f"{part}_club_chordotonal_organ_neuron")], 100.0)
            ls.channels["campaniform"] = Channel(sub.index.to_numpy()[_pop(sub, f"{part}_campaniform_sensillum_neuron")], 120.0)
            tactile = np.concatenate([_pop(sub, f"{part}_bristle_neuron", "tactile"),
                                      _pop(sub, f"{part}_taste_bristle_tactile_neuron")])
            ls.channels["tactile"] = Channel(sub.index.to_numpy()[tactile], 50.0)
            ls.channels["sugar"] = Channel(sub.index.to_numpy()[_pop(sub, f"{part}_taste_bristle_gustatory_neuron", "sugar")], 100.0)
            self.legs.append(ls)
        self.sugar = 0.0   # concentration de sucre normalisée sous les tarses (environnement)

    def _s(self, data: mujoco.MjData, name: str) -> float:
        return float(data.sensordata[self.sensor[name]])

    def read(self, data: mujoco.MjData) -> None:
        m = self.model
        for ls in self.legs:
            leg = ls.leg

            def jr(seg, ax, leg=leg):
                return m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{seg}_{ax}")]

            for seg, ax, key in (("coxa", "yaw", "coxa_yaw"), ("coxa", "pitch", "coxa_pitch")):
                lo, hi = jr(seg, ax)
                mid = 0.5 * (lo + hi)
                q = self._s(data, f"{leg}_{seg}_{ax}_pos")
                dev = (q - mid) / (0.5 * (hi - lo))
                ls.channels[f"hairplate_{key}_neg"].value = np.clip(-dev, 0, 1)
                ls.channels[f"hairplate_{key}_pos"].value = np.clip(dev, 0, 1)
            lo, hi = jr("tibia", "pitch")
            q = self._s(data, f"{leg}_tibia_pitch_pos")
            dev = (q - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
            ls.channels["claw_flex"].value = np.clip(dev, 0, 1)
            ls.channels["claw_ext"].value = np.clip(-dev, 0, 1)
            w = self._s(data, f"{leg}_tibia_pitch_vel") / 20.0          # rad/s, ~20 rad/s en marche rapide
            ls.channels["hook_flex"].value = np.clip(w, 0, 1)
            ls.channels["hook_ext"].value = np.clip(-w, 0, 1)
            adr = self.sensor[f"{leg}_load"]                              # force (µN) au bout du tarse
            load = float(np.linalg.norm(data.sensordata[adr:adr + 3]))
            ls.channels["campaniform"].value = np.clip(load / 5.0, 0, 1)   # ~ poids du corps / 2
            touch = sum(self._s(data, f"{leg}_tarsus{k}_contact") for k in range(1, 6))
            ls.channels["tactile"].value = 1.0 if touch > 0.01 else 0.0
            ls.channels["club"].value = np.clip(abs(w) * 0.2, 0, 1)
            ls.channels["sugar"].value = self.sugar if touch > 0.01 else 0.0

    def spikes(self) -> np.ndarray:
        """Neurones sensoriels forcés à tirer sur ce pas (Poisson à partir des grandeurs lues)."""
        out = []
        for ls in self.legs:
            for ch in ls.channels.values():
                if ch.value > 0 and ch.neurons.size:
                    p = ch.r_max * ch.value * self.dt_ms / 1000.0
                    hit = ch.neurons[self.rng.random(ch.neurons.size) < p]
                    if hit.size:
                        out.append(hit)
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([(ls.leg, k, ch.neurons.size, ch.r_max) for ls in self.legs for k, ch in ls.channels.items()],
                            columns=["leg", "channel", "n_neurons", "r_max_hz"])
