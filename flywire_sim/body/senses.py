"""Interface capteurs MuJoCo / environnement -> neurones sensoriels du BANC (spikes forcés, Poisson).

Chaque canal relie une population officielle du BANC (`Class`/`Sub Class`/`Function`/`Body Part`) à une
grandeur normalisée dans [0, 1] lue dans MuJoCo ou dans `Environment` ; le neurone tire à
r_spont + r_max x grandeur (Poisson), avec une adaptation optionnelle (soustraction d'une moyenne glissante).
Physiologie et hypothèses : docs/capteurs.md.

Pattes (Tuthill & Wilson 2016) : hair plates (angles thorax-coxa), claw (angle fémur-tibia), hook (sens de
la vitesse), club (vibration), campaniformes (charge), soies tactiles (contact), soies gustatives
(sucre, amer, eau, sel, phéromone de contact sous les tarses).
Tête / corps : soies tactiles de la tête, du thorax, de l'abdomen, des antennes et de la trompe ;
organe de Johnston A/B (vibration de l'antenne : son, Kamikouchi 2009) et C/D/E/F (déviation statique :
gravité, vent, Yorozu 2009) ; ORN antennaires et palpaux par classe d'odeur (Hallem & Carlson 2006) ;
neurones gustatifs du labelle et des taste pegs (Hiroi 2002) ; cellules « cooling »/« heating » phasiques
de l'ariste (Budelli 2019) ; cellules sèches/humides du sacculus (Enjin 2016, Knecht 2017) ; chordotonal
du cou et hair plate prosternal (angles du cou) ; organe de Wheeler (vibration du thorax) ; neurones
d'oxygénation abdominaux (hypoxie). Halteres, aile et tegula : silencieux (mouche sans ailes, au repos).
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pandas as pd

from .environment import ODOR_CLASSES, Environment
from .model import LEG_NAME, LEGS, SIDE

HEAD_BRISTLES = ("frontal", "frontoorbital", "interocellar", "interommatidial", "occipital_dorsal", "ocellar",
                 "orbital", "postocellar", "postorbital_dorsal", "postorbital_ventral", "vibrissa", "eye")
ORN_SPONT_HZ = 8.0            # taux spontané des ORN (Hallem & Carlson 2006 : ~8 Hz en moyenne, 0-30 selon le récepteur)
LEG_TASTES = ("sugar", "bitter", "water", "low_salt", "contact_pheromone")
LABELLUM_TASTES = ("sugar", "bitter", "water", "low_salt", "fatty_acids", "amino_acids", "heavy_metal",
                   "aversive", "contact_pheromone")
TASTE_RMAX = {"sugar": 100.0, "bitter": 60.0, "water": 60.0, "low_salt": 40.0, "fatty_acids": 40.0,
              "amino_acids": 40.0, "heavy_metal": 40.0, "aversive": 40.0, "contact_pheromone": 40.0}


@dataclass
class Channel:
    neurons: np.ndarray       # indices réseau
    r_max: float
    value: float = 0.0        # grandeur normalisée courante dans [0, 1]
    r_spont: float = 0.0      # taux spontané (Hz)
    tau_adapt_ms: float = 0.0 # 0 = pas d'adaptation ; sinon soustraction d'une moyenne glissante
    adapt_frac: float = 0.7   # part de la réponse soutenue qui s'adapte
    _mean: float = 0.0

    def rate(self, dt_ms: float) -> float:
        v = self.value
        if self.tau_adapt_ms > 0:
            self._mean += (v - self._mean) * dt_ms / self.tau_adapt_ms
            v = max(0.0, v - self.adapt_frac * self._mean)
        return self.r_spont + self.r_max * v


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
    def __init__(self, model: mujoco.MjModel, neurons: pd.DataFrame, dt_ms: float, seed: int = 0,
                 env: Environment | None = None):
        n = neurons.reset_index(drop=True)
        for c in ("cls", "sub_class", "function", "body_part", "side"):
            n[c] = n[c].fillna("")
        self.n = n
        self.model, self.dt_ms = model, dt_ms
        self.env = env or Environment()
        self.rng = np.random.default_rng(seed)
        self.legs: list[LegSenses] = []
        self.body: dict[str, Channel] = {}
        self.sensor = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i): model.sensor_adr[i] for i in range(model.nsensor)}
        self.site = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i): i for i in range(model.nsite)}
        self._build_legs(n)
        self._build_body(n)
        self.sugar = 0.0   # rétro-compatibilité : sucre uniforme sous les tarses (s'ajoute à l'environnement)
        self._temp_prev = {"l": None, "r": None}
        self._vib = {"l": 0.0, "r": 0.0, "thorax": 0.0}

    # ---- construction --------------------------------------------------------------------------
    def _build_legs(self, n: pd.DataFrame) -> None:
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
            ls.channels["tactile"] = Channel(sub.index.to_numpy()[tactile], 50.0, tau_adapt_ms=300.0, adapt_frac=0.8)
            for taste in LEG_TASTES:
                idx = _pop(sub, f"{part}_taste_bristle_gustatory_neuron", taste)
                ls.channels[taste] = Channel(sub.index.to_numpy()[idx], TASTE_RMAX[taste], tau_adapt_ms=1000.0, adapt_frac=0.5)
            self.legs.append(ls)

    def _build_body(self, n: pd.DataFrame) -> None:
        b = self.body

        def sel(**kw) -> np.ndarray:
            m = np.ones(len(n), dtype=bool)
            for col, val in kw.items():
                col_s = n[col]
                m &= col_s.isin(val).to_numpy() if isinstance(val, tuple) else col_s.str.contains(val, regex=False).to_numpy()
            return np.flatnonzero(m)

        # soies tactiles (phasico-toniques, adaptation rapide)
        b["bristle_head"] = Channel(sel(cls="bristle_neuron", body_part=HEAD_BRISTLES), 100.0, tau_adapt_ms=200.0, adapt_frac=0.8)
        b["bristle_thorax"] = Channel(sel(cls="bristle_neuron", body_part=("thorax",)), 100.0, tau_adapt_ms=200.0, adapt_frac=0.8)
        b["bristle_abdomen"] = Channel(sel(cls="bristle_neuron", body_part=("abdomen",)), 100.0, tau_adapt_ms=200.0, adapt_frac=0.8)
        b["bristle_proboscis"] = Channel(np.concatenate([
            sel(cls="bristle_neuron", body_part=("labellum", "haustellum", "maxillary_palp")),
            sel(cls="taste_peg_tactile_neuron", function="tactile")]), 100.0, tau_adapt_ms=200.0, adapt_frac=0.8)
        for s in ("l", "r"):
            side = SIDE[s]
            b[f"bristle_antenna_{s}"] = Channel(sel(cls="bristle_neuron", body_part=("antenna",), side=(side,)), 100.0,
                                                tau_adapt_ms=200.0, adapt_frac=0.8)
            # organe de Johnston : A/B vibration (son ; A basses fréquences, B hautes), C/D/E/F position
            b[f"jo_vib_{s}"] = Channel(sel(sub_class=("johnstons_organ_A_neuron", "johnstons_organ_B_neuron"), side=(side,)),
                                       120.0, tau_adapt_ms=500.0, adapt_frac=0.5)
            for grp, axis in (("E", "pitch"), ("F", "pitch"), ("C", "yaw"), ("D", "yaw")):
                halves = _split(sel(sub_class=(f"johnstons_organ_{grp}_neuron",), side=(side,)), 2)
                b[f"jo_{grp}_{axis}_neg_{s}"] = Channel(halves[0], 60.0, r_spont=2.0)
                b[f"jo_{grp}_{axis}_pos_{s}"] = Channel(halves[1], 60.0, r_spont=2.0)
            # olfaction : ORN par classe d'odeur (annotations `Function`), taux spontané ~8 Hz, adaptation
            for organ in ("antenna", "maxillary_palp"):
                for odor in ODOR_CLASSES:
                    idx = sel(sub_class=(f"{organ}_olfactory_receptor_neuron",), function=odor, side=(side,))
                    if idx.size:
                        b[f"orn_{organ}_{odor}_{s}"] = Channel(idx, 200.0, r_spont=ORN_SPONT_HZ, tau_adapt_ms=500.0, adapt_frac=0.5)
            # thermo / hygro (ariste et sacculus)
            b[f"cooling_{s}"] = Channel(sel(cls="thermosensory_receptor_neuron", function="cooling", side=(side,)), 60.0, r_spont=5.0)
            b[f"heating_{s}"] = Channel(sel(cls="thermosensory_receptor_neuron", function="heating", side=(side,)), 60.0, r_spont=5.0)
            b[f"evaporation_{s}"] = Channel(sel(cls="thermosensory_receptor_neuron", function="evaporation", side=(side,)), 40.0, r_spont=3.0)
            b[f"dry_{s}"] = Channel(sel(cls="hygrosensory_receptor_neuron", function="Ir40a", side=(side,)), 40.0, r_spont=3.0)
            b[f"humid_{s}"] = Channel(sel(cls="hygrosensory_receptor_neuron", function="humid", side=(side,)), 40.0, r_spont=3.0)
        # ORN sans classe d'odeur annotée : taux spontané seul
        known = np.concatenate([c.neurons for k, c in b.items() if k.startswith("orn_")])
        orn_all = sel(cls="olfactory_receptor_neuron")
        b["orn_unknown"] = Channel(np.setdiff1d(orn_all, known), 0.0, r_spont=ORN_SPONT_HZ)
        # goût du labelle (soies + taste pegs), actif quand le labelle touche le sol
        for taste in LABELLUM_TASTES:
            idx = np.concatenate([sel(sub_class=("labellum_taste_bristle_gustatory_neuron",), function=taste),
                                  sel(sub_class=("labellum_taste_peg_gustatory_neuron",), function=taste)])
            b[f"labellum_{taste}"] = Channel(idx, TASTE_RMAX[taste], tau_adapt_ms=1000.0, adapt_frac=0.5)
        # proprioception du cou et vibration du thorax
        neck = np.concatenate([sel(sub_class=("neck_chordotonal_organ_neuron",)), sel(sub_class=("prosternal_hair_plate_neuron",))])
        b["neck"] = Channel(neck, 80.0)
        b["wheelers_organ"] = Channel(sel(sub_class=("wheelers_chordotonal_organ_neuron",)), 100.0, tau_adapt_ms=300.0)
        # oxygénation (abdomen) : hypoxie ambiante
        b["oxygenation"] = Channel(sel(cls="oxygenation_neuron"), 30.0)

    # ---- lecture ---------------------------------------------------------------------------------
    def _s(self, data: mujoco.MjData, name: str) -> float:
        return float(data.sensordata[self.sensor[name]])

    def _hp(self, key: str, x: float, tau_ms: float = 20.0) -> float:
        """Composante rapide (passe-haut) de x : x - moyenne glissante."""
        self._vib[key] += (x - self._vib[key]) * self.dt_ms / tau_ms
        return x - self._vib[key]

    def read(self, data: mujoco.MjData) -> None:
        self._read_legs(data)
        self._read_body(data)

    def _read_legs(self, data: mujoco.MjData) -> None:
        m, env = self.model, self.env
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
            contact = touch > 0.01
            ls.channels["tactile"].value = 1.0 if contact else 0.0
            ls.channels["club"].value = np.clip(abs(w) * 0.2, 0, 1)
            tastes = env.tastes_at(data.site_xpos[self.site[f"{leg}_claw"]]) if contact and env.taste_patches else {}
            for taste in LEG_TASTES:
                v = tastes.get(taste, 0.0) + (self.sugar if taste == "sugar" else 0.0)
                ls.channels[taste].value = min(1.0, v) if contact else 0.0

    def _read_body(self, data: mujoco.MjData) -> None:
        b, env, dt = self.body, self.env, self.dt_ms
        touch = {k: self._s(data, f"{k}_contact") for k in ("head", "thorax", "labellum", "l_antenna", "r_antenna")}
        abdomen = sum(self._s(data, f"{seg}_contact") for seg in ("c_abdomen12", "c_abdomen3", "c_abdomen4", "c_abdomen5", "c_abdomen6"))
        b["bristle_head"].value = np.clip(touch["head"] / 2.0, 0, 1)             # µN : ~ 1/5 du poids
        b["bristle_thorax"].value = np.clip(touch["thorax"] / 2.0, 0, 1)
        b["bristle_abdomen"].value = np.clip(abdomen / 2.0, 0, 1)
        b["bristle_proboscis"].value = np.clip(touch["labellum"] / 2.0, 0, 1)
        labellum_pos = data.site_xpos[self.site["labellum"]]
        tastes = env.tastes_at(labellum_pos) if touch["labellum"] > 0.01 and env.taste_patches else {}
        for taste in LABELLUM_TASTES:
            b[f"labellum_{taste}"].value = tastes.get(taste, 0.0)
        for s in ("l", "r"):
            b[f"bristle_antenna_{s}"].value = np.clip(touch[f"{s}_antenna"] / 0.5, 0, 1)
            # organe de Johnston : vibration = vitesse angulaire passe-haut (son, marche) ; référence
            # 1° d'amplitude à 200 Hz ~ 22 rad/s ; position = déviation statique (gravité, vent), réf. 3°
            vel = np.hypot(self._s(data, f"{s}_antenna_pitch_vel"), self._s(data, f"{s}_antenna_yaw_vel"))
            b[f"jo_vib_{s}"].value = np.clip(abs(self._hp(s, vel)) / 22.0, 0, 1)
            for axis in ("pitch", "yaw"):
                q = np.rad2deg(self._s(data, f"{s}_antenna_{axis}_pos")) / 3.0
                for grp in (("E", "F") if axis == "pitch" else ("C", "D")):
                    b[f"jo_{grp}_{axis}_neg_{s}"].value = np.clip(-q, 0, 1)
                    b[f"jo_{grp}_{axis}_pos_{s}"].value = np.clip(q, 0, 1)
            arista = data.site_xpos[self.site[f"{s}_arista"]]
            odors = env.odors_at(arista) if env.odor_sources else {}
            for organ, pos in (("antenna", arista), ("maxillary_palp", labellum_pos)):
                od = odors if organ == "antenna" else (env.odors_at(pos) if env.odor_sources else {})
                for odor in ODOR_CLASSES:
                    ch = b.get(f"orn_{organ}_{odor}_{s}")
                    if ch is not None:
                        ch.value = np.clip(od.get(odor, 0.0), 0, 1)
            # thermo : cellules phasiques (dT/dt, réf. 1 °C/s -> r_max) ; hygro : tonique
            t = env.temperature_at(arista)
            prev = self._temp_prev[s]
            if prev is None:
                prev = t
            t_lp = prev + (t - prev) * dt / 50.0          # inertie thermique de l'ariste (~50 ms)
            dTdt = (t_lp - prev) / (dt / 1000.0) / 0.5    # réf. 0,5 °C/s -> r_max (seuil ~0,5 °C, Budelli 2019)
            self._temp_prev[s] = t_lp
            b[f"cooling_{s}"].value = np.clip(-dTdt, 0, 1)
            b[f"heating_{s}"].value = np.clip(dTdt, 0, 1)
            b[f"dry_{s}"].value = np.clip(1.0 - env.humidity, 0, 1)
            b[f"humid_{s}"].value = np.clip(env.humidity, 0, 1)
            b[f"evaporation_{s}"].value = np.clip((1.0 - env.humidity) * 0.5 - 0.5 * dTdt, 0, 1)
        neck = max(abs(self._s(data, f"{j}_pos")) for j in ("neck_yaw", "neck_pitch", "neck_roll"))
        b["neck"].value = np.clip(np.rad2deg(neck) / 40.0, 0, 1)
        adr = self.sensor["thorax_acc"]
        acc = float(np.linalg.norm(data.sensordata[adr:adr + 3]))
        b["wheelers_organ"].value = np.clip(abs(self._hp("thorax", acc, 5.0)) / 5000.0, 0, 1)   # mm/s², ~0.5 g
        b["oxygenation"].value = np.clip(env.hypoxia, 0, 1)

    # ---- sortie ----------------------------------------------------------------------------------
    def _channels(self):
        for ls in self.legs:
            yield from ls.channels.values()
        yield from self.body.values()

    def spikes(self) -> np.ndarray:
        """Neurones sensoriels forcés à tirer sur ce pas (Poisson à partir des grandeurs lues)."""
        out = []
        for ch in self._channels():
            if not ch.neurons.size:
                continue
            r = ch.rate(self.dt_ms)
            if r > 0:
                hit = ch.neurons[self.rng.random(ch.neurons.size) < r * self.dt_ms / 1000.0]
                if hit.size:
                    out.append(hit)
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def summary(self) -> pd.DataFrame:
        rows = [(ls.leg, k, ch.neurons.size, ch.r_max, ch.r_spont) for ls in self.legs for k, ch in ls.channels.items()]
        rows += [("body", k, ch.neurons.size, ch.r_max, ch.r_spont) for k, ch in self.body.items()]
        return pd.DataFrame(rows, columns=["organ", "channel", "n_neurons", "r_max_hz", "r_spont_hz"])
