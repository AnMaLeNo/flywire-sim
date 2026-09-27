"""Expériences sensorielles en scène 3D (capteurs v1) : la mouche est posée sur le sol, cerveau + moelle
BANC en boucle fermée, et l'on mesure la réponse des populations sensorielles et centrales à un stimulus
par rapport à une condition témoin. Résultats -> results/senses_*.txt.

  python scripts/experiments_senses.py [posture|vision|odor|taste|wind|all] [--ms 300]
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flywire_sim import banc, data
from flywire_sim.body.environment import (
    Environment,
    OdorSource,
    TastePatch,
)
from flywire_sim.body.sim import BodyBrainSim

OBJECT_XML = '<geom name="target" type="box" size="1.5 0.3 1.5" pos="{x} {y} 1.5" rgba="0 0 0 1"/>'


def rates(sim: BodyBrainSim, ms: float, by: str = "cls") -> pd.Series:
    sp = np.concatenate(sim.spike_neurons) if sim.spike_neurons else np.empty(0, dtype=int)
    key = sim.neurons[by].fillna("?").to_numpy()
    n = pd.Series(key).value_counts()
    c = pd.Series(key[sp]).value_counts()
    return (c / n.reindex(c.index) / (ms / 1000.0)).sort_values(ascending=False)


def side_rates(sim: BodyBrainSim, ms: float, mask: np.ndarray) -> dict:
    sp = np.concatenate(sim.spike_neurons) if sim.spike_neurons else np.empty(0, dtype=int)
    side = sim.neurons.side.to_numpy()
    out = {}
    for s in ("left", "right"):
        idx = np.flatnonzero(mask & (side == s))
        out[s] = float(np.isin(sp, idx).sum() / max(1, idx.size) / (ms / 1000.0))
    return out


PARAMS = {"v2": banc.CALIBRATED_V2, "calibrated": banc.CALIBRATED}
CHOSEN = ["v2"]


def run(env: Environment | None, ms: float, vision: bool = False, extra_xml: str = "", **kw) -> BodyBrainSim:
    sim = BodyBrainSim(params=PARAMS[CHOSEN[0]], env=env, vision=vision, extra_xml=extra_xml, **kw)
    sim.run(ms, spike_log=True)
    return sim


def exp_posture(ms: float, out: list) -> None:
    sim = run(None, ms)
    d = sim.data
    z = d.qpos[2]
    ant = {s: np.rad2deg([d.sensor(f"{s}_antenna_pitch_pos").data[0], d.sensor(f"{s}_antenna_yaw_pos").data[0]]) for s in ("l", "r")}
    out.append(f"[posture] thorax z={z:.3f} mm ; déviation gravitaire des antennes (pitch, yaw, °) : {ant}")
    r = rates(sim, ms)
    out.append("[posture] taux spontanés (Hz/neurone) : " + ", ".join(f"{k}={v:.1f}" for k, v in r.head(12).items()))
    out.append(f"[posture] MN de patte : {sim.is_leg_mn[np.concatenate(sim.spike_neurons)].sum() / 391 / (ms / 1000):.1f} Hz/MN")


def exp_vision(ms: float, out: list) -> None:
    ctrl = run(None, ms, vision=True)
    obj = run(None, ms, vision=True, extra_xml=OBJECT_XML.format(x=1.5, y=3.0))   # objet noir à gauche
    n = ctrl.neurons
    for label, cls in (("photorécepteurs R7/R8", ["photoreceptor_neuron"]), ("lamina L1-L5", ["lamina_monopolar"]),
                       ("médulla intrinsèque", ["medulla_intrinsic"]), ("transmédullaires", ["transmedullary"]),
                       ("lobula columnar (LC)", ["lobula_columnar"]), ("T4/T5", ["T4", "T5"])):
        m = n.cls.isin(cls).to_numpy() | n.cell_type.isin(cls).to_numpy()
        a, b = side_rates(ctrl, ms, m), side_rates(obj, ms, m)
        out.append(f"[vision] {label:22s} témoin L/R = {a['left']:.1f}/{a['right']:.1f} Hz ; objet noir à gauche L/R = {b['left']:.1f}/{b['right']:.1f} Hz")
    lum = {s: float(obj.eyes.lum[obj.eyes.side == s].mean()) for s in ("l", "r")}
    lum0 = {s: float(ctrl.eyes.lum[ctrl.eyes.side == s].mean()) for s in ("l", "r")}
    out.append(f"[vision] log-luminance Rh1 moyenne par œil : témoin {lum0}, objet {lum}")


def exp_odor(ms: float, out: list) -> None:
    ctrl = run(None, ms)
    src = Environment(odor_sources=[OdorSource(np.array([1.0, 2.0, 0.5]), {"yeasty": 1.0}, radius=2.0)])
    odor = run(src, ms)
    n = ctrl.neurons
    for label, m in (("ORN levure", (n.sub_class == "antenna_olfactory_receptor_neuron").to_numpy() & n.function.str.contains("yeasty").to_numpy()),
                     ("ORN autres", (n.cls == "olfactory_receptor_neuron").to_numpy() & ~n.function.str.contains("yeasty").to_numpy()),
                     ("PN du lobe antennaire", (n.cls == "antennal_lobe_projection_neuron").to_numpy()),
                     ("LN du lobe antennaire", (n.cls == "antennal_lobe_local_neuron").to_numpy()),
                     ("cellules de Kenyon", (n.cls == "kenyon_cell").to_numpy()),
                     ("neurones descendants", (n.super_class == "descending").to_numpy())):
        a, b = side_rates(ctrl, ms, m), side_rates(odor, ms, m)
        out.append(f"[odeur] {label:22s} témoin L/R = {a['left']:.1f}/{a['right']:.1f} Hz ; source levure à gauche L/R = {b['left']:.1f}/{b['right']:.1f} Hz")
    out.append("[odeur] concentration lue aux aristes : " + str({k: round(v.value, 2) for k, v in odor.senses.body.items() if k.startswith("orn_antenna_yeasty")}))


def exp_taste(ms: float, out: list) -> None:
    ctrl = run(None, ms)
    sugar = run(Environment(taste_patches=[TastePatch(np.zeros(3), {"sugar": 1.0}, radius=5.0)]), ms)
    n = ctrl.neurons
    for label, m in (("GRN sucre (tarses)", n.sub_class.str.contains("leg_taste_bristle_gustatory").to_numpy() & n.function.str.contains("sugar").to_numpy()),
                     ("GRN amer (tarses)", n.sub_class.str.contains("leg_taste_bristle_gustatory").to_numpy() & n.function.str.contains("bitter").to_numpy()),
                     ("MN de la trompe", (n.cls == "proboscis_motor_neuron").to_numpy() if "proboscis_motor_neuron" in set(n.cls) else n.cell_type.str.startswith("MN9").to_numpy()),
                     ("neurones descendants", (n.super_class == "descending").to_numpy())):
        a, b = side_rates(ctrl, ms, m), side_rates(sugar, ms, m)
        out.append(f"[goût] {label:22s} témoin L/R = {a['left']:.1f}/{a['right']:.1f} Hz ; sol sucré L/R = {b['left']:.1f}/{b['right']:.1f} Hz")


def exp_wind(ms: float, out: list) -> None:
    ctrl = run(None, ms)
    wind = run(Environment(wind=np.array([0.0, -500.0, 0.0])), ms)       # vent venant de la gauche
    sound = run(Environment(sound=1.0, sound_hz=200.0), ms)
    n = ctrl.neurons
    jo_pos = n.sub_class.isin([f"johnstons_organ_{g}_neuron" for g in "CDEF"]).to_numpy()
    jo_vib = n.sub_class.isin(["johnstons_organ_A_neuron", "johnstons_organ_B_neuron"]).to_numpy()
    for label, m in (("JO C-F (position)", jo_pos), ("JO A/B (vibration)", jo_vib),
                     ("neurones descendants", (n.super_class == "descending").to_numpy())):
        a, b, c = side_rates(ctrl, ms, m), side_rates(wind, ms, m), side_rates(sound, ms, m)
        out.append(f"[vent/son] {label:20s} témoin L/R = {a['left']:.1f}/{a['right']:.1f} ; vent 0,5 m/s de gauche L/R = {b['left']:.1f}/{b['right']:.1f} ; chant 200 Hz L/R = {c['left']:.1f}/{c['right']:.1f} Hz")
    for sim, lab in ((ctrl, "témoin"), (wind, "vent"), (sound, "son")):
        d = sim.data
        ant = {s: np.rad2deg([d.sensor(f"{s}_antenna_pitch_pos").data[0], d.sensor(f"{s}_antenna_yaw_pos").data[0]]).round(2).tolist() for s in ("l", "r")}
        out.append(f"[vent/son] {lab}: angles antennaires (pitch, yaw, °) {ant} ; vibration JO lue {sim.senses.body['jo_vib_l'].value:.2f}/{sim.senses.body['jo_vib_r'].value:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all")
    ap.add_argument("--ms", type=float, default=300.0)
    ap.add_argument("--params", choices=sorted(PARAMS), default="v2")
    a = ap.parse_args()
    CHOSEN[0] = a.params
    exps = {"posture": exp_posture, "vision": exp_vision, "odor": exp_odor, "taste": exp_taste, "wind": exp_wind}
    todo = exps if a.which == "all" else {a.which: exps[a.which]}
    data.RESULTS.mkdir(parents=True, exist_ok=True)
    for name, fn in todo.items():
        out: list[str] = []
        fn(a.ms, out)
        txt = "\n".join(out)
        print(txt, flush=True)
        (data.RESULTS / f"senses_{name}_{a.params}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
