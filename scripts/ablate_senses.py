"""Diagnostic § 9.4 : quel groupe de canaux sensoriels allume l'attracteur de la moelle au repos (corps posé).
Usage : ablate_senses.py W GAMMA MS."""
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import replace

import numpy as np

sys.path.insert(0, ".")
from flywire_sim import banc
from flywire_sim.body.sim import BodyBrainSim

w, g, ms = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
sim = BodyBrainSim(replace(banc.CALIBRATED_V2, w_syn=w, size_norm=g), brain="fafb")
keys = [k for k, _ in sim.senses._channels()]
leg = [k for k in keys if ":" in k]; body = [k for k in keys if ":" not in k]
grp = {"proprio patte (claw/hook/club/campaniform/hairplate)": [k for k in leg if re.search(r":(claw|hook|club|campaniform|hairplate)", k)],
       "tactile patte": [k for k in leg if ":tactile" in k],
       "goût patte": [k for k in leg if not re.search(r":(claw|hook|club|campaniform|hairplate|tactile)", k)],
       "corps/tête (soies, JO, ORN, thermo/hygro, labellum, cou...)": body}
grp["aucun"] = []; grp["tous"] = keys
n = sim.neurons; sc = n.super_class.to_numpy(dtype=str); bp = n.body_part.to_numpy(dtype=str)
for name, ks in grp.items():
    s2 = BodyBrainSim(replace(banc.CALIBRATED_V2, w_syn=w, size_norm=g), brain="fafb") if name != next(iter(grp)) else sim
    s2.senses.enabled = set(ks)
    s2.run(ms, None, 0.0, spike_log=True)
    spk = np.concatenate(s2.spike_neurons) if s2.spike_neurons else np.empty(0, int)
    r = np.bincount(spk, minlength=len(n)) / (ms / 1000)
    inn = r[sc == "ventral_nerve_cord_intrinsic"]; leg = np.isin(bp, ["front_leg", "middle_leg", "hind_leg"])
    print(f"[{name}: {len(ks)} canaux] total {r.sum():.0f} spk/s | IN VNC {inn.mean():.1f} Hz ({(inn>0).mean():.0%}, >100 Hz {(inn>100).sum()}) | "
          f"MN patte {r[(sc=='motor') & leg].mean():.1f} Hz | MN aile {r[(sc=='motor') & (bp=='wing')].mean():.1f} Hz | "
          f"AN {r[sc=='ascending'].mean():.1f} | DN {r[sc=='descending'].mean():.1f} | cerveau {r[np.isin(sc, ['central_brain_intrinsic','optic_lobe_intrinsic'])].mean():.2f}", flush=True)
