"""Diagnostic § 9.5 : mécanismes limitant la fréquence (adaptation, dépression) face à l'attracteur de la moelle,
corps + capteurs, γ = 1, w 0,9 ; repos puis DNg100. Rien n'est adopté par défaut."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import replace

import numpy as np

sys.path.insert(0, ".")
from flywire_sim import banc
from flywire_sim.body.sim import BodyBrainSim

ms = 600.0
configs = {"base w0.9 g1": {}, "adapt 1mV/100ms": {"adapt_b": 1.0, "tau_adapt": 100.0}, "std U0.2/500ms": {"std_U": 0.2},
           "adapt+std": {"adapt_b": 1.0, "tau_adapt": 100.0, "std_U": 0.2}}
for name, kw in configs.items():
    for stim in (False, True):
        p = replace(banc.CALIBRATED_V2, w_syn=0.9, size_norm=1.0, **kw)
        sim = BodyBrainSim(p, brain="fafb")
        if p.std_U:
            sim.lif.std_U = np.maximum(sim.lif.std_U, np.float32(p.std_U)); sim.stepper = sim.lif.stepper()
        n = sim.neurons; sc = n.super_class.to_numpy(dtype=str); bp = n.body_part.to_numpy(dtype=str)
        dn = sim.find(super_class="descending", cell_type=["DNg100"]) if stim else None
        sim.run(ms, dn, 50.0, spike_log=True)
        r = np.bincount(np.concatenate(sim.spike_neurons), minlength=len(n)) / (ms / 1000)
        inn = r[sc == "ventral_nerve_cord_intrinsic"]; leg = np.isin(bp, ["front_leg", "middle_leg", "hind_leg"]); mnl = r[(sc == "motor") & leg]
        print(f"[{name} | {'DNg100' if stim else 'repos'}] total {r.sum():.0f} | IN VNC {inn.mean():.1f} Hz ({(inn>0).mean():.0%}, >100 Hz {(inn>100).sum()}) | "
              f"MN patte {mnl.mean():.1f} Hz ({(mnl>0).mean():.0%}) | MN aile {r[(sc=='motor') & (bp=='wing')].mean():.1f} | AN {r[sc=='ascending'].mean():.1f} | "
              f"DN {r[sc=='descending'].mean():.1f} | cerveau {r[np.isin(sc, ['central_brain_intrinsic','optic_lobe_intrinsic'])].mean():.2f}", flush=True)
