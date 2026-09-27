"""Diagnostic § 9.1 : taux des afférents forcés par canal, mouche posée au repos. Usage : senses_rest_rates.py W GAMMA MS."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from flywire_sim import banc
from flywire_sim.body.sim import BodyBrainSim

w, g, ms = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
sim = BodyBrainSim(replace(banc.CALIBRATED_V2, w_syn=w, size_norm=g), brain="fafb")
sim.run(ms, None, 0.0, spike_log=True)
spk = np.concatenate(sim.spike_neurons); r = np.bincount(spk, minlength=len(sim.neurons)) / (ms / 1000)
rows = []
for key, ch in sim.senses._channels():
    if ch.neurons.size:
        rows.append((key, ch.neurons.size, r[ch.neurons].mean(), (r[ch.neurons] > 0).mean(), r[ch.neurons].sum()))
df = pd.DataFrame(rows, columns=["canal", "n", "Hz moyen", "frac actifs", "spk/s total"])
df["groupe"] = df.canal.str.replace(r"^(lf|lm|lh|rf|rm|rh):", "", regex=True).str.replace(r"\d+$", "", regex=True)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(df.groupby("groupe").agg(n=("n", "sum"), hz_moyen=("Hz moyen", "mean"), frac=("frac actifs", "mean"), total=("spk/s total", "sum")).sort_values("total", ascending=False).round(2).to_string())
print("total forcé spk/s :", round(df["spk/s total"].sum()))
