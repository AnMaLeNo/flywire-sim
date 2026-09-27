"""Expérience en boucle fermée : cerveau BANC complet + corps MuJoCo. Stimule une population de neurones
descendants « marche » et observe les motoneurones, les muscles et le déplacement du corps."""
import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image

from flywire_sim import banc, data
from flywire_sim.body.sim import BodyBrainSim
from flywire_sim.lif import LIFParams

# population « marche » large (Pugliese et al. 2025 ; Cheong et al. 2024) ; par défaut on stimule DNg100 seul
WALK_DN = ["DNp09", "DNa01", "DNa02", "DNa03", "DNa04", "DNa05", "DNa06", "DNa07", "DNb01", "DNb02", "DNb05",
           "DNg13", "DNg100", "DNp10", "DNp42", "DNa10", "DNa11", "DNa14", "DNa15", "DNp25"]

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=300.0, help="ms")
ap.add_argument("--rate", type=float, default=50.0, help="Hz de stimulation des DN")
ap.add_argument("--dn", nargs="+", default=["DNg100"], help="types de DN stimulés ('walk' = population large)")
ap.add_argument("--w", type=float, default=banc.CALIBRATED.w_syn)
ap.add_argument("--gamma", type=float, default=banc.CALIBRATED.size_norm)
ap.add_argument("--adapt", type=float, default=banc.CALIBRATED.adapt_b)
ap.add_argument("--no-stim", action="store_true")
ap.add_argument("--video", action="store_true")
ap.add_argument("--out", default="body_brain_loop")
args = ap.parse_args()

sim = BodyBrainSim(LIFParams(w_syn=args.w, size_norm=args.gamma, adapt_b=args.adapt))
print(f"réseau : {sim.net.W.shape[0]} neurones ; MN de patte reliés aux muscles : {sim.mmap.mn_idx.size} "
      f"({sim.mmap.unmapped.shape[0]} non reliés : {sim.mmap.unmapped.cell_type.value_counts().to_dict()})")
s = sim.senses.summary()
print("neurones sensoriels de patte branchés :", int(s.n_neurons.sum()), "->", s.groupby("channel").n_neurons.sum().to_dict())
dn = sim.find(super_class="descending", cell_type=WALK_DN if args.dn == ["walk"] else args.dn)
print(f"DN stimulés ({args.dn}) : {dn.size} à {args.rate} Hz")

frames = []
renderer = mujoco.Renderer(sim.model, 480, 854) if args.video else None


def render(m, d, t):
    renderer.update_scene(d, camera="side")
    frames.append(renderer.render())


x0 = sim.data.qpos[:3].copy()
tr = sim.run(args.duration, None if args.no_stim else dn, args.rate, render=render if args.video else None,
             spike_log=True, progress=True)
pos = np.array(tr.thorax_pos)
print(f"thorax : déplacement {np.round(pos[-1] - x0, 3)} mm ; hauteur finale {pos[-1, 2]:.3f} mm")
spk = np.concatenate(sim.spike_neurons) if sim.spike_neurons else np.empty(0, int)
counts = np.bincount(spk, minlength=sim.net.W.shape[0])
rates = counts / (args.duration / 1000)
n = sim.neurons
print(f"neurones actifs : {(counts > 0).sum()} ; MN pattes actifs : {(counts[sim.mmap.mn_idx] > 0).sum()}/{sim.mmap.mn_idx.size} "
      f"(taux moyen des actifs {rates[sim.mmap.mn_idx][counts[sim.mmap.mn_idx] > 0].mean() if (counts[sim.mmap.mn_idx] > 0).any() else 0:.0f} Hz)")
mn = n.iloc[sim.mmap.mn_idx].assign(rate=rates[sim.mmap.mn_idx])
print("taux moyen (Hz) par muscle :", mn.groupby("cell_type").rate.mean().round(0).sort_values(ascending=False).to_dict())
print("taux moyen (Hz) par patte :", mn.groupby(["body_part", "side"]).rate.mean().round(0).to_dict())
ctrl = np.array(tr.ctrl)
print(f"activation musculaire moyenne : {ctrl.mean():.3f}, max {ctrl.max():.2f}")
print(f"spikes sensoriels forcés / ms : {np.mean(tr.n_sens_spikes):.1f} ; spikes totaux / ms : {np.mean(tr.n_spikes):.0f}")
if frames:
    out = data.RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(out / f"{i:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "50", "-i", str(out / "%04d.png"),
                    "-pix_fmt", "yuv420p", str(data.RESULTS / f"{args.out}.mp4")], check=True)
    print("vidéo ->", data.RESULTS / f"{args.out}.mp4")
