"""Expérience en boucle fermée : cerveau BANC complet + corps MuJoCo. Stimule une population de neurones
descendants « marche » et observe les motoneurones, les muscles et le déplacement du corps."""
import argparse
import os
import subprocess
from dataclasses import replace

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image

from flywire_sim import banc, data
from flywire_sim.body.model import LEGS
from flywire_sim.body.sim import BodyBrainSim

# population « marche » large (Pugliese et al. 2025 ; Cheong et al. 2024) ; par défaut on stimule DNg100 seul
WALK_DN = ["DNp09", "DNa01", "DNa02", "DNa03", "DNa04", "DNa05", "DNa06", "DNa07", "DNb01", "DNb02", "DNb05",
           "DNg13", "DNg100", "DNp10", "DNp42", "DNa10", "DNa11", "DNa14", "DNa15", "DNp25"]

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=300.0, help="ms")
ap.add_argument("--rate", type=float, default=50.0, help="Hz de stimulation des DN")
ap.add_argument("--dn", nargs="+", default=["DNg100"], help="types de DN stimulés ('walk' = population large)")
ap.add_argument("--w", type=float, default=banc.CALIBRATED_V2.w_syn, help="mV par synapse (défaut : CALIBRATED_V2)")
ap.add_argument("--gamma", type=float, default=banc.CALIBRATED_V2.size_norm)
ap.add_argument("--adapt", type=float, default=banc.CALIBRATED_V2.adapt_b)
ap.add_argument("--no-stim", action="store_true")
ap.add_argument("--brain", default="banc", choices=["banc", "fafb"])
ap.add_argument("--video", action="store_true")
ap.add_argument("--out", default="body_brain_loop")
args = ap.parse_args()

sim = BodyBrainSim(replace(banc.CALIBRATED_V2, w_syn=args.w, size_norm=args.gamma, adapt_b=args.adapt), brain=args.brain)
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
groups = {"DN": n.super_class.eq("descending"), "AN": n.super_class.eq("ascending"),
          "IN VNC": n.super_class.eq("ventral_nerve_cord_intrinsic"),
          "MN patte": n.super_class.eq("motor") & n.body_part.isin(["front_leg", "middle_leg", "hind_leg"]),
          "MN aile": n.super_class.eq("motor") & n.body_part.eq("wing"),
          "cerveau": n.super_class.isin(["central_brain_intrinsic", "optic_lobe_intrinsic", "visual_projection"])}
print("taux par groupe :", " | ".join(f"{k} {rates[g.to_numpy()].mean():.1f} Hz ({(counts[g.to_numpy()] > 0).mean():.0%} actifs)"
                                     for k, g in groups.items()))
# marche : appui de chaque patte (capteurs tactiles du tarse), nombre de levers et alternance gauche/droite
con = np.array(tr.contact) > 0.01
duty = con.mean(axis=0)
lifts = np.diff(con.astype(int), axis=0).clip(min=0).sum(axis=0)
print("appui par patte (fraction du temps) :", {leg: round(float(v), 2) for leg, v in zip(LEGS, duty)},
      "; levers :", {leg: int(v) for leg, v in zip(LEGS, lifts)})
same = [np.corrcoef(con[:, i], con[:, j])[0, 1] if con[:, i].std() * con[:, j].std() > 0 else np.nan
        for i, j in ((0, 3), (1, 4), (2, 5))]
print("corrélation d'appui gauche/droite par paire (tripode attendu : < 0) :", np.round(same, 2).tolist())
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
