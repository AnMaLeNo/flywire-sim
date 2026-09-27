"""Test en boucle ouverte des muscles (sans cerveau) : rythme tripode scripté sur les actionneurs
nommés d'après les muscles BANC. Vérifie que le corps peut avancer avec ces muscles et ces couples."""
import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image

from flywire_sim import data
from flywire_sim.body.model import LEGS, build_mjcf

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=2.0)
ap.add_argument("--period", type=float, default=0.1, help="période du pas (s) ; ~10 Hz chez la drosophile")
ap.add_argument("--video", action="store_true")
args = ap.parse_args()

m = mujoco.MjModel.from_xml_string(build_mjcf())
d = mujoco.MjData(m)
aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
for j in range(m.njnt):
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
        d.qpos[m.jnt_qposadr[j]] = m.qpos_spring[m.jnt_qposadr[j]]
mujoco.mj_forward(m, d)
claws = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_claw") for leg in LEGS]
d.qpos[2] += 0.03 - min(d.site_xpos[s][2] for s in claws)

# tripode : {lf, rm, lh} en phase, {rf, lm, rh} en opposition
TRIPOD = {"lf": 0.0, "rm": 0.0, "lh": 0.0, "rf": 0.5, "lm": 0.5, "rh": 0.5}
SWING = ["trochanter_flexor", "tibia_flexor", "sternal_anterior_rotator"]          # lever + avancer
STANCE = ["trochanter_extensor", "tergotrochanter", "tibia_extensor", "sternal_posterior_rotator", "long_tendon_muscle"]

frames = []
r = mujoco.Renderer(m, 480, 854) if args.video else None
x0 = d.qpos[0].copy()
n = int(args.duration / m.opt.timestep)
for k in range(n):
    t = k * m.opt.timestep
    for leg in LEGS:
        ph = (t / args.period + TRIPOD[leg]) % 1.0
        swing = ph < 0.35
        for mu in SWING:
            a = aid(f"{leg}_{mu}")
            if a >= 0:
                d.ctrl[a] = 0.8 if swing else 0.0
        for mu in STANCE:
            a = aid(f"{leg}_{mu}")
            if a >= 0:
                d.ctrl[a] = 0.0 if swing else 0.6
        d.ctrl[aid(f"{leg}_adhesion")] = 0.0 if swing else 1.0
    mujoco.mj_step(m, d)
    if r is not None and k % 200 == 0:   # 50 images/s
        r.update_scene(d, camera="side")
        frames.append(r.render())
print(f"déplacement du thorax : dx={d.qpos[0]-x0:.3f} mm, dy={d.qpos[1]:.3f} mm en {args.duration} s ; "
      f"hauteur {d.qpos[2]:.3f} mm ; vitesse moyenne {(d.qpos[0]-x0)/args.duration:.2f} mm/s (marche réelle : 10-30 mm/s)")
quat = d.qpos[3:7]
print("axe vertical du thorax :", np.round([2*(quat[1]*quat[3]-quat[0]*quat[2]), 2*(quat[2]*quat[3]+quat[0]*quat[1]), 1-2*(quat[1]**2+quat[2]**2)], 3))
if frames:
    out = data.RESULTS / "body_muscle_test"
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(out / f"{i:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "50", "-i", str(out / "%04d.png"),
                    "-pix_fmt", "yuv420p", str(data.RESULTS / "body_muscle_test.mp4")], check=True)
    print("vidéo ->", data.RESULTS / "body_muscle_test.mp4")
