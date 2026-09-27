"""Charge le corps, le laisse tomber sur le sol, vérifie qu'il tient debout (ressorts articulaires passifs,
sans muscle), et rend une image."""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image

from flywire_sim import data
from flywire_sim.body.model import build_mjcf

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=1.0)
ap.add_argument("--adhesion", type=float, default=0.0)
args = ap.parse_args()

m = mujoco.MjModel.from_xml_string(build_mjcf())
d = mujoco.MjData(m)
print(f"corps : {m.nbody} segments, {m.njnt} articulations ({m.nv} DoF), {m.nu} actionneurs, {m.nsensor} capteurs, "
      f"masse totale {mujoco.mj_getTotalmass(m)*1000:.3f} mg")
for i in range(m.nu):
    if "adhesion" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
        d.ctrl[i] = args.adhesion
# pose neutre debout : chaque articulation à son angle de repos (springref), puis on pose les pattes au sol
for j in range(m.njnt):
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
        d.qpos[m.jnt_qposadr[j]] = m.qpos_spring[m.jnt_qposadr[j]]
mujoco.mj_forward(m, d)
claws = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_claw") for leg in ("lf", "lm", "lh", "rf", "rm", "rh")]
lowest = min(d.site_xpos[s][2] for s in claws)
d.qpos[2] += 0.03 - lowest
mujoco.mj_forward(m, d)
z0 = d.qpos[2]
print("hauteur des griffes au départ (mm) :", np.round([d.site_xpos[s][2] for s in claws], 3).tolist())
r = mujoco.Renderer(m, 720, 1280)
data.RESULTS.mkdir(exist_ok=True)
r.update_scene(d, camera="side")
Image.fromarray(r.render()).save(data.RESULTS / "body_init_side.png")
n = int(args.duration / m.opt.timestep)
zs = []
for k in range(n):
    mujoco.mj_step(m, d)
    if k % 1000 == 0:
        zs.append(d.qpos[2])
quat = d.qpos[3:7]
up = np.array([2 * (quat[1] * quat[3] - quat[0] * quat[2]), 2 * (quat[2] * quat[3] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)])
print(f"hauteur thorax : départ {z0:.3f} mm -> {d.qpos[2]:.3f} mm ; trajectoire (tous les 100 ms) : {np.round(zs, 3).tolist()}")
print(f"axe vertical du thorax (0,0,1 = droit) : {np.round(up, 3)} ; vitesse finale {np.linalg.norm(d.qvel[:3]):.3f} mm/s")
contacts = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) for c in d.contact}
print(f"{d.ncon} contacts, segments en contact : {sorted(contacts)}")
for cam in ("side", "front", "top"):
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(data.RESULTS / f"body_stand_{cam}.png")
print("images ->", data.RESULTS / "body_stand_side.png")
