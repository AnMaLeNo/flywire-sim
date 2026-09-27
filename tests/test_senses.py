"""Tests des capteurs v1 : environnement, modèle (caméras oculaires, soies, antennes), géométrie de l'œil."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import pandas as pd
import pytest

from flywire_sim.body.environment import Environment, OdorSource, TastePatch
from flywire_sim.body.model import build_mjcf
from flywire_sim.body.vision import FACES, Retina, build_retinas, robust_sphere_center


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(build_mjcf())


def test_environment_fields():
    env = Environment(odor_sources=[OdorSource(np.zeros(3), {"yeasty": 1.0}, radius=2.0)],
                      taste_patches=[TastePatch(np.zeros(3), {"sugar": 0.8}, radius=1.0)],
                      hot_spot=(np.zeros(3), 10.0, 1.0))
    assert env.odors_at(np.zeros(3))["yeasty"] == pytest.approx(1.0)
    assert env.odors_at(np.array([2.0, 0, 0]))["yeasty"] == pytest.approx(np.exp(-0.5))
    assert env.odors_at(np.zeros(3))["aversive"] == 0.0
    assert env.tastes_at(np.array([0.5, 0, 0]))["sugar"] == 0.8
    assert env.tastes_at(np.array([1.5, 0, 0]))["sugar"] == 0.0
    assert env.temperature_at(np.zeros(3)) == pytest.approx(35.0)
    assert env.temperature_at(np.array([50.0, 0, 0])) == pytest.approx(25.0)


def test_model_has_eye_cameras_body_bristles_and_antenna_joints(model):
    for side in ("l", "r"):
        for f in FACES:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{side}_eye_{f}")
            assert cid >= 0 and model.cam_fovy[cid] == 90.0
        for j in ("antenna_pitch", "antenna_yaw"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_arista") >= 0
    for s in ("head", "thorax", "labellum", "l_antenna", "c_abdomen6"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{s}_contact") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "neck_yaw_pos") >= 0


def test_eye_cameras_render(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    r = mujoco.Renderer(model, height=16, width=16)
    imgs = []
    for side in ("l", "r"):
        for f in FACES:
            r.update_scene(d, camera=f"{side}_eye_{f}")
            imgs.append(r.render().astype(float).mean())
    r.close()
    assert all(0 < v < 255 for v in imgs)
    assert imgs[FACES.index("up")] > imgs[FACES.index("down")]   # ciel plus clair que le sol


def test_robust_sphere_center_ignores_outliers():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    P = np.array([10.0, 20.0, 30.0]) + 100.0 * v
    P = np.r_[P, rng.uniform(-50, 50, size=(60, 3))]
    c = robust_sphere_center(P, radius=100.0)
    assert np.allclose(c, [10.0, 20.0, 30.0], atol=3.0)


def test_build_retinas_maps_shell_to_hemifield():
    rng = np.random.default_rng(1)
    rows, pos = [], []
    rid = 1
    for side, cx in (("left", 800.0), ("right", 200.0)):
        # coquille hémisphérique de rayon 100 µm tournée vers l'extérieur (±x du volume)
        v = rng.normal(size=(600, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v[:, 0] = np.abs(v[:, 0]) * (1 if side == "left" else -1)
        for t, p in zip(np.tile(["R7", "R8", "L1", "L2", "L5", "Mi1"], 100), np.array([cx, 150.0, 100.0]) + 100 * v):
            rows.append((rid, t, side))
            pos.append((rid, *p))
            rid += 1
    neurons = pd.DataFrame(rows, columns=["root_id", "cell_type", "side"])
    positions = pd.DataFrame(pos, columns=["root_id", "x", "y", "z"])
    rets = build_retinas(neurons, positions)
    assert [r.side for r in rets] == ["l", "r"]
    for r in rets:
        assert isinstance(r, Retina) and r.idx.size == 600
        assert np.allclose(np.linalg.norm(r.dirs, axis=1), 1.0)
        az = np.degrees(np.arctan2(r.dirs[:, 1], r.dirs[:, 0]))
        lat = 1 if r.side == "l" else -1
        assert abs(np.median(lat * az) - 60.0) < 15.0       # axe optique à ~60° du côté de l'œil
        assert (lat * az > -45).mean() > 0.9               # champ presque entièrement ipsilatéral


def test_tarsal_touch_and_taste_channels():
    import numpy as np

    from flywire_sim.body.environment import Environment, TastePatch
    from flywire_sim.body.senses import Channel
    from flywire_sim.body.sim import BodyBrainSim

    ch = Channel(neurons=np.arange(3), r_max=100.0, r_spont=5.0, tau_adapt_ms=100.0, adapt_frac=0.5)
    ch.value = 1.0
    r0 = ch.rate(0.1)
    for _ in range(5000):
        r = ch.rate(0.1)
    assert r0 > 100.0 and 50.0 < r < r0          # adaptation soutenue vers r_spont + (1 - frac) r_max

    env = Environment(taste_patches=[TastePatch(np.zeros(3), {"sugar": 1.0}, radius=5.0)])
    sim = BodyBrainSim(env=env)
    sim.run(30)
    d = sim.data
    touch = [sum(sim.senses._s(d, f"{ls.leg}_tarsus{k}_contact") for k in range(1, 6)) for ls in sim.senses.legs]
    assert all(t > 0 for t in touch), touch          # les 6 tarses posés sont détectés par les soies tactiles
    assert all(ls.channels["sugar"].value == 1.0 for ls in sim.senses.legs)
    assert all(ls.channels["bitter"].value == 0.0 for ls in sim.senses.legs)
    # soies tactiles par tarsomère : seuls les segments en contact répondent, chaque canal a ses neurones
    for ls in sim.senses.legs:
        vals = [ls.channels[f"tactile{k}"].value for k in range(1, 6)]
        assert 1.0 in vals
        assert all(v == (1.0 if sim.senses._s(d, f"{ls.leg}_tarsus{k}_contact") > 0.01 else 0.0)
                   for k, v in enumerate(vals, start=1))
        assert all(ls.channels[f"tactile{k}"].neurons.size > 0 for k in range(1, 6))
        assert ls.channels["contact_pheromone"].neurons.size > 0 and ls.channels["club"].neurons.size > 0
