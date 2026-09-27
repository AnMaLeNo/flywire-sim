"""Tests du corps v0 (MuJoCo) : chargement, six pattes, actionneurs/capteurs, stabilité sous gravité."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import pytest

from flywire_sim.body.model import LEG_MUSCLES, LEGS, build_mjcf


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(build_mjcf())


def standing(model):
    d = mujoco.MjData(model)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            d.qpos[model.jnt_qposadr[j]] = model.qpos_spring[model.jnt_qposadr[j]]
    mujoco.mj_forward(model, d)
    claws = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_claw") for leg in LEGS]
    d.qpos[2] += 0.03 - min(d.site_xpos[s][2] for s in claws)
    mujoco.mj_forward(model, d)
    return d


def test_loads_without_wings(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    assert not any("wing" in n for n in names)
    assert {"c_thorax", "c_head", "c_abdomen6", "c_haustellum", "l_haltere"} <= set(names)


def test_mass_is_that_of_a_female_fly(model):
    mg = mujoco.mj_getTotalmass(model) * 1000
    assert 0.7 < mg < 1.3, mg      # femelle adulte : ~0.8-1.2 mg


def test_six_legs_with_seven_active_dof(model):
    for leg in LEGS:
        joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
        active = [j for j in joints if j.startswith(f"{leg}_") and not any(f"tarsus{k}" in j for k in "2345")]
        assert len(active) == 7, active
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_claw") >= 0


def test_every_banc_leg_muscle_has_an_actuator(model):
    for leg in LEGS:
        for muscle in LEG_MUSCLES:
            if muscle == "tergopleural_promotor" and leg[1] != "f" or muscle == "TTMn" and leg[1] != "m":
                continue
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{muscle}") >= 0, (leg, muscle)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_adhesion") >= 0
        for s in ("coxa_yaw_pos", "tibia_pitch_vel", "tarsus5_contact", "load"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{leg}_{s}") >= 0


def test_stands_under_gravity(model):
    d = standing(model)
    for _ in range(3000):        # 300 ms
        mujoco.mj_step(model, d)
    q = d.qpos[3:7]
    up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
    assert up_z > 0.95, "le corps a basculé"
    assert 0.5 < d.qpos[2] < 1.5, d.qpos[2]
    assert np.linalg.norm(d.qvel[:3]) < 2.0
    touching = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) for c in d.contact}
    assert all(f"{leg}_tarsus5" in touching for leg in LEGS), touching
    total = sum(abs(mujoco.mj_contactForce(model, d, i, f := np.zeros(6)) or f[0]) for i in range(d.ncon))
    weight = mujoco.mj_getTotalmass(model) * 9810
    assert abs(total - weight) / weight < 0.15, (total, weight)


def test_muscles_move_the_body(model):
    d = standing(model)
    x0 = d.qpos[0]
    tripod = {"lf": 0.0, "rm": 0.0, "lh": 0.0, "rf": 0.5, "lm": 0.5, "rh": 0.5}
    aid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
    for k in range(10000):
        t = k * model.opt.timestep
        for leg in LEGS:
            swing = (t / 0.1 + tripod[leg]) % 1.0 < 0.35
            for mu in ("trochanter_flexor", "tibia_flexor", "sternal_anterior_rotator"):
                d.ctrl[aid[f"{leg}_{mu}"]] = 0.8 if swing else 0.0
            for mu in ("trochanter_extensor", "tibia_extensor", "sternal_posterior_rotator"):
                d.ctrl[aid[f"{leg}_{mu}"]] = 0.0 if swing else 0.6
            d.ctrl[aid[f"{leg}_adhesion"]] = 0.0 if swing else 1.0
        mujoco.mj_step(model, d)
    assert d.qpos[0] - x0 > 1.0, "le rythme tripode doit faire avancer le corps"
    q = d.qpos[3:7]
    assert 1 - 2 * (q[1] ** 2 + q[2] ** 2) > 0.9
