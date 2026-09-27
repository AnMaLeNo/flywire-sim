"""Unités motrices (motor_units.py, muscles.py) : lois volume -> classe / force / cinétique ajustées sur
Azevedo et al. 2020, et leur mise en œuvre dans la conversion spikes -> couple."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flywire_sim.body import motor_units as mu
from flywire_sim.body.model import build_mjcf
from flywire_sim.body.muscles import MotorMap, Muscles
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus

V_FAST, V_INT, V_SLOW = mu.V_FAST_NM3, 7.2e11, mu.V_SLOW_NM3   # rangs 1, 2-4 et accessoires distaux du pool


def test_laws_reproduce_azevedo_2020_measurements():
    # force par spike au bout du tibia : ~10 µN (rapide), ~1 µN (intermédiaire), 0,013 µN (lent)
    f = mu.torque_per_spike(np.array([V_FAST, V_INT, V_SLOW])) / mu.TIBIA_MM
    assert f[0] == pytest.approx(10.0)
    assert 0.5 < f[1] < 3.0
    assert 0.005 < f[2] < 0.03
    assert mu.torque_per_spike(np.array([5 * V_FAST]))[0] == mu.T_FAST   # pas d'extrapolation au-delà du rapide
    # potentiels de repos -68 / -60 / -48 mV
    v = mu.rest_potential(np.array([V_FAST, V_INT, V_SLOW]))
    assert v[0] == pytest.approx(-66, abs=2.5) and v[1] == pytest.approx(-60, abs=2.5) and v[2] == pytest.approx(-48, abs=1)
    assert list(mu.mn_class(np.array([V_FAST, V_INT, V_SLOW]))) == ["fast", "intermediate", "slow"]
    # résistance d'entrée relative : 700 / 150 MΩ entre lent et rapide
    g = mu.input_gain(np.array([V_SLOW, V_FAST]), 2.5e11)
    assert g[0] / g[1] == pytest.approx(700 / 150) and g[0] > 1 > g[1]
    # gradient monotone : plus gros -> plus de force, saturation plus précoce, secousse plus brève
    vs = np.geomspace(V_SLOW, V_FAST, 20)
    assert (np.diff(mu.torque_per_spike(vs)) > 0).all()
    assert (np.diff(mu.twitch_tetanus_ratio(vs)) < 0).all()
    assert all((np.diff(t) < 0).all() for t in mu.twitch_tau_ms(vs))


def _one_unit_model():
    model = mujoco.MjModel.from_xml_string(build_mjcf())
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lf_tibia_flexor")
    return model, act


def _run(volume, spike_steps, n_steps=1500, dt=0.1):
    model, act = _one_unit_model()
    mmap = MotorMap(np.array([0]), np.array([act]), np.array(["lf"]), np.array([volume]), pd.DataFrame())
    m = Muscles(model, mmap, dt, adhesion_from_ltm=False)
    d = mujoco.MjData(model)
    torque = np.empty(n_steps)
    for k in range(n_steps):
        m.step(np.array([0]) if k in spike_steps else np.empty(0, dtype=np.int64), d)
        torque[k] = m.torque[0]
        assert d.ctrl[act] == pytest.approx(min(1.0, m.torque[0] / abs(model.actuator_gear[act, 0])))
    return torque


def test_fast_twitch_kinetics_and_summation():
    dt = 0.1
    one = _run(V_FAST, {0})
    assert one.max() == pytest.approx(mu.T_FAST, rel=1e-3)          # une secousse = force par spike
    assert 12 <= one.argmax() * dt <= 25                            # pic ~20 ms (Fig. 4A)
    assert one[int(70 / dt)] < 0.1 * one.max()                      # retour à la base en ~60-70 ms
    two = _run(V_FAST, {0, int(8 / dt)})
    assert 1.4 < two.max() / one.max() < 1.9                        # 2 spikes -> 1,6 x un spike (Fig. 4E)
    train = _run(V_FAST, set(range(0, int(200 / dt), int(10 / dt))))   # 20 spikes à 100 Hz
    tet = mu.tetanic_torque(np.array([V_FAST]))[0]
    assert 2.0 < train.max() / one.max() <= tet / one.max() + 1e-9  # plateau 2-3,5 x un spike (Fig. 4D), borné par le tétanos


def test_slow_unit_sums_linearly_and_relaxes_slowly():
    dt = 0.1
    rate_hz = 30.0                                                  # activité de repos des MN lents
    steps = set(range(0, int(1000 / dt), int(1000 / rate_hz / dt)))
    slow = _run(V_SLOW, steps, n_steps=int(1500 / dt))
    sustained = slow[int(800 / dt):int(1000 / dt)].mean()
    # ~11 secousses intégrées (τ relaxation 300 ms), loin de la saturation (A = 50) : sommation ~linéaire
    assert 5 * mu.torque_per_spike(V_SLOW) < sustained < 20 * mu.torque_per_spike(V_SLOW)
    assert slow[int(1300 / dt)] < 0.5 * sustained                   # relaxation en quelques centaines de ms
    assert slow[int(1500 / dt) - 1] > 0.05 * sustained


def test_front_leg_flexor_pool_matches_measured_maximal_force():
    from flywire_sim import manc
    m = manc.load_neurons()
    pool = m[(m.super_class == "motor") & (m.body_part == "front_leg")
             & m.cell_type.isin(["tibia_flexor", "accessory_tibia_flexor"])]
    for side in ("left", "right"):
        v = pool[pool.side == side]["Volume (nm^3)"].astype(float).to_numpy()
        assert len(v) >= 12
        tet = mu.tetanic_torque(v).sum() / mu.TIBIA_MM       # µN au bout du tibia
        assert 60 < tet < 160, tet                           # ~100 µN mesurés (Azevedo 2020, Fig. 1)
        assert (mu.mn_class(v) == "fast").sum() >= 1     # un MN rapide par pool (Azevedo 2020)
        assert (mu.mn_class(v) == "slow").sum() >= 1 and (mu.mn_class(v) == "intermediate").sum() >= 2


def test_lif_accepts_per_neuron_rest_potential():
    W = sp.csc_matrix((3, 3), dtype=np.float32)
    p = LIFParams()
    v_rest = np.array([-52.0, -60.0, -48.0])
    net = LIFNetwork(W, p, v_rest=v_rest)
    res = net.run(50.0, [], record_v=np.arange(3))
    assert np.allclose(res.v_trace[-1], v_rest, atol=1e-3)
    with pytest.raises(AssertionError):
        LIFNetwork(W, p, v_rest=np.array([-52.0]))
    # gain d'entrée par neurone : multiplie la ligne (entrées) de W
    W1 = sp.csc_matrix(np.array([[0, 0], [2, 0]], dtype=np.float32))
    assert LIFNetwork(W1, p, gain=np.array([1.0, 0.25])).W[1, 0] == pytest.approx(0.5)
    # un MN « lent » à -48 mV est recruté par une entrée que le même neurone à -68 mV ignore
    W2 = sp.csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32))
    net2 = LIFNetwork(W2, LIFParams(w_syn=25.0), v_rest=np.array([-52.0, -68.0, -48.0]))
    r = net2.run(100.0, [Stimulus(np.array([0]), 1e5, t_stop=0.1)]).rates_hz()
    assert r[2] > 0 and r[1] == 0
