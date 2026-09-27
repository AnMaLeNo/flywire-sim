"""Boucle fermée cerveau (BANC, LIF) <-> corps (MuJoCo).

À chaque pas (dt commun = 0.1 ms) :
  1. capteurs MuJoCo -> grandeurs normalisées -> spikes forcés des neurones sensoriels de patte (senses.py)
  2. un pas de LIF sur le réseau BANC complet (spikes sensoriels forcés + stimulation expérimentale)
  3. spikes des motoneurones de patte -> activation musculaire -> ctrl des actionneurs (muscles.py)
  4. un pas de MuJoCo
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pandas as pd

from .. import banc, data
from ..lif import LIFNetwork, LIFParams, LIFStepper
from ..network import Network
from .model import LEGS, build_mjcf
from .muscles import Muscles, build_motor_map
from .senses import Senses

NETWORK_FILE = data.PROCESSED / "network_banc888_min5.npz"


def load_banc(min_synapses: int = 5) -> tuple[Network, pd.DataFrame]:
    if NETWORK_FILE.exists() and min_synapses == 5:
        net = Network.load(NETWORK_FILE)
    else:
        net = banc.build(min_synapses)
    n = banc.load_neurons().set_index("root_id").reindex(net.root_ids).reset_index()
    for c in ("function", "body_part", "cell_type", "side", "super_class", "sub_class", "region"):
        n[c] = n[c].fillna("")
    return net, n


@dataclass
class Trace:
    t_ms: list = field(default_factory=list)
    thorax_pos: list = field(default_factory=list)
    n_spikes: list = field(default_factory=list)
    n_mn_spikes: list = field(default_factory=list)
    n_sens_spikes: list = field(default_factory=list)
    ctrl: list = field(default_factory=list)


class BodyBrainSim:
    def __init__(self, params: LIFParams | None = None, seed: int = 0, sugar: float = 0.0):
        self.net, self.neurons = load_banc()
        self.params = params or LIFParams(w_syn=0.21, adapt_b=3.0)
        assert abs(self.params.dt - 0.1) < 1e-9, "dt cerveau = dt physique = 0.1 ms"
        self.lif = LIFNetwork(self.net.W, self.params, seed=seed)
        self.stepper: LIFStepper = self.lif.stepper()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf())
        assert abs(self.model.opt.timestep * 1000 - self.params.dt) < 1e-9
        self.data = mujoco.MjData(self.model)
        self.mmap = build_motor_map(self.model, self.neurons)
        self.muscles = Muscles(self.model, self.mmap, self.params.dt)
        self.senses = Senses(self.model, self.neurons, self.params.dt, seed=seed)
        self.senses.sugar = sugar
        self.is_leg_mn = np.zeros(self.net.W.shape[0], dtype=bool)
        self.is_leg_mn[self.mmap.mn_idx] = True
        self.rng = np.random.default_rng(seed + 1)
        self.reset_pose()

    def reset_pose(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                d.qpos[m.jnt_qposadr[j]] = m.qpos_spring[m.jnt_qposadr[j]]
        mujoco.mj_forward(m, d)
        claws = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_claw") for leg in LEGS]
        d.qpos[2] += 0.03 - min(d.site_xpos[s][2] for s in claws)
        mujoco.mj_forward(m, d)

    def find(self, **query) -> np.ndarray:
        """Indices réseau des neurones dont les colonnes valent / contiennent les valeurs données
        (ex. find(super_class="descending", cell_type=["DNa01", "DNa02"]))."""
        m = np.ones(len(self.neurons), dtype=bool)
        for col, val in query.items():
            s = self.neurons[col]
            m &= s.isin(val).to_numpy() if isinstance(val, (list, tuple, set)) else s.eq(val).to_numpy()
        return np.flatnonzero(m)

    def run(self, duration_ms: float, stim_idx: np.ndarray | None = None, stim_rate_hz: float = 0.0,
            record_every_ms: float = 1.0, render=None, render_every_ms: float = 20.0,
            spike_log: bool = False, progress: bool = False) -> Trace:
        m, d, st = self.model, self.data, self.stepper
        dt = self.params.dt
        n_steps = round(duration_ms / dt)
        rec_every, rend_every = round(record_every_ms / dt), round(render_every_ms / dt)
        p_stim = stim_rate_hz * dt / 1000.0
        tr = Trace()
        self.spike_times, self.spike_neurons = [], []
        acc = np.zeros(3)
        for k in range(n_steps):
            self.senses.read(d)
            forced = [self.senses.spikes()]
            if stim_idx is not None and p_stim > 0:
                forced.append(stim_idx[self.rng.random(stim_idx.size) < p_stim])
            forced = np.concatenate(forced)
            spiking = st.step(forced)
            self.muscles.step(spiking, d)
            mujoco.mj_step(m, d)
            n_mn = int(self.is_leg_mn[spiking].sum())
            acc += (spiking.size, n_mn, forced.size)
            if spike_log and spiking.size:
                self.spike_times.append(np.full(spiking.size, st.t_ms, dtype=np.float32))
                self.spike_neurons.append(spiking.astype(np.int32))
            if k % rec_every == rec_every - 1:
                tr.t_ms.append(st.t_ms)
                tr.thorax_pos.append(d.qpos[:3].copy())
                tr.n_spikes.append(acc[0]); tr.n_mn_spikes.append(acc[1]); tr.n_sens_spikes.append(acc[2])
                tr.ctrl.append(d.ctrl.copy())
                acc[:] = 0
            if render is not None and k % rend_every == 0:
                render(m, d, st.t_ms)
            if progress and k % round(50 / dt) == 0:
                print(f"  t={st.t_ms:6.1f} ms  thorax x={d.qpos[0]:.3f} z={d.qpos[2]:.3f}  "
                      f"spikes/ms={np.mean(tr.n_spikes[-50:]) if tr.n_spikes else 0:.0f}  MN/ms={np.mean(tr.n_mn_spikes[-50:]) if tr.n_mn_spikes else 0:.1f}", flush=True)
        return tr
