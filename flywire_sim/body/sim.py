"""Boucle fermée cerveau (BANC, LIF) <-> corps (MuJoCo).

À chaque pas (dt commun = 0.1 ms) :
  1. capteurs MuJoCo + environnement -> grandeurs normalisées -> spikes forcés des neurones sensoriels
     (senses.py) ; toutes les `vision_period_ms` : rendu par œil -> photorécepteurs (vision.py), dont la
     sortie est injectée en courant (L1-L3) ou en spikes forcés (R7/R8) à chaque pas
  2. un pas de LIF sur le réseau BANC complet (spikes sensoriels forcés + stimulation expérimentale)
  3. spikes des motoneurones de patte -> activation musculaire -> ctrl des actionneurs (muscles.py)
  4. forces d'environnement (vent, son) sur les antennes, puis un pas de MuJoCo
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pandas as pd

from .. import banc, completeness, data
from ..lif import LIFNetwork, LIFParams, LIFStepper
from ..network import Network
from .environment import Environment
from .model import LEGS, build_mjcf
from .muscles import Muscles, build_motor_map
from .senses import Senses
from .vision import Eyes, build_retinas

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


# traînée de l'ariste : F = WIND_DRAG x (vent - vitesse) [µN par mm/s] ; 500 mm/s -> ~5° de déviation.
# Son : force sinusoïdale d'amplitude SOUND_FORCE x env.sound (1 -> ~1° d'oscillation statique équivalente).
WIND_DRAG = 1.7e-4
SOUND_FORCE = 0.017


class BodyBrainSim:
    def __init__(self, params: LIFParams | None = None, seed: int = 0, sugar: float = 0.0,
                 env: Environment | None = None, vision: bool = False, vision_period_ms: float = 5.0,
                 vision_size: int = 32, extra_xml: str = "", orn_std_u: float = banc.ORN_STD_U,
                 eln_gain: float = banc.ELN_ELECTRICAL_GAIN, completeness_correction: bool = True):
        self.net, self.neurons = load_banc()
        self.params = params or banc.CALIBRATED_V2
        assert abs(self.params.dt - 0.1) < 1e-9, "dt cerveau = dt physique = 0.1 ms"
        W = banc.clamp_afferents(self.net.W, self.neurons)
        if completeness_correction:
            W = completeness.apply(W, self.net.root_ids)
        if eln_gain != 1.0:
            W = banc.synaptic_efficacy(W, self.neurons, self.net.sign, eln_gain)
        self.lif = LIFNetwork(W, self.params, seed=seed, std_U=banc.depression_U(self.neurons, orn_std_u))
        self.stepper: LIFStepper = self.lif.stepper()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(extra_xml=extra_xml))
        assert abs(self.model.opt.timestep * 1000 - self.params.dt) < 1e-9
        self.data = mujoco.MjData(self.model)
        self.mmap = build_motor_map(self.model, self.neurons)
        self.muscles = Muscles(self.model, self.mmap, self.params.dt)
        self.env = env or Environment()
        self.senses = Senses(self.model, self.neurons, self.params.dt, seed=seed, env=self.env)
        self.senses.sugar = sugar
        self.eyes: Eyes | None = None
        if vision:
            retinas = build_retinas(self.neurons, banc.load_positions())
            self.eyes = Eyes(self.model, retinas, period_ms=vision_period_ms, size=vision_size, seed=seed)
        self.vision_period_ms = vision_period_ms
        self.head = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "c_head")
        self.antenna = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_funiculus") for s in ("l", "r")}
        self.arista = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_arista") for s in ("l", "r")}
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

    def apply_environment_forces(self) -> None:
        """Vent (traînée) et son (force oscillante) appliqués à la base de chaque ariste, repère monde."""
        m, d, env = self.model, self.data, self.env
        d.qfrc_applied[:] = 0.0
        if not np.any(env.wind) and env.sound <= 0:
            return
        t = self.stepper.t_ms / 1000.0
        head = d.xmat[self.head].reshape(3, 3)
        for s in ("l", "r"):
            body, site = self.antenna[s], self.arista[s]
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, site, vel, 0)
            f = WIND_DRAG * (np.asarray(env.wind) - vel[3:])
            if env.sound > 0:
                f = f + head @ np.array([1.0, 0.0, 0.0]) * SOUND_FORCE * env.sound * np.sin(2 * np.pi * env.sound_hz * t)
            mujoco.mj_applyFT(m, d, f, np.zeros(3), d.site_xpos[site], body, d.qfrc_applied)

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
        vis_every = max(1, round(self.vision_period_ms / dt))
        for k in range(n_steps):
            self.senses.read(d)
            forced = [self.senses.spikes()]
            currents = None
            if self.eyes is not None:
                if k % vis_every == 0:
                    self.eyes.update(d)
                forced.append(self.eyes.spikes(dt, self.rng))
                currents = self.eyes.currents(dt / self.params.tau_syn)
            if stim_idx is not None and p_stim > 0:
                forced.append(stim_idx[self.rng.random(stim_idx.size) < p_stim])
            forced = np.concatenate(forced)
            spiking = st.step(forced, currents)
            self.muscles.step(spiking, d)
            self.apply_environment_forces()
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
