"""Simulateur maison de neurones « leaky integrate-and-fire » (LIF) pour tout le cerveau.

Modèle (par neurone i) :
    dv_i/dt = (g_i - (v_i - V_rest)) / tau_m
    dg_i/dt = -g_i / tau_syn
    à chaque spike du neurone j (après un délai t_delay) : g_i += W[i, j] * w_syn
    si v_i >= V_thresh : spike, v_i = V_reset, période réfractaire.

Constantes biophysiques issues de l'électrophysiologie de la drosophile telles que
rapportées dans la littérature (V_rest -52 mV, V_thresh -45 mV, R_m 10 kOhm.cm², C_m 2 µF/cm²
=> tau_m 20 ms, réfractaire 2.2 ms, tau_syn 5 ms, délai 1.8 ms). w_syn est le seul paramètre libre.
L'implémentation (numpy/scipy, intégration exacte des exponentielles, tampon circulaire pour le
délai) est entièrement écrite ici.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LIFParams:
    v_rest: float = -52.0      # mV
    v_reset: float = -52.0     # mV
    v_thresh: float = -45.0    # mV
    tau_m: float = 20.0        # ms  (R_m * C_m = 10 kOhm.cm2 * 2 uF/cm2)
    tau_syn: float = 5.0       # ms
    t_refractory: float = 2.2  # ms
    t_delay: float = 1.8       # ms
    w_syn: float = 0.275       # mV par synapse
    dt: float = 0.1            # ms
    # Freins physiologiques optionnels (0 = désactivé) :
    # adaptation de fréquence (courant hyperpolarisant ajouté à chaque spike, relaxe en tau_adapt)
    adapt_b: float = 0.0       # mV par spike
    tau_adapt: float = 100.0   # ms
    # dépression synaptique à court terme (Tsodyks-Markram) : à chaque spike présynaptique une
    # fraction U des ressources x est libérée (poids effectif ∝ U*x), x récupère en tau_rec
    std_U: float = 0.0
    tau_rec: float = 500.0     # ms


@dataclass
class Stimulus:
    """Activation poissonienne des neurones `indices` à `rate_hz`.

    mode="spike" (défaut) : les neurones sont forcés à émettre un spike (c'est l'entrée sensorielle,
    comme une activation optogénétique) ; mode="current" : chaque événement ajoute `amplitude_mv`
    à la conductance synaptique du neurone, qui décide lui-même de tirer."""
    indices: np.ndarray
    rate_hz: float
    mode: str = "spike"
    amplitude_mv: float = 10.0
    t_start: float = 0.0
    t_stop: float = np.inf


@dataclass
class SimResult:
    spike_times: np.ndarray     # ms
    spike_neurons: np.ndarray   # index de neurone
    duration_ms: float
    n_neurons: int

    def rates_hz(self) -> np.ndarray:
        counts = np.bincount(self.spike_neurons, minlength=self.n_neurons)
        return counts / (self.duration_ms / 1000.0)


class LIFNetwork:
    def __init__(self, W: sp.csc_matrix, params: LIFParams | None = None, seed: int = 0):
        # W[post, pre] en synapses signées ; on la garde en CSC pour extraire vite les colonnes des
        # neurones qui ont tiré (W[:, spiking] @ 1).
        self.W = W.tocsc().astype(np.float32)
        self.p = params or LIFParams()
        self.n = W.shape[0]
        self.rng = np.random.default_rng(seed)

    def stepper(self) -> "LIFStepper":
        return LIFStepper(self)

    def run(self, duration_ms: float, stimuli: list[Stimulus], record_v: np.ndarray | None = None,
            progress: bool = False) -> SimResult:
        p, n = self.p, self.n
        dt = p.dt
        n_steps = round(duration_ms / dt)
        st = self.stepper()

        st_idx, st_p, st_amp, st_win, st_force = [], [], [], [], []
        for s in stimuli:
            st_idx.append(np.asarray(s.indices, dtype=np.int64))
            st_p.append(min(1.0, s.rate_hz * dt / 1000.0))
            st_amp.append(np.float32(s.amplitude_mv))
            st_win.append((s.t_start, s.t_stop))
            st_force.append(s.mode == "spike")

        times, neurons = [], []
        v_trace = [] if record_v is not None else None

        for step in range(n_steps):
            t = step * dt
            # entrées externes poissoniennes
            forced, currents = [], []
            for idx, prob, amp, force, (t0, t1) in zip(st_idx, st_p, st_amp, st_force, st_win):
                if t0 <= t < t1:
                    hit = idx[self.rng.random(idx.size) < prob]
                    if force:
                        forced.append(hit)
                    else:
                        currents.append((hit, amp))
            spiking = st.step(np.concatenate(forced) if forced else None, currents)
            if spiking.size:
                times.append(np.full(spiking.size, t, dtype=np.float32))
                neurons.append(spiking.astype(np.int32))
            if v_trace is not None:
                v_trace.append(st.v[record_v].copy())
            if progress and step % int(100 / dt) == 0:
                print(f"  t={t:7.1f} ms  spikes cumulés={sum(len(x) for x in neurons)}", flush=True)

        res = SimResult(
            spike_times=np.concatenate(times) if times else np.empty(0, np.float32),
            spike_neurons=np.concatenate(neurons) if neurons else np.empty(0, np.int32),
            duration_ms=duration_ms,
            n_neurons=n,
        )
        if v_trace is not None:
            res.v_trace = np.stack(v_trace)  # type: ignore[attr-defined]
        return res


class LIFStepper:
    """État du réseau intégré pas à pas (un appel à `step` = un pas `dt`), pour boucler le cerveau avec
    un corps : à chaque pas on force les spikes des neurones sensoriels et on lit les spikes moteurs.
    Même schéma numérique que `LIFNetwork.run` (qui l'utilise)."""

    def __init__(self, net: LIFNetwork):
        p, n = net.p, net.n
        self.net, self.p, self.n = net, p, n
        dt = p.dt
        self.decay_m = np.float32(np.exp(-dt / p.tau_m))
        self.decay_s = np.float32(np.exp(-dt / p.tau_syn))
        # incrément de v dû à g sur un pas, avec g constant sur le pas (schéma exponentiel)
        self.gain = np.float32(1.0 - self.decay_m)
        self.delay_steps = max(1, round(p.t_delay / dt))
        self.refr_steps = round(p.t_refractory / dt)
        self.v = np.full(n, p.v_rest, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.a = np.zeros(n, dtype=np.float32)          # adaptation (mV)
        self.decay_a = np.float32(np.exp(-dt / p.tau_adapt))
        self.x = np.ones(n, dtype=np.float32)           # ressources synaptiques par neurone présynaptique
        self.rec = np.float32(dt / p.tau_rec)
        self.use_std = p.std_U > 0
        self.refr = np.zeros(n, dtype=np.int32)
        # tampon circulaire : incréments de g à appliquer dans `delay_steps` pas
        self.ring = np.zeros((self.delay_steps, n), dtype=np.float32)
        self.step_count = 0

    @property
    def t_ms(self) -> float:
        return self.step_count * self.p.dt

    def step(self, forced: np.ndarray | None = None,
             currents: list[tuple[np.ndarray, float]] | None = None) -> np.ndarray:
        """Avance d'un pas. `forced` : neurones forcés à tirer (entrée sensorielle) ; `currents` :
        (indices, amplitude_mv) ajoutés à la conductance. Retourne les indices des neurones qui tirent."""
        p, v, g, a, x, refr = self.p, self.v, self.g, self.a, self.x, self.refr
        step = self.step_count
        slot = step % self.delay_steps

        # 1) entrées retardées arrivant maintenant
        g += self.ring[slot]
        self.ring[slot] = 0.0

        # 2) entrées externes
        if currents:
            for hit, amp in currents:
                g[hit] += np.float32(amp)

        # 3) intégration exacte sur un pas : v relaxe vers v_rest + g
        v += (p.v_rest + g - a - v) * self.gain
        g *= self.decay_s
        if p.adapt_b:
            a *= self.decay_a
        if self.use_std:
            x += (1.0 - x) * self.rec

        # 4) réfractaire : maintenu au reset
        in_refr = refr > 0
        v[in_refr] = p.v_reset
        refr[in_refr] -= 1

        # 5) spikes (seuil franchi, ou spike forcé par un stimulus hors période réfractaire)
        if forced is not None and forced.size:
            v[forced[refr[forced] == 0]] = p.v_thresh
        spiking = np.flatnonzero(v >= p.v_thresh)
        if spiking.size:
            v[spiking] = p.v_reset
            refr[spiking] = self.refr_steps
            # propagation vers les cibles, appliquée après le délai
            if p.adapt_b:
                a[spiking] += p.adapt_b
            if self.use_std:
                eff = x[spiking].astype(np.float32)   # poids nominal quand x=1
                x[spiking] *= 1.0 - p.std_U
                dg = self.net.W[:, spiking] @ eff
            else:
                dg = self.net.W[:, spiking].sum(axis=1)
            self.ring[(step + self.delay_steps) % self.delay_steps] += np.asarray(dg).ravel() * p.w_syn
        self.step_count += 1
        return spiking
