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

    def run(self, duration_ms: float, stimuli: list[Stimulus], record_v: np.ndarray | None = None,
            progress: bool = False) -> SimResult:
        p, n = self.p, self.n
        dt = p.dt
        n_steps = round(duration_ms / dt)
        decay_m = np.float32(np.exp(-dt / p.tau_m))
        decay_s = np.float32(np.exp(-dt / p.tau_syn))
        # incrément de v dû à g sur un pas, avec g constant sur le pas (schéma exponentiel)
        gain = np.float32(1.0 - decay_m)
        delay_steps = max(1, round(p.t_delay / dt))
        refr_steps = round(p.t_refractory / dt)

        v = np.full(n, p.v_rest, dtype=np.float32)
        g = np.zeros(n, dtype=np.float32)
        refr = np.zeros(n, dtype=np.int32)
        # tampon circulaire : incréments de g à appliquer dans `delay_steps` pas
        ring = np.zeros((delay_steps, n), dtype=np.float32)

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
            slot = step % delay_steps

            # 1) entrées retardées arrivant maintenant
            g += ring[slot]
            ring[slot] = 0.0

            # 2) entrées externes poissoniennes
            forced = []
            for idx, prob, amp, force, (t0, t1) in zip(st_idx, st_p, st_amp, st_force, st_win):
                if t0 <= t < t1:
                    hit = idx[self.rng.random(idx.size) < prob]
                    if force:
                        forced.append(hit)
                    else:
                        g[hit] += amp

            # 3) intégration exacte sur un pas : v relaxe vers v_rest + g
            v += (p.v_rest + g - v) * gain
            g *= decay_s

            # 4) réfractaire : maintenu au reset
            in_refr = refr > 0
            v[in_refr] = p.v_reset
            refr[in_refr] -= 1

            # 5) spikes (seuil franchi, ou spike forcé par un stimulus hors période réfractaire)
            if forced:
                f = np.concatenate(forced)
                v[f[refr[f] == 0]] = p.v_thresh
            spiking = np.flatnonzero(v >= p.v_thresh)
            if spiking.size:
                v[spiking] = p.v_reset
                refr[spiking] = refr_steps
                # propagation vers les cibles, appliquée après le délai
                dg = self.W[:, spiking].sum(axis=1)
                ring[(step + delay_steps) % delay_steps] += np.asarray(dg).ravel() * p.w_syn
                times.append(np.full(spiking.size, t, dtype=np.float32))
                neurons.append(spiking.astype(np.int32))

            if v_trace is not None:
                v_trace.append(v[record_v].copy())
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
