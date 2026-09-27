"""Principe de taille (docs/calibration.md § 9) : le PSP par synapse reçu par un neurone est divisé par sa taille
relative (volume officiel / V_REF), fournie explicitement au LIF ; aucune connexion ne change."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.size import relative_size


def _fan_out() -> sp.csc_matrix:
    # neurone 0 -> neurones 1 et 2, 30 synapses chacun
    W = sp.lil_matrix((3, 3), dtype=np.float32)
    W[1, 0] = 30.0
    W[2, 0] = 30.0
    return W.tocsc()


def test_size_norm_requires_explicit_sizes():
    with pytest.raises(ValueError):
        LIFNetwork(_fan_out(), LIFParams(size_norm=1.0))


def test_larger_neuron_receives_smaller_psp_per_synapse():
    p = LIFParams(w_syn=1.0, size_norm=1.0)
    size = np.array([1.0, 1.0, 4.0])
    net = LIFNetwork(_fan_out(), p, seed=0, size=size)
    res = net.run(200.0, [Stimulus(np.array([0]), 1e5, t_stop=0.1)], record_v=np.array([1, 2]))
    v = np.asarray(res.v_trace) - p.v_rest
    assert v[:, 0].max() > 0
    assert v[:, 1].max() == pytest.approx(v[:, 0].max() / 4.0, rel=1e-3)
    assert np.allclose(net.W[:, 0].toarray().ravel(), [0.0, 30.0, 7.5])


def test_relative_size_manc_table_clamps_truncated_neurons():
    n = pd.DataFrame({"root_id": [1, 2, 3, 4],
                      "super_class": ["ventral_nerve_cord_intrinsic", "motor", "descending", "sensory"],
                      "Volume (nm^3)": [5e11, 1e12, 5e10, np.nan]})
    s = relative_size(n, ref_nm3=2.5e11)
    assert np.allclose(s, [2.0, 4.0, 1.0, 1.0])   # DN tronqué (0,2) borné à 1 ; sans volume -> 1
