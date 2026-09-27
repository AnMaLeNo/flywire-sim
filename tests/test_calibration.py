"""Mécanismes de la calibration ciblée (docs/calibration.md § 2e passe) : dépression par neurone
présynaptique dans le LIF, efficacité des synapses électriques eLN -> PN/eLN. Aucune connexion ni aucun
neurone du BANC ne doit changer."""
import numpy as np
import pytest
import scipy.sparse as sp

from flywire_sim import banc
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus


def _chain(n_pre: int) -> sp.csc_matrix:
    # n_pre neurones présynaptiques (0..n_pre-1) -> un neurone cible (n_pre), 20 synapses chacun
    W = sp.lil_matrix((n_pre + 1, n_pre + 1), dtype=np.float32)
    for j in range(n_pre):
        W[n_pre, j] = 40.0
    return W.tocsc()


def test_per_neuron_depression_only_affects_selected_presynaptic_neurons():
    W = _chain(2)
    p = LIFParams(w_syn=0.5, tau_rec=500.0)
    # neurone 0 dépressif (U=0.5), neurone 1 non ; chacun tire seul à 50 Hz vers la même cible
    for pre, expected_depressed in ((0, True), (1, False)):
        U = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        net = LIFNetwork(W, p, seed=0, std_U=U)
        res = net.run(1000.0, [Stimulus(np.array([pre]), 100.0)])
        r = res.rates_hz()
        ref = LIFNetwork(W, p, seed=0).run(1000.0, [Stimulus(np.array([pre]), 100.0)]).rates_hz()
        assert ref[2] > 10.0
        if expected_depressed:
            assert r[2] < ref[2]
        else:
            assert r[2] == ref[2]


def test_per_neuron_depression_matches_scalar_when_uniform():
    W = _chain(3)
    p = LIFParams(w_syn=0.5, std_U=0.3, tau_rec=300.0)
    a = LIFNetwork(W, p, seed=1).run(500.0, [Stimulus(np.arange(3), 80.0)])
    b = LIFNetwork(W, LIFParams(w_syn=0.5, tau_rec=300.0), seed=1,
                   std_U=np.full(4, 0.3, dtype=np.float32)).run(500.0, [Stimulus(np.arange(3), 80.0)])
    assert np.array_equal(a.spike_times, b.spike_times) and np.array_equal(a.spike_neurons, b.spike_neurons)


@pytest.fixture(scope="module")
def net_neurons():
    from flywire_sim.body.sim import load_banc
    return load_banc()


def test_antennal_lobe_populations(net_neurons):
    net, n = net_neurons
    pops = banc.antennal_lobe_populations(n, net.sign)
    assert len(pops["ORN"]) == 3006 and len(pops["PN"]) == 699
    assert len(pops["eLN"]) + len(pops["iLN"]) == 429
    assert (n.super_class.to_numpy()[pops["ORN"]] == "sensory").all()
    # les eLN sont cholinergiques d'après les annotations officielles (aucun vérifié GABA/glutamate)
    v = n.nt_verified.fillna("").str.lower().to_numpy()[pops["eLN"]]
    assert not np.isin(v, ["gaba", "glutamate"]).any()


def test_depression_only_on_orns(net_neurons):
    net, n = net_neurons
    U = banc.depression_U(n, 0.5)
    pops = banc.antennal_lobe_populations(n, net.sign)
    assert (U[pops["ORN"]] == 0.5).all() and np.count_nonzero(U) == len(pops["ORN"])


def test_synaptic_efficacy_keeps_connectome_support(net_neurons):
    net, n = net_neurons
    W = banc.clamp_afferents(net.W, n)
    W2 = banc.synaptic_efficacy(W, n, net.sign, 0.1)
    A, B = W.tocsr(), W2.tocsr()
    A.eliminate_zeros(); B.eliminate_zeros()
    assert A.nnz == B.nnz and np.array_equal(A.indices, B.indices) and np.array_equal(A.indptr, B.indptr)
    pops = banc.antennal_lobe_populations(n, net.sign)
    ratio = sp.coo_matrix(B.multiply(A.power(-1)))       # poids relatif par connexion
    is_eln_src = np.isin(ratio.col, pops["eLN"])
    is_target = np.isin(ratio.row, np.concatenate([pops["PN"], pops["eLN"]]))
    scaled = is_eln_src & is_target
    assert np.allclose(ratio.data[scaled], 0.1) and np.allclose(ratio.data[~scaled], 1.0)
    assert scaled.sum() > 5000                           # eLN -> PN (6 820) + eLN -> eLN (4 276) connexions
    # eLN -> iLN (chimiques) inchangées
    is_iln = np.isin(ratio.row, pops["iLN"])
    assert np.allclose(ratio.data[is_eln_src & is_iln], 1.0)
