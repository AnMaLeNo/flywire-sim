import numpy as np
import pandas as pd
import scipy.sparse as sp

from flywire_sim import banc


def test_verified_lowercase_and_compound_nt_are_inhibitory():
    nt = np.array(["GABA", "GLUT", "gaba", "glutamate", "gaba,nitric_oxide", "HIST", "histamine",
                   "ACH", "acetylcholine", "acetylcholine,histamine", "dopamine", "UNKNOWN"])
    inh = banc.is_inhibitory(nt)
    assert inh.tolist() == [True] * 7 + [False, False, True, False, False]


def test_clamp_afferents_zeroes_sensory_rows_only():
    W = sp.csc_matrix(np.array([[0, 2, 0], [3, 0, -1], [4, 5, 0]], dtype=np.float32))
    neurons = pd.DataFrame({"super_class": ["sensory", "ventral_nerve_cord_intrinsic", np.nan]})
    Wc = banc.clamp_afferents(W, neurons).toarray()
    assert Wc[0].tolist() == [0, 0, 0]
    assert Wc[1:].tolist() == W.toarray()[1:].tolist()
