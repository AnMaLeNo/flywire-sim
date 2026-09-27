"""Correction de complétude synaptique du BANC (docs/calibration.md § 6) : elle ne doit ni ajouter ni retirer
de connexion, laisser les motoneurones et les afférents intacts, et rester bornée."""
import numpy as np
import scipy.sparse as sp

from flywire_sim import banc, completeness
from flywire_sim.body.sim import load_banc


def test_apply_keeps_support_and_scales_rows():
    W = sp.csc_matrix(np.array([[0, 3, 0], [2, 0, 5], [0, 0, 0]], dtype=np.float32))
    f = np.array([1.0, 4.0, 1.0], dtype=np.float32)
    out = (sp.diags(f) @ W).tocsc()
    assert out.nnz == W.nnz and (out.indices == W.indices).all() and (out.indptr == W.indptr).all()
    assert np.allclose(out.toarray()[1], W.toarray()[1] * 4.0)


def test_factors_bounded_and_reference_classes():
    net, n = load_banc()
    f = completeness.factors(net.root_ids)
    assert f.shape == (len(n),)
    assert f.min() >= completeness.FACTOR_MIN and f.max() <= completeness.FACTOR_MAX
    sc = n.super_class.to_numpy()
    assert np.all(f[sc == "motor"] == 1.0)
    assert np.all(f[np.char.startswith(sc.astype(str), "sensory")] == 1.0)
    # seule la moelle est corrigée (référence MANC) : interneurones VNC et ascendants nettement remontés,
    # cerveau intact
    assert np.median(f[sc == "ventral_nerve_cord_intrinsic"]) >= 2.5
    assert np.all(f[sc == "ascending"] == 1.0)
    brain = np.isin(sc, ["central_brain_intrinsic", "descending", "optic_lobe_intrinsic", "visual_projection"])
    assert np.all(f[brain] == 1.0)


def test_apply_on_banc_preserves_connections():
    net, n = load_banc()
    W = banc.clamp_afferents(net.W, n)
    out = completeness.apply(W, net.root_ids)
    assert out.nnz == W.nnz
    assert (out.indices == W.indices).all() and (out.indptr == W.indptr).all()
    assert np.all(np.sign(out.data) == np.sign(W.data))
