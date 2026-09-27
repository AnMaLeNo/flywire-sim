"""Réseau hybride cerveau BANC + moelle MANC (flywire_sim/hybrid.py) : pont déterministe par type et côté,
aucune synapse inventée (chaque entrée de W vient d'une paire officielle BANC ou MANC), signes conservés."""
import numpy as np
import pandas as pd
import pytest

from flywire_sim import banc, hybrid, manc
from flywire_sim.body.sim import load_network


@pytest.fixture(scope="module")
def net_neurons():
    return load_network()


def test_bridge_is_deterministic_and_side_preserving():
    b = banc.load_neurons().head(0)
    b = pd.DataFrame({"super_class": ["descending"] * 3 + ["ascending"] * 2, "cell_type": ["DNx"] * 3 + ["ANy"] * 2,
                      "side": ["left", "left", "right", "left", "right"]})
    m = pd.DataFrame({"super_class": ["descending"] * 4 + ["ascending"] * 3, "cell_type": ["DNx"] * 4 + ["ANy"] * 3,
                      "side": ["left", "right", "right", "right", "left", "left", "center"]})
    br = hybrid.bridge(b, m)
    assert np.array_equal(br.fused, hybrid.bridge(b, m).fused)
    fb, fm = br.fused[:, 0], br.fused[:, 1]
    assert (b.side.to_numpy()[fb] == m.side.to_numpy()[fm]).all() and len(np.unique(fm)) == len(fm) == len(np.unique(fb))
    assert len(br.fused) == 3                            # DNx : 1 gauche + 1 droite ; ANy : 1 gauche
    assert br.unpaired.tolist() == [2, 3, 5, 6]          # DNx droites surnuméraires, ANy gauche surnuméraire, ANy 'center'
    assert hybrid.bridge(b.iloc[:0], m).fused.shape == (0, 2)


def test_hybrid_composition(net_neurons):
    net, n = net_neurons
    assert net.W.shape == (len(n), len(n)) and len(np.unique(net.root_ids)) == len(n)
    assert not n.super_class.eq("ventral_nerve_cord_intrinsic").to_numpy()[n.source.eq("banc").to_numpy()].any()
    assert (n.source.to_numpy()[n.super_class.eq("ventral_nerve_cord_intrinsic").to_numpy()] == "manc").all()
    leg_mn = n[n.super_class.eq("motor") & n.body_part.isin(["front_leg", "middle_leg", "hind_leg"])]
    assert leg_mn.source.eq("manc").all() and len(leg_mn) == 396
    fused = n[n.source.eq("fused")]
    assert fused.super_class.isin(hybrid.BRIDGED).all() and fused.manc_id.is_unique and (fused.manc_id > 0).all()
    assert len(fused[fused.super_class.eq("descending")]) >= 900
    unpaired = n[n.bridge.eq("unpaired")]
    assert unpaired.source.eq("manc").all() and unpaired.super_class.isin(hybrid.BRIDGED).all()
    assert n.bridge.isin(["", "fused", "unpaired"]).all()
    # cerveau intact : les neurones optiques / centraux du BANC sont tous là
    b = banc.load_neurons()
    for sc in ("optic_lobe_intrinsic", "central_brain_intrinsic", "visual_projection"):
        assert n.super_class.eq(sc).sum() == b.super_class.eq(sc).sum()
    # afférents de patte MANC annotés par organe (transfert BANC)
    leg = n[n.super_class.str.startswith("sensory") & n.body_part.isin(["front_leg", "middle_leg", "hind_leg"])]
    assert leg[leg.super_class.eq("sensory")].source.eq("manc").all()   # les ascendants (goût) sont pontés, pas remplacés
    for organ in ("claw_chordotonal_organ_neuron", "hook_chordotonal_organ_neuron", "club_chordotonal_organ_neuron",
                  "hair_plate_neuron", "bristle_neuron", "taste_bristle_gustatory_neuron"):
        assert leg.sub_class.str.endswith(organ).sum() > 50, organ
    assert leg.function.str.contains("sugar").sum() > 100 and leg.function.str.contains("bitter").sum() > 50


def test_no_invented_synapse_and_signs(net_neurons):
    net, n = net_neurons
    rng = np.random.default_rng(0)
    W = net.W.tocoo()
    k = rng.choice(W.nnz, 4000, replace=False)
    post, pre, val = W.row[k], W.col[k], W.data[k]
    src = n.source.to_numpy()
    bid = n.root_id.to_numpy()
    mid = n.manc_id.to_numpy()
    cb = banc.load_connections()
    cm = manc.load_connections()
    pb = cb.groupby(["pre_root_id", "post_root_id"]).syn_count.sum()
    pm = cm.groupby(["pre_root_id", "post_root_id"]).syn_count.sum()
    for i, j, v in zip(post, pre, val):
        in_b = src[i] != "manc" and src[j] != "manc" and (bid[j], bid[i]) in pb.index
        in_m = mid[i] > 0 and mid[j] > 0 and (mid[j], mid[i]) in pm.index
        assert in_b or in_m, (i, j, v)
        assert np.sign(v) == net.sign[j] or (src[j] == "fused" and in_m), (i, j)   # signe MANC possible pour la colonne moelle d'un fusionné
    assert (net.sign[n.super_class.eq("motor").to_numpy()] == 1).all()
    inh = n.nt_pred.fillna("").str.contains("GABA|GLUT", regex=True) & n.source.eq("manc") & ~n.super_class.eq("motor")
    assert (net.sign[inh.to_numpy()] == -1).all()
