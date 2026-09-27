"""Réseau hybride cerveau FAFB v783 + moelle MANC (brain="fafb") : pont par type MANC-compatible (`bridge_type`,
pierre de Rosette BANC), aucun neurone FAFB retiré, aucune synapse inventée, capteurs de tête annotés."""
import numpy as np
import pandas as pd
import pytest

from flywire_sim import data, fafb, hybrid, manc
from flywire_sim.body.sim import load_network
from flywire_sim.body.vision import build_retinas


@pytest.fixture(scope="module")
def net_neurons():
    return load_network(brain="fafb")


def test_bridge_types_majority_only():
    ftypes, mtypes = {"DNg100", "AN_GNG_1", "AN_GNG_2", "SA_x", "AN_GNG_3"}, {"DNg100", "AN07B072", "AN10B046"}
    b = pd.DataFrame({
        "super_class": ["descending", "ascending", "ascending", "ascending", "ascending", "ascending", "ascending", "central"],
        "cell_type": ["DNg100", "AN07B072", "AN07B072", "AN10B046", "AN07B072", "AN10B046", "AN10B046", "AN07B072"],
        "Alternative Cell Type(s)": [None, "AN_GNG_1", None, None, "AN_GNG_2", "AN_GNG_2", None, "AN_GNG_3"],
        "labels": [None, None, "AN_GNG_1", "AN_GNG_1", None, None, "SA_x", None],
    })
    t = fafb.bridge_types(b, ftypes, mtypes)
    assert t["DNg100"] == "DNg100"                 # nomenclature déjà commune
    assert t["AN_GNG_1"] == "AN07B072"             # 2/3 -> majorité
    assert "AN_GNG_2" not in t                     # 1 vs 1 : pas de majorité, pas de pont
    assert t["SA_x"] == "AN10B046"                 # 1/1
    assert "AN_GNG_3" not in t                     # seuls les neurones pontés du BANC servent de pierre de Rosette


def test_fafb_neurons_schema():
    n = fafb.load_neurons()
    assert n.root_id.is_unique and set(hybrid.COLUMNS) <= set(n.columns)
    s = n[n.super_class.eq("sensory")]
    for sub in ("antenna_olfactory_receptor_neuron", "johnstons_organ_A_neuron", "johnstons_organ_B_neuron",
                "johnstons_organ_E_neuron", "retina_photoreceptor_neuron", "interommatidial_bristle_neuron",
                "labellum_taste_bristle_gustatory_neuron", "antenna_hygrosensory_receptor_neuron"):
        assert s.sub_class.eq(sub).sum() > 20, sub
    assert s.function.str.contains("sugar", na=False).sum() > 100 and s.function.str.contains("cooling", na=False).sum() > 5
    assert s.side.isin(["left", "right"]).mean() > 0.99
    dn = n[n.super_class.eq("descending")]
    assert dn.bridge_type.notna().sum() > 1000 and (dn.bridge_type.dropna() == dn.cell_type.dropna().reindex(dn.bridge_type.dropna().index)).mean() > 0.4
    pos = fafb.load_positions()
    assert {"root_id", "x", "y", "z"} <= set(pos.columns) and pos.root_id.is_unique


def test_hybrid_fafb_composition(net_neurons):
    net, n = net_neurons
    assert net.W.shape == (len(n), len(n)) and len(np.unique(net.root_ids)) == len(n)
    f = fafb.load_neurons()
    for sc in ("optic_lobe_intrinsic", "central_brain_intrinsic", "visual_projection", "sensory", "motor", "descending",
               "ascending", "sensory_ascending"):           # aucun neurone du FAFB retiré (pas de moelle dans le FAFB)
        assert n[n.source.isin(["fafb", "fused"])].super_class.eq(sc).sum() == f.super_class.eq(sc).sum(), sc
    assert (n.source.to_numpy()[n.super_class.eq("ventral_nerve_cord_intrinsic").to_numpy()] == "manc").all()
    leg_mn = n[n.super_class.eq("motor") & n.body_part.isin(["front_leg", "middle_leg", "hind_leg"])]
    assert leg_mn.source.eq("manc").all() and len(leg_mn) == 396
    fused = n[n.source.eq("fused")]
    assert fused.super_class.isin(hybrid.BRIDGED).all() and fused.manc_id.is_unique and (fused.manc_id > 0).all()
    assert fused.root_id.is_unique and 900 <= fused.super_class.eq("descending").sum() <= 1000
    assert 500 <= fused.super_class.eq("ascending").sum() <= 800 and fused.super_class.eq("sensory_ascending").sum() < 100
    assert n.bridge.isin(["", "fused", "unpaired"]).all()
    unpaired = n[n.bridge.eq("unpaired")]
    assert unpaired.source.eq("manc").all() and unpaired.super_class.isin(hybrid.BRIDGED).all()
    assert (fused.side.to_numpy() == manc.load_neurons().set_index("root_id").side.reindex(fused.manc_id).to_numpy()).all()


def test_fafb_no_invented_synapse_and_signs(net_neurons):
    net, n = net_neurons
    rng = np.random.default_rng(1)
    W = net.W.tocoo()
    k = rng.choice(W.nnz, 4000, replace=False)
    post, pre, val = W.row[k], W.col[k], W.data[k]
    src, bid, mid = n.source.to_numpy(), n.root_id.to_numpy(), n.manc_id.to_numpy()
    pf = data.load_connections().groupby(["pre_root_id", "post_root_id"]).syn_count.sum()
    pm = manc.load_connections().groupby(["pre_root_id", "post_root_id"]).syn_count.sum()
    for i, j, v in zip(post, pre, val):
        in_f = src[i] != "manc" and src[j] != "manc" and (bid[j], bid[i]) in pf.index
        in_m = mid[i] > 0 and mid[j] > 0 and (mid[j], mid[i]) in pm.index
        assert in_f or in_m, (i, j, v)
        assert np.sign(v) == net.sign[j] or (src[j] == "fused" and in_m), (i, j)
    assert (net.sign[n.super_class.eq("motor").to_numpy()] == 1).all()
    inh = n.nt_pred.fillna("").str.contains("GABA|GLUT", regex=True) & ~n.super_class.eq("motor") & ~n.source.eq("fused")
    assert (net.sign[inh.to_numpy()] == -1).all()


def test_fafb_retinas_both_eyes(net_neurons):
    _, n = net_neurons
    rets = build_retinas(n, fafb.load_positions())
    assert [r.side for r in rets] == ["l", "r"]
    axes = {}
    for r in rets:
        assert len(r.idx) > 5000 and {"R7", "R8", "L1", "L2", "Mi1"} <= set(r.types)
        assert (n.side.to_numpy()[r.idx] == {"l": "left", "r": "right"}[r.side]).all()
        axes[r.side] = r.dirs.mean(axis=0) / np.linalg.norm(r.dirs.mean(axis=0))
    assert axes["l"][0] > 0.5 and axes["r"][0] > 0.5                # les deux yeux regardent vers l'avant...
    assert axes["l"][1] > 0.5 and axes["r"][1] < -0.5               # ... et chacun de son côté (y = gauche)
    assert abs(axes["l"][2]) < 0.2 and abs(axes["r"][2]) < 0.2
