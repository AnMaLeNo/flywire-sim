"""Moelle (ganglion ventral) : export officiel Codex du MANC v1.2.1 (Takemura, Hayworth et al. 2024 ;
Marin et al. 2024 ; Cheong et al. 2024), mâle adulte. Sert de moelle au réseau hybride (hybrid.py) : le
BANC (femelle) ne récupère qu'environ 0,2x des synapses des interneurones de la moelle (docs/calibration.md
§ 6), alors que le MANC est complet.

Les colonnes sont harmonisées sur celles de banc.load_neurons() : super_class (`intrinsic_neuron` ->
`ventral_nerve_cord_intrinsic`), body_part des MN (classe `fl`/`ml`/`hl`/`wm`/`hm`/`ad`/`nm`) et des
afférents (segment de `Sub Class`), côté (`Soma side`, ou côté du neuropile/nerf pour les afférents).
Même convention de signe que banc.py : NT prédit, GABA/glutamate inhibiteurs, MN forcés excitateurs.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import banc, data
from .network import Network

MANC = data.RAW / "manc121"

SUPER_CLASS = {"intrinsic_neuron": "ventral_nerve_cord_intrinsic", "intrinsic": "ventral_nerve_cord_intrinsic",
               "interneuron": "ventral_nerve_cord_intrinsic", "efferent": "visceral_circulatory",
               "efferent_ascending": "ascending_visceral_circulatory"}
MN_BODY_PART = {"fl": "front_leg", "ml": "middle_leg", "hl": "hind_leg", "wm": "wing", "hm": "haltere",
                "ad": "abdomen", "nm": "neck", "xm": "thoracic_abdominal"}
SENSORY_BODY_PART = {"prothoracic_leg": "front_leg", "mesothoracic_leg": "middle_leg", "metathoracic_leg": "hind_leg",
                     "wing_margin": "wing_margin", "wing": "wing", "haltere": "haltere", "abdomen": "abdomen",
                     "notum": "thorax", "neck": "neck", "ventral_prothorax": "prosternal_organ"}
SENSORY_CLASS = {"mechanosensory_bristle": "bristle_neuron", "campaniform_sensilla": "campaniform_sensillum_neuron",
                 "chordotonal_organ": "chordotonal_organ_neuron", "hair_plate": "hair_plate_neuron",
                 "taste_bristle": "taste_bristle_gustatory_neuron", "strand_receptor": "strand_neuron"}


def load_neurons() -> pd.DataFrame:
    n = pd.read_csv(MANC / "neurons.csv.gz", low_memory=False)
    n = n.rename(columns={
        "Root ID": "root_id", "Predicted NT type": "nt_pred", "Verified NT type": "nt_verified",
        "Body Part": "body_part", "Function": "function", "Flow": "flow", "Super Class": "super_class",
        "Class": "cls", "Sub Class": "sub_class", "Nerve": "nerve", "Soma side": "side",
        "Primary Cell Type": "cell_type", "Community labels": "labels", "Top in/out region": "region",
        "Hemilineage": "hemilineage",
    })
    n["root_id"] = n.root_id.astype(np.int64)
    n["super_class"] = n.super_class.map(lambda s: SUPER_CLASS.get(s, s))
    for c in ("body_part", "function", "side", "cls", "sub_class"):   # colonnes vides lues en float
        n[c] = n[c].astype(object)
    is_mn = n.super_class.eq("motor")
    n.loc[is_mn, "body_part"] = n.loc[is_mn, "cls"].map(MN_BODY_PART)
    is_sens = n.super_class.fillna("").str.startswith("sensory")
    sub = n.sub_class.fillna("")
    seg = sub.str.extract(r"(prothoracic_leg|mesothoracic_leg|metathoracic_leg|wing_margin|wing|haltere|abdomen|notum|neck|ventral_prothorax)")[0]
    n.loc[is_sens, "body_part"] = seg[is_sens].map(SENSORY_BODY_PART)
    side_np = sub.str.extract(r"_(L|R)(?:\.|$)")[0].map({"L": "left", "R": "right"})
    side_nerve = n.nerve.fillna("").str.extract(r"_(L|R)$")[0].map({"L": "left", "R": "right"})
    n.loc[is_sens, "side"] = side_np[is_sens].fillna(side_nerve[is_sens]).fillna(n.side[is_sens])
    # classe d'organe au niveau MANC ; hybrid.transfer_annotations affine (claw/hook/club, goût) par type BANC
    n.loc[is_sens, "cls"] = n.loc[is_sens, "cls"].map(SENSORY_CLASS)
    n.loc[is_sens, "sub_class"] = np.where(n.loc[is_sens, "cls"].notna() & n.loc[is_sens, "body_part"].notna(),
                                           n.loc[is_sens, "body_part"].astype(str) + "_" + n.loc[is_sens, "cls"].astype(str), np.nan)
    n["function"] = pd.Series(np.nan, index=n.index, dtype=object)
    return n


def load_connections() -> pd.DataFrame:
    return pd.read_csv(MANC / "connections_princeton.csv.gz",
                       dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "syn_count": np.int32},
                       usecols=["pre_root_id", "post_root_id", "syn_count"])


def build(min_synapses: int = 5) -> Network:
    neurons = load_neurons()
    root_ids = np.sort(neurons.root_id.to_numpy())
    idx = pd.Index(root_ids)
    nrn = neurons.set_index("root_id").reindex(root_ids)
    nt = nrn.nt_pred.fillna("UNKNOWN").to_numpy(dtype=str)
    sign = np.where(banc.is_inhibitory(nt), -1.0, 1.0).astype(np.float32)
    sign[(nrn.super_class == "motor").to_numpy()] = 1.0

    pairs = load_connections().groupby(["pre_root_id", "post_root_id"], sort=False).syn_count.sum().reset_index()
    pairs = pairs[pairs.syn_count >= min_synapses]
    pre = idx.get_indexer(pairs.pre_root_id.to_numpy())
    post = idx.get_indexer(pairs.post_root_id.to_numpy())
    keep = (pre >= 0) & (post >= 0)
    pre, post = pre[keep], post[keep]
    w = (pairs.syn_count.to_numpy()[keep] * sign[pre]).astype(np.float32)
    W = sp.csc_matrix((w, (post, pre)), shape=(len(root_ids), len(root_ids)))
    W.sum_duplicates()
    return Network(root_ids=root_ids, W=W, sign=sign, nt_type=nt)
