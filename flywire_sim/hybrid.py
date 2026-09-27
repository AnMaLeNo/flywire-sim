"""Réseau hybride : cerveau du BANC ou du FAFB (femelles) + moelle du MANC (mâle), pont par type cellulaire.

Pourquoi : les entrées des interneurones de la moelle du BANC ne sont récupérées qu'à ~0,2x (docs/calibration.md
§ 6), ce qui éteint la voie DN -> prémoteurs -> MN ; le MANC est complet. On garde donc le BANC pour tout ce
qui est dans le cerveau et on remplace ses neurones résidents de la moelle (interneurones, MN et afférents
des pattes/ailes/abdomen...) par le MANC entier.

Pont : les neurones qui traversent le connectif (DN, AN, afférents ascendants) existent dans les deux jeux
sous le même nom de type (nomenclature commune Cheong et al. 2024, Stürner et al. 2025). Chaque paire
(type, côté) est fusionnée en un seul neurone : entrées du cerveau (lignes BANC) + entrées de la moelle
(lignes MANC), sorties dans le cerveau (colonne BANC) + sorties dans la moelle (colonne MANC). Quand un type
compte plus de neurones dans un jeu que dans l'autre, les surnuméraires du MANC reçoivent une copie des
entrées cérébrales (DN) ou des sorties cérébrales (AN) d'un frère de même type et côté ; les surnuméraires du
BANC restent des neurones du cerveau sans prolongement dans la moelle. Aucune synapse n'est inventée : chaque
entrée de W vient d'une paire (pré, post) officielle de l'un des deux exports.

Cerveau FAFB v783 (brain="fafb", docs/calibration.md § 8) : mêmes règles ; le type de pont d'un neurone FAFB
est `bridge_type` (fafb.bridge_types : nomenclature MANC via le BANC), et le FAFB n'ayant pas de moelle,
aucun neurone n'en est retiré.

Annotations des afférents : le MANC n'annote les afférents qu'au niveau de la classe (soie, campaniforme,
chordotonal, plaque pilifère, soie gustative) ; le BANC les annote par organe (claw/hook/club, tarse...) et
fonction gustative. Les deux jeux partagent les types (SNta29, SNpp50...) : on transfère au MANC, type par type,
l'annotation majoritaire du BANC (pureté médiane 0,99).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import banc, data, fafb, manc
from .network import Network

BRAINS = {"banc": banc, "fafb": fafb}

BRIDGED = ("descending", "ascending", "sensory_ascending", "ascending_visceral_circulatory")
VNC_REGIONS = {"T1_PRONM", "T2_MESONM", "T3_METANM", "ABDNM", "HTCT", "INTTCT", "WTCT", "AMNP", "NTCT", "LTCT",
               "T2_MVAC", "WTCT_UTCT_T2", "HTCT_UTCT_T3"}
VNC_BODY_PARTS = {"front_leg", "middle_leg", "hind_leg", "wing", "wing_margin", "wing_base", "wing_tegula", "haltere",
                  "abdomen", "abdominal_wall", "thorax", "prosternal_organ", "thoracic_abdominal", "uterus",
                  "reproductive_tract", "notum"}
COLUMNS = ("root_id", "super_class", "cls", "sub_class", "function", "body_part", "side", "cell_type", "region",
           "hemilineage", "nt_pred", "nt_verified", "flow", "nerve", "labels", "bridge_type")


def network_file(brain: str) -> Path:
    return data.PROCESSED / ("network_hybrid_min5.npz" if brain == "banc" else f"network_hybrid_{brain}_min5.npz")


def neurons_file(brain: str) -> Path:
    return data.PROCESSED / ("neurons_hybrid.csv.gz" if brain == "banc" else f"neurons_hybrid_{brain}.csv.gz")


def vnc_resident(b: pd.DataFrame) -> np.ndarray:
    """Neurones du BANC remplacés par le MANC : interneurones de la moelle, et MN / afférents / non classés
    dont le neuropile principal ou la partie du corps est dans la moelle. Les classes pontées restent."""
    sc = b.super_class.fillna("")
    reg = b.region.fillna("").str.split(".").str[0]
    in_vnc = reg.isin(VNC_REGIONS) | b.body_part.fillna("").isin(VNC_BODY_PARTS)
    return (sc.eq("ventral_nerve_cord_intrinsic") | (~sc.isin(BRIDGED) & in_vnc)).to_numpy()


def transfer_annotations(m: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Affine cls / sub_class / function des afférents du MANC avec l'annotation BANC majoritaire de leur type."""
    bs = b[b.super_class.fillna("").str.startswith("sensory") & b.cell_type.notna()].copy()
    part = bs.body_part.fillna("")
    organ = [s[len(p) + 1:] if isinstance(s, str) and p and s.startswith(p + "_") else s
             for s, p in zip(bs.sub_class, part)]
    bs["organ"] = organ

    def majority(x: pd.Series) -> str | float:
        x = x.dropna()
        return x.value_counts().index[0] if len(x) else np.nan

    ref = bs.groupby("cell_type").agg(cls=("cls", majority), organ=("organ", majority), function=("function", majority))
    m = m.copy()
    is_sens = m.super_class.fillna("").str.startswith("sensory")
    t = m.loc[is_sens, "cell_type"]
    cls = t.map(ref.cls)
    organ = t.map(ref.organ)
    m.loc[is_sens, "cls"] = cls.fillna(m.loc[is_sens, "cls"])
    bp = m.loc[is_sens, "body_part"]
    sub = np.where(organ.notna() & bp.notna(), bp.astype(str) + "_" + organ.astype(str), organ)
    m.loc[is_sens, "sub_class"] = pd.Series(sub, index=t.index).fillna(m.loc[is_sens, "sub_class"])
    m.loc[is_sens, "function"] = t.map(ref.function)
    return m


@dataclass
class Bridge:
    fused: np.ndarray       # [k, 2] : (index BANC, index MANC) fusionnés en un seul noeud
    unpaired: np.ndarray    # [k] : index MANC des DN/AN sans homologue BANC (type absent, ou surnuméraires) ;
                            # ils restent dans la moelle sans aucune connexion cérébrale (rien n'est inventé)


def bridge(b: pd.DataFrame, m: pd.DataFrame) -> Bridge:
    """`b` : neurones du cerveau ; leur type de pont est `bridge_type` s'il existe, sinon `cell_type`."""
    fused, unpaired = [], []
    btype = b.bridge_type if "bridge_type" in b.columns else b.cell_type
    bb = b[b.super_class.isin(BRIDGED) & btype.notna()].assign(cell_type=btype)
    mm = m[m.super_class.isin(BRIDGED)]
    groups_b = {k: v.index.to_numpy() for k, v in bb.groupby(["super_class", "cell_type", "side"])}
    for key, mi in mm.groupby(["super_class", "cell_type", "side"], dropna=False).groups.items():
        mi = np.asarray(list(mi))
        bi = groups_b.get(key)
        k = 0 if bi is None else min(len(bi), len(mi))
        fused.extend(zip(bi[:k], mi[:k]) if k else ())
        unpaired.extend(mi[k:])
    return Bridge(np.array(fused, dtype=np.int64).reshape(-1, 2), np.array(sorted(unpaired), dtype=np.int64))


def build(min_synapses: int = 5, brain: str = "banc") -> tuple[Network, pd.DataFrame]:
    mod = BRAINS[brain]
    nb_net = mod.build(min_synapses)
    nm_net = manc.build(min_synapses)
    b = mod.load_neurons().set_index("root_id").reindex(nb_net.root_ids).reset_index()
    ref = b if brain == "banc" else banc.load_neurons()     # annotations d'organe des afférents : toujours le BANC
    m = transfer_annotations(manc.load_neurons(), ref).set_index("root_id").reindex(nm_net.root_ids).reset_index()
    keep_b = ~vnc_resident(b) if brain == "banc" else np.ones(len(b), dtype=bool)
    br = bridge(b, m)

    # indices de noeuds : BANC conservés d'abord (ordre BANC), puis neurones MANC non fusionnés
    node_b = np.full(len(b), -1, dtype=np.int64)
    node_b[keep_b] = np.arange(keep_b.sum())
    assert (node_b[br.fused[:, 0]] >= 0).all()
    node_m = np.full(len(m), -1, dtype=np.int64)
    node_m[br.fused[:, 1]] = node_b[br.fused[:, 0]]
    standalone = np.flatnonzero(node_m < 0)
    node_m[standalone] = keep_b.sum() + np.arange(len(standalone))
    n_nodes = keep_b.sum() + len(standalone)

    Wb = nb_net.W.tocoo()
    Wm = nm_net.W.tocoo()
    rows = [node_b[Wb.row], node_m[Wm.row]]
    cols = [node_b[Wb.col], node_m[Wm.col]]
    vals = [Wb.data, Wm.data]
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    ok = (r >= 0) & (c >= 0)
    W = sp.csc_matrix((v[ok].astype(np.float32), (r[ok], c[ok])), shape=(n_nodes, n_nodes))
    W.sum_duplicates()

    cols_b = [c for c in COLUMNS if c in b.columns]
    nb_tab = b.loc[keep_b, cols_b].copy()
    nb_tab["source"] = brain
    nb_tab["manc_id"] = -1
    nb_tab.loc[nb_tab.index[node_b[br.fused[:, 0]]], "manc_id"] = m.root_id.to_numpy()[br.fused[:, 1]]
    nb_tab.loc[nb_tab.index[node_b[br.fused[:, 0]]], "source"] = "fused"
    nm_tab = m.loc[standalone, [c for c in COLUMNS if c in m.columns]].copy()
    nm_tab["source"] = "manc"
    nm_tab["manc_id"] = nm_tab.root_id
    bridge_kind = pd.Series("", index=m.index)
    bridge_kind.iloc[br.unpaired] = "unpaired"
    nm_tab["bridge"] = bridge_kind.iloc[standalone].to_numpy()
    nb_tab["bridge"] = np.where(nb_tab.source.eq("fused"), "fused", "")
    neurons = pd.concat([nb_tab, nm_tab], ignore_index=True)

    sign = np.concatenate([nb_net.sign[keep_b], nm_net.sign[standalone]])
    nt = np.concatenate([nb_net.nt_type[keep_b], nm_net.nt_type[standalone]])
    net = Network(root_ids=neurons.root_id.to_numpy(dtype=np.int64), W=W, sign=sign, nt_type=nt)
    return net, neurons


def load(min_synapses: int = 5, brain: str = "banc") -> tuple[Network, pd.DataFrame]:
    nf, tf = network_file(brain), neurons_file(brain)
    if min_synapses == 5 and nf.exists() and tf.exists():
        return Network.load(nf), pd.read_csv(tf, low_memory=False)
    net, neurons = build(min_synapses, brain)
    if min_synapses == 5:
        net.save(nf)
        neurons.to_csv(tf, index=False)
    return net, neurons
