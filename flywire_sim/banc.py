"""Réseau signé à partir de l'export officiel Codex du BANC v888 (Bates, Phelps, Kim, Yang et al. 2026) :
cerveau + connectif cervical + ganglion ventral d'une même femelle. Même convention que network.py :
W[post, pre] = signe(pre) * nb_synapses(pre -> post).

Le signe vient du NT vérifié quand il existe, sinon du NT prédit (Eckstein et al. 2024).
Les motoneurones sont forcés excitateurs : leurs prédictions NT sont connues pour être peu fiables
(beaucoup prédits GABA alors que les MN de mouche sont glutamatergiques/cholinergiques et
qu'il n'existe pas de MN GABAergique chez la drosophile, Lesser et al. 2024) ; leurs cibles sont
de toute façon des muscles, hors du réseau.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data
from .network import INHIBITORY_NT, Network

BANC = data.RAW / "banc888"


def load_neurons() -> pd.DataFrame:
    n = pd.read_csv(BANC / "neurons.csv.gz", low_memory=False)
    n = n.rename(columns={
        "Root ID": "root_id", "Predicted NT type": "nt_pred", "Verified NT type": "nt_verified",
        "Body Part": "body_part", "Function": "function", "Flow": "flow", "Super Class": "super_class",
        "Class": "cls", "Sub Class": "sub_class", "Nerve": "nerve", "Soma side": "side",
        "Primary Cell Type": "cell_type", "Community labels": "labels", "Top in/out region": "region",
    })
    n["root_id"] = n.root_id.astype(np.int64)
    return n


def load_connections() -> pd.DataFrame:
    return pd.read_csv(BANC / "connections_princeton.csv.gz",
                       dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "syn_count": np.int32},
                       usecols=["pre_root_id", "post_root_id", "syn_count"])


def build(min_synapses: int = 3) -> Network:
    neurons = load_neurons()
    root_ids = np.sort(neurons.root_id.to_numpy())
    idx = pd.Index(root_ids)
    nrn = neurons.set_index("root_id").reindex(root_ids)
    nt = nrn.nt_verified.fillna(nrn.nt_pred).fillna("UNKNOWN").to_numpy(dtype=str)
    sign = np.where(np.isin(nt, list(INHIBITORY_NT)), -1.0, 1.0).astype(np.float32)
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
