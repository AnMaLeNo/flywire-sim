"""Construction du réseau : matrice de poids signée à partir des connexions officielles Codex v783.

Convention : W[post, pre] = signe(pre) * nombre_de_synapses(pre -> post).
Le signe vient du neurotransmetteur dominant prédit par neurone (Eckstein, Bates et al. 2024,
colonne nt_type de neurons.csv) : GABA et glutamate -> inhibiteur (-1), tout le reste -> excitateur (+1).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data

INHIBITORY_NT = {"GABA", "GLUT"}
ANNOTATION_NT = {"acetylcholine": "ACH", "glutamate": "GLUT", "gaba": "GABA", "dopamine": "DA",
                 "serotonin": "SER", "octopamine": "OCT"}


@dataclass
class Network:
    root_ids: np.ndarray           # index -> root_id
    W: sp.csc_matrix               # [post, pre], synapses signées (float32)
    sign: np.ndarray               # +1 / -1 par neurone
    nt_type: np.ndarray            # NT dominant par neurone (str)

    @property
    def n(self) -> int:
        return len(self.root_ids)

    def index_of(self, root_ids) -> np.ndarray:
        pos = pd.Index(self.root_ids).get_indexer(np.asarray(root_ids, dtype=np.int64))
        if (pos < 0).any():
            raise KeyError(f"root_ids absents du réseau : {np.asarray(root_ids)[pos < 0]}")
        return pos

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            root_ids=self.root_ids,
            sign=self.sign,
            nt_type=self.nt_type,
            W_data=self.W.data,
            W_indices=self.W.indices,
            W_indptr=self.W.indptr,
            W_shape=np.array(self.W.shape),
        )

    @classmethod
    def load(cls, path: Path) -> "Network":
        z = np.load(path, allow_pickle=False)
        W = sp.csc_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
        return cls(root_ids=z["root_ids"], W=W, sign=z["sign"], nt_type=z["nt_type"])


def build(min_synapses: int = 5) -> Network:
    neurons = data.load_neurons()
    conns = data.load_connections()

    root_ids = np.sort(neurons.root_id.to_numpy(dtype=np.int64))
    idx = pd.Index(root_ids)

    nt = neurons.set_index("root_id").nt_type.reindex(root_ids)
    # Codex laisse ~20k neurones sans NT dominant ; on complète avec le `top_nt` des annotations
    # systématiques Schlegel et al. (mêmes prédictions Eckstein et al., agrégées par neurone).
    ann = data.load_annotations().set_index("root_id").top_nt.reindex(root_ids)
    nt = nt.fillna(ann.map(ANNOTATION_NT)).fillna("UNKNOWN").to_numpy(dtype=str)
    sign = np.where(np.isin(nt, list(INHIBITORY_NT)), -1.0, 1.0).astype(np.float32)

    # Codex sépare les connexions par neuropile : on agrège par paire (pre, post).
    pairs = conns.groupby(["pre_root_id", "post_root_id"], sort=False).syn_count.sum().reset_index()
    pairs = pairs[pairs.syn_count >= min_synapses]

    pre = idx.get_indexer(pairs.pre_root_id.to_numpy())
    post = idx.get_indexer(pairs.post_root_id.to_numpy())
    keep = (pre >= 0) & (post >= 0)
    pre, post = pre[keep], post[keep]
    w = (pairs.syn_count.to_numpy()[keep] * sign[pre]).astype(np.float32)

    W = sp.csc_matrix((w, (post, pre)), shape=(len(root_ids), len(root_ids)))
    W.sum_duplicates()
    return Network(root_ids=root_ids, W=W, sign=sign, nt_type=nt)
