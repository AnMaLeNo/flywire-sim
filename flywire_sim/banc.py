"""Réseau signé à partir de l'export officiel Codex du BANC v888 (Bates, Phelps, Kim, Yang et al. 2026) :
cerveau + connectif cervical + ganglion ventral d'une même femelle. Même convention que network.py :
W[post, pre] = signe(pre) * nb_synapses(pre -> post).

Le signe vient du NT vérifié quand il existe, sinon du NT prédit (Eckstein et al. 2024).
Les motoneurones sont forcés excitateurs : leurs prédictions NT sont connues pour être peu fiables
(beaucoup prédits GABA alors que les MN de mouche sont glutamatergiques/cholinergiques et
qu'il n'existe pas de MN GABAergique chez la drosophile, Lesser et al. 2024) ; leurs cibles sont
de toute façon des muscles, hors du réseau.
"""
import gzip
import pickle
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data
from .lif import LIFParams
from .network import INHIBITORY_NT, Network

BANC = data.RAW / "banc888"

# Régime retenu par scripts/banc_calibrate.py (voir docs/calibration.md) : DNg100 à 50 Hz recrute
# ~100/391 MN de patte à 10-35 Hz (médiane ~20 Hz, cf. MN lents ~30 Hz, Azevedo et al. 2020), le VNC
# reste stable (~3000 neurones actifs, ~35 Hz) sans emballement du cerveau.
CALIBRATED = LIFParams(w_syn=2.0, size_norm=0.5)


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


# résolution des voxels du volume BANC (x, y, z) en nm ; les positions de `neuron_attributes` sont en voxels
VOXEL_NM = np.array([4.0, 4.0, 45.0])
POSITIONS_FILE = data.PROCESSED / "banc888_positions.csv.gz"


def load_positions() -> pd.DataFrame:
    """Position de référence de chaque neurone (export officiel Codex `neuron_attributes`), en µm.
    Colonnes : root_id, x, y, z. Sert à la rétinotopie (voir body/vision.py)."""
    if POSITIONS_FILE.exists():
        return pd.read_csv(POSITIONS_FILE)
    with gzip.open(BANC / "neuron_attributes.pickle.gz") as f:
        attrs = pickle.load(f)
    rows = []
    for rid, a in attrs.items():
        nums = re.findall(r"-?\d+", str(a.get("position", "")))
        if len(nums) >= 3:
            rows.append((int(rid), *(float(v) for v in nums[:3])))
    pos = pd.DataFrame(rows, columns=["root_id", "x", "y", "z"])
    pos[["x", "y", "z"]] = pos[["x", "y", "z"]].to_numpy() * VOXEL_NM / 1000.0
    data.PROCESSED.mkdir(parents=True, exist_ok=True)
    pos.to_csv(POSITIONS_FILE, index=False)
    return pos


def load_connections() -> pd.DataFrame:
    return pd.read_csv(BANC / "connections_princeton.csv.gz",
                       dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "syn_count": np.int32},
                       usecols=["pre_root_id", "post_root_id", "syn_count"])


def is_inhibitory(nt: np.ndarray) -> np.ndarray:
    """Le NT prédit est en majuscules (GABA, GLUT, HIST...) ; le NT vérifié est en minuscules et peut
    être composé ('gaba,nitric_oxide'). GABA et glutamate sont inhibiteurs dans le SNC de la drosophile
    (GluCl, Liu & Wilson 2013), l'histamine des photorécepteurs aussi (HisCl/Ort, Gengs et al. 2002)."""
    low = pd.Series(nt).str.lower()
    return (low.str.contains("gaba|glutamate|histamine") | low.isin({t.lower() for t in INHIBITORY_NT | {"HIST"}})).to_numpy()


def build(min_synapses: int = 3) -> Network:
    neurons = load_neurons()
    root_ids = np.sort(neurons.root_id.to_numpy())
    idx = pd.Index(root_ids)
    nrn = neurons.set_index("root_id").reindex(root_ids)
    nt = nrn.nt_verified.fillna(nrn.nt_pred).fillna("UNKNOWN").to_numpy(dtype=str)
    sign = np.where(is_inhibitory(nt), -1.0, 1.0).astype(np.float32)
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


def clamp_afferents(W: sp.csc_matrix, neurons: pd.DataFrame) -> sp.csc_matrix:
    """Supprime les entrées synaptiques centrales des neurones sensoriels (lignes de W à zéro).

    Les afférents primaires tirent à partir de leur courant récepteur ; les synapses centrales sur leurs
    terminaisons sont de l'inhibition présynaptique (GABA-B) qui module la libération sans déclencher de
    spikes (Root et al. 2008, Clarke et al. 2015 ; voir docs/calibration.md). Dans le modèle, seuls les
    spikes forcés par les capteurs (senses.py / Stimulus) peuvent les activer."""
    is_sens = neurons.super_class.fillna("").str.startswith("sensory").to_numpy()
    keep = sp.diags((~is_sens).astype(np.float32))
    return (keep @ W).tocsc()
