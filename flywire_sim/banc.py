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
# CALIBRATED (PR #2, « v1 ») avait été réglé sur DNg100 -> MN avec afférents silencieux. Dès que les capteurs
# tirent (ORN ~8 Hz, hygro/thermo toniques), son gain global (w x7 par rapport à Shiu) embrase le cerveau
# entier ; la 2e passe (docs/calibration.md § 5) revient aux paramètres de Shiu et al. 2024 et n'ajoute que
# des mécanismes spécifiques du lobe antennaire (ci-dessous). Le gain DN -> MN devra être traité dans le
# ganglion ventral lui-même (propriétés des MN), pas par un gain global.
CALIBRATED_V2 = LIFParams(tau_rec=200.0)

# Mécanismes spécifiques du lobe antennaire (docs/calibration.md § 2e passe). Ils ne touchent ni aux
# neurones ni aux connexions du BANC : seule l'efficacité par spike de certaines synapses change.
#  - Dépression à court terme des terminaisons ORN (Kazama & Wilson 2008 : probabilité de libération
#    élevée, uEPSC déjà réduit de ~40 % à la fréquence spontanée (7 Hz), dépression forte et rapide pendant les
#    trains odorants). Tsodyks-Markram par neurone présynaptique : fraction U libérée par spike, récupération
#    en ORN_TAU_REC (= tau_rec des paramètres LIF) ; à 8 Hz spontanés x_ss = 1/(1+U·f·tau) ~ 0.7 (Kazama & Wilson 2008 : ~40 % de dépression à 7 Hz), à 80 Hz ~ 0.17.
ORN_STD_U = 0.3
ORN_TAU_REC = CALIBRATED_V2.tau_rec
#  - Les LN excitateurs (eLN, cholinergiques) excitent les PN (et les autres eLN) par jonctions électriques,
#    pas par transmission chimique (Yaksi & Wilson 2010 : insensible au Cd2+, aboli par shakB2 ;
#    Huang et al. 2010 : couplage eLN-PN et eLN-eLN). Un spike d'eLN (~40 mV, souvent atténué à ~10 mV)
#    ne transmet au PN qu'une fraction (coefficient de couplage << 1) ; les synapses eLN -> LN inhibiteurs
#    restent chimiques et inchangées. ELN_ELECTRICAL_GAIN est le facteur appliqué au nombre de synapses
#    eLN -> PN/eLN ; sa valeur n'est pas mesurée dans la littérature (paramètre incertain, calibré).
ELN_ELECTRICAL_GAIN = 0.1


def antennal_lobe_populations(neurons: pd.DataFrame, sign: np.ndarray) -> dict[str, np.ndarray]:
    """Indices des ORN, PN, LN excitateurs (eLN) et inhibiteurs (iLN) du lobe antennaire d'après les
    annotations officielles (`cls`) et le signe déduit des neurotransmetteurs (ACh -> eLN)."""
    cls = neurons.cls.to_numpy()
    ln = cls == "antennal_lobe_local_neuron"
    return {
        "ORN": np.flatnonzero(cls == "olfactory_receptor_neuron"),
        "PN": np.flatnonzero(cls == "antennal_lobe_projection_neuron"),
        "eLN": np.flatnonzero(ln & (sign > 0)),
        "iLN": np.flatnonzero(ln & (sign < 0)),
    }


def depression_U(neurons: pd.DataFrame, orn_U: float = ORN_STD_U) -> np.ndarray:
    """Fraction de ressources libérée par spike, par neurone présynaptique : `orn_U` pour les ORN, 0 ailleurs."""
    U = np.zeros(len(neurons), dtype=np.float32)
    U[neurons.cls.to_numpy() == "olfactory_receptor_neuron"] = orn_U
    return U


def synaptic_efficacy(W: sp.csc_matrix, neurons: pd.DataFrame, sign: np.ndarray,
                      eln_gain: float = ELN_ELECTRICAL_GAIN) -> sp.csc_matrix:
    """Pondère les synapses eLN -> PN et eLN -> eLN (électriques) par `eln_gain` ; les autres sont
    inchangées. Retourne une nouvelle matrice (même support de connexions)."""
    pops = antennal_lobe_populations(neurons, sign)
    W = W.tocsc(copy=True)
    W.eliminate_zeros()
    src = pops["eLN"]
    dst = np.zeros(W.shape[0], dtype=bool)
    dst[pops["PN"]] = True
    dst[pops["eLN"]] = True
    for j in src:
        a, b = W.indptr[j], W.indptr[j + 1]
        rows = W.indices[a:b]
        W.data[a:b][dst[rows]] *= eln_gain
    return W


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
