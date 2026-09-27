"""Taille officielle des neurones (volume du maillage, nm^3) et taille relative pour le principe de taille.

Kazama & Wilson 2008 (Neuron, ORN -> PN) : le courant synaptique unitaire croît avec le volume de l'arbre
dendritique du PN alors que le potentiel unitaire (uEPSP ~ 6,2 mV) est constant d'un glomérule à l'autre ;
dans le FAFB v783, le nombre de synapses par paire ORN -> PN est lui aussi proportionnel au volume du PN
(DM4 57 syn / 3,1e12 nm^3 ... VM2 12 / 6,8e11, VA1v 6 / 7,5e11). Le PSP par synapse est donc ~ 1/volume :
c'est ce que le LIF applique via `size_norm` avec la taille relative renvoyée ici (Pugliese et al. 2025
utilisent la même normalisation par le volume MANC).

Volumes : FAFB `cell_stats.csv.gz` (size_nm), MANC `Volume (nm^3)` ; un neurone fusionné FAFB+MANC
(descendant, ascendant) totalise ses deux moitiés. La référence est un volume absolu, commun aux deux jeux.

Neurones tronqués : le maillage d'un afférent, d'un ascendant non fusionné (FAFB) ou d'un descendant non fusionné
(MANC) ne couvre que sa partie contenue dans le volume imagé ; son vrai volume est plus grand, d'un facteur
inconnu. Leur taille relative est bornée à >= 1 : jamais amplifiés au-delà du poids de référence."""
import numpy as np
import pandas as pd

from . import data, manc

V_REF_NM3 = 2.5e11    # ~ volume médian d'un neurone du MANC (2,46e11) ; ~88e percentile du FAFB (7,8e10 médian)


def fafb_volumes() -> pd.Series:
    st = pd.read_csv(data.CODEX / "cell_stats.csv.gz", usecols=["root_id", "size_nm"],
                     dtype={"root_id": np.int64, "size_nm": np.float64})
    return st.set_index("root_id").size_nm


def manc_volumes() -> pd.Series:
    m = manc.load_neurons()
    return pd.Series(m["Volume (nm^3)"].astype(np.float64).to_numpy(), index=m.root_id.to_numpy())


def volume_nm3(neurons: pd.DataFrame) -> np.ndarray:
    """Volume officiel par neurone (NaN si absent), table alignée sur les indices du réseau.

    Tables acceptées : FAFB seul, MANC seul (colonne `Volume (nm^3)`), hybride (`source` = fafb/manc/fused
    avec `manc_id`)."""
    n = neurons.reset_index(drop=True)
    if "Volume (nm^3)" in n.columns:
        return n["Volume (nm^3)"].astype(np.float64).to_numpy()
    if "source" not in n.columns:
        return fafb_volumes().reindex(n.root_id.to_numpy(dtype=np.int64)).to_numpy()
    src = n.source.to_numpy(dtype=str)
    vol = np.full(len(n), np.nan)
    brain = np.isin(src, ["fafb", "fused"])
    vol[brain] = fafb_volumes().reindex(n.root_id.to_numpy(dtype=np.int64)[brain]).to_numpy()
    cord = np.isin(src, ["manc", "fused"])
    mv = manc_volumes().reindex(n.manc_id.to_numpy(dtype=np.int64)[cord]).to_numpy()
    vol[cord] = np.where(np.isnan(vol[cord]), 0.0, vol[cord]) + mv
    return vol


TRUNCATED_FAFB = ("sensory", "sensory_ascending", "ascending")
TRUNCATED_MANC = ("sensory", "descending")


def truncated(neurons: pd.DataFrame) -> np.ndarray:
    """Neurones dont le corps cellulaire / une partie de l'arbre est hors du volume imagé (volume partiel)."""
    n = neurons.reset_index(drop=True)
    sc = n.super_class.fillna("").to_numpy(dtype=str)
    if "Volume (nm^3)" in n.columns:
        return np.isin(sc, TRUNCATED_MANC)
    if "source" not in n.columns:
        return np.isin(sc, TRUNCATED_FAFB)
    src = n.source.to_numpy(dtype=str)
    return ((src == "fafb") & np.isin(sc, TRUNCATED_FAFB)) | ((src == "manc") & np.isin(sc, TRUNCATED_MANC))


def relative_size(neurons: pd.DataFrame, ref_nm3: float = V_REF_NM3) -> np.ndarray:
    """volume / ref ; sans volume officiel -> 1 (poids inchangé) ; tronqué -> max(volume / ref, 1)."""
    v = volume_nm3(neurons) / ref_nm3
    v = np.where(np.isfinite(v) & (v > 0), v, 1.0)
    return np.where(truncated(neurons), np.maximum(v, 1.0), v).astype(np.float32)
