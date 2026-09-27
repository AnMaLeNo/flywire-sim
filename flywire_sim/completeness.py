"""Correction de complétude synaptique du BANC.

Le BANC v888 n'attache que 18 % de ses liens synaptiques à un neurone identifié des deux côtés, contre
41,9 % pour FAFB et 44,4 % pour MANC (Bates et al. 2026, Methods « Synapse detection »). Le nombre de
synapses d'une connexion BANC est donc un sous-échantillon de la connexion réelle, et le sous-échantillonnage
est très inégal : entrées des motoneurones de patte ~1x MANC, entrées des interneurones de la moelle ~0,2x,
entrées des neurones centraux ~0,3x FAFB (voir docs/calibration.md § 6).

Le poids par synapse du modèle LIF (Shiu et al. 2024, 0,275 mV) a été ajusté sur les comptes FAFB. Pour
l'appliquer aux comptes BANC on rééchelonne les synapses *existantes* de chaque neurone postsynaptique par
l'inverse de son taux de récupération estimé, mesuré type par type sur les exports officiels FAFB (cerveau)
et MANC (moelle) : rapport des médianes de synapses d'entrée par neurone entre le BANC et le jeu de référence.
Aucune connexion n'est ajoutée ni retirée, les poids relatifs des partenaires d'un neurone sont conservés.
Hiérarchie d'estimation : type cellulaire apparié -> (super-classe, hémilignée) -> super-classe -> 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import banc, data

MANC_RAW = data.RAW / "manc121"
CACHE = data.PROCESSED / "banc_completeness.csv"
MIN_SYN = 5  # même seuil de connexion dans les trois jeux
MIN_REF_IN = 50  # synapses d'entrée médianes minimales du type de référence
MIN_N = 2  # neurones par type, dans chaque jeu
FACTOR_MIN, FACTOR_MAX = 1.0, 3.0
# super-classes effectivement corrigées dans la simulation : la moelle seulement (référence MANC, même
# organe ; la correction du cerveau sur référence FAFB rend le cerveau instable, cf. docs/calibration.md § 6)
APPLY = frozenset({"ventral_nerve_cord_intrinsic"})

# super-classe BANC -> (jeu de référence, super-classe de référence)
REFERENCE = {
    "central_brain_intrinsic": ("fafb", "central"),
    "optic_lobe_intrinsic": ("fafb", "optic"),
    "visual_projection": ("fafb", "visual_projection"),
    "visual_centrifugal": ("fafb", "visual_centrifugal"),
    "descending": ("fafb", "descending"),
    "ventral_nerve_cord_intrinsic": ("manc", "intrinsic_neuron"),
    "ascending": ("manc", "ascending"),
}
# Motoneurones : entrées BANC ~ MANC au total (MN <- IN 0,92, MN <- DN 1,02, MN <- sensoriel 0,99) mais
# très variables par type dans les deux sens (individu, sexe) : aucun déficit systématique, facteur 1.
# Neurones sensoriels : entrées non simulées (afférents clampés), facteur 1.


def _inputs(conn: pd.DataFrame, root_ids: np.ndarray) -> np.ndarray:
    c = conn[conn.syn_count >= MIN_SYN]
    s = c.groupby("post_root_id").syn_count.sum()
    return s.reindex(root_ids).fillna(0).to_numpy(dtype=np.float64)


def _fafb() -> pd.DataFrame:
    ann = data.load_annotations()
    ct = data.load_cell_types()[["root_id", "primary_type"]]
    n = ann[["root_id", "super_class", "ito_lee_hemilineage"]].rename(columns={"ito_lee_hemilineage": "hemilineage"})
    n = n.merge(ct, on="root_id", how="left")
    n = n.rename(columns={"primary_type": "cell_type"})
    n["n_in"] = _inputs(data.load_connections(), n.root_id.to_numpy())
    return n


def _manc() -> pd.DataFrame:
    n = pd.read_csv(MANC_RAW / "neurons.csv.gz", low_memory=False).rename(columns={
        "Root ID": "root_id", "Super Class": "super_class", "Primary Cell Type": "cell_type",
        "Hemilineage": "hemilineage"})[["root_id", "super_class", "cell_type", "hemilineage"]]
    n["n_in"] = _inputs(pd.read_csv(MANC_RAW / "connections_princeton.csv.gz"), n.root_id.to_numpy())
    return n


def _banc() -> pd.DataFrame:
    n = banc.load_neurons()[["root_id", "super_class", "cell_type", "hemilineage"]].copy()
    n["n_in"] = _inputs(banc.load_connections(), n.root_id.to_numpy())
    return n


def _medians(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    d = df[df.n_in > 0].dropna(subset=keys)
    for k in keys:
        d = d[d[k] != ""]
    return d.groupby(keys).n_in.agg(n="size", med="median")


def compute() -> pd.DataFrame:
    """Facteur de complétude par neurone BANC (colonnes : root_id, ratio, factor, level).

    ratio = entrées BANC du neurone / médiane de référence de son type (repli : hémilignée, puis
    super-classe) ; factor = 1 / ratio borné à [FACTOR_MIN, FACTOR_MAX] (jamais < 1 : la correction ne
    fait que compenser des synapses non récupérées, elle n'en retire pas).
    """
    b = _banc()
    for c in ("super_class", "cell_type", "hemilineage"):
        b[c] = b[c].fillna("").astype(str)
    refs = {"fafb": _fafb(), "manc": _manc()}
    for r in refs.values():
        for c in ("super_class", "cell_type", "hemilineage"):
            r[c] = r[c].fillna("").astype(str)
    target = np.full(len(b), np.nan)
    level = np.array([""] * len(b), dtype=object)
    for sc, (ds, ref_sc) in REFERENCE.items():
        m = (b.super_class == sc).to_numpy()
        if not m.any():
            continue
        ref = refs[ds][refs[ds].super_class == ref_sc]
        sub = b[m]
        for keys, name in ((["cell_type"], "type"), (["hemilineage"], "hemilineage")):
            mr = _medians(ref, keys)
            mr = mr[(mr.n >= MIN_N) & (mr.med >= MIN_REF_IN)]
            k = sub[keys[0]].map(mr.med).to_numpy(dtype=np.float64)
            fill = np.isnan(target[m]) & ~np.isnan(k)
            idx = np.flatnonzero(m)[fill]
            target[idx] = k[fill]
            level[idx] = name
        rest = m & np.isnan(target)
        if rest.any():
            target[rest] = np.median(ref.n_in[ref.n_in > 0])
            level[rest] = "super_class"
    n_in = b.n_in.to_numpy(dtype=np.float64)
    ratio = np.where(np.isnan(target) | (n_in <= 0), np.nan, n_in / target)
    out = pd.DataFrame(
        {"root_id": b.root_id.to_numpy(), "super_class": b.super_class.to_numpy(), "ratio": ratio, "level": level}
    )
    out["factor"] = np.where(np.isnan(ratio), 1.0, np.clip(1.0 / ratio, FACTOR_MIN, FACTOR_MAX))
    out.loc[np.isnan(ratio), "level"] = "none"
    return out


def load(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE)
    out = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    return out


def factors(root_ids: np.ndarray) -> np.ndarray:
    """Facteur multiplicatif des entrées synaptiques (1 hors des super-classes de APPLY)."""
    t = load().set_index("root_id").reindex(root_ids)
    f = t.factor.fillna(1.0).to_numpy(dtype=np.float32)
    return np.where(t.super_class.isin(APPLY).to_numpy(), f, np.float32(1.0))


def apply(W: sp.csc_matrix, root_ids: np.ndarray) -> sp.csc_matrix:
    """Rééchelonne les entrées (lignes) de chaque neurone postsynaptique par son facteur de complétude."""
    return (sp.diags(factors(root_ids)) @ W).tocsc()
