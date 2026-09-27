"""Cerveau : export officiel Codex du FAFB v783 (FlyWire, Dorkenwald et al. 2024 ; annotations Schlegel et al.
2024), femelle adulte, cerveau complet sans moelle. Sert de cerveau au réseau hybride (hybrid.py) à la place
du BANC : l'export BANC ne récupère que 0,68x des synapses excitatrices et 0,58x des inhibitrices reçues par
les DN du FAFB (docs/calibration.md § 7.2), ce qui entretient une boucle moelle -> AN -> cerveau -> DN.

Les colonnes sont harmonisées sur celles de banc.load_neurons(). Le FAFB n'annote les afférents que par
classe (`cell_class` : olfactory, gustatory, mechanosensory...) et type (ORN_DA1, JO-B, BM_InOm, R7...) ;
l'organe, la fonction et la partie du corps utilisés par body/senses.py sont **transférés depuis le BANC par
type cellulaire** (nomenclature commune ; annotation majoritaire du type, pureté >= 0,9 sauf un type), et
complétés par une petite table explicite pour les sous-classes FAFB sans type partagé (`FAFB_SUB_CLASS`).

Pont vers la moelle MANC : les DN du FAFB portent en partie la nomenclature commune (DNg100, DNa02...), mais
les AN (`AN_GNG_165`...) et les afférents ascendants (`SA_...`) ont des noms propres au FAFB, sans équivalent
direct dans le MANC (`AN07B072`...). Le BANC, qui porte les deux nomenclatures (cell_type MANC-compatible ;
`Alternative Cell Type(s)` et `Community labels` contenant les noms FAFB), sert de pierre de Rosette
(`bridge_types`) : type FAFB -> type MANC = type BANC majoritaire des neurones BANC portant ce nom FAFB,
retenu seulement si la majorité est stricte (part > BRIDGE_PURITY). Aucune correspondance n'est inventée hors
de ces annotations officielles ; les neurones sans type appariable restent des neurones du cerveau sans moelle.
"""
import re

import numpy as np
import pandas as pd

from . import banc, data, manc, network
from .network import Network

BRIDGE_PURITY = 0.5
POSITIONS_FILE = data.PROCESSED / "fafb783_positions.csv.gz"

SUPER_CLASS = {"optic": "optic_lobe_intrinsic", "central": "central_brain_intrinsic"}
CLASS = {"ALLN": "antennal_lobe_local_neuron", "ALPN": "antennal_lobe_projection_neuron", "ALIN": "antennal_lobe_input_neuron",
         "ALON": "antennal_lobe_output_neuron", "Kenyon_Cell": "kenyon_cell", "MBON": "mushroom_body_output_neuron",
         "DAN": "dopaminergic_neuron", "MBIN": "mushroom_body_input_neuron", "CX": "central_complex_neuron",
         "LHLN": "lateral_horn_local_neuron", "LHCENT": "lateral_horn_centrifugal_neuron", "TuBu": "tubercle_bulb_neuron",
         "olfactory": "olfactory_receptor_neuron", "gustatory": "gustatory_receptor_neuron",
         "mechanosensory": "mechanosensory_neuron", "thermosensory": "thermosensory_receptor_neuron",
         "hygrosensory": "hygrosensory_receptor_neuron", "visual": "photoreceptor_neuron"}
# sous-classes FAFB sans type partagé avec le BANC -> (cls, sub_class, body_part, function) au schéma BANC.
# `sugar/water` : le FAFB ne sépare pas les GRN sucre (Gr64f) et eau (ppk28) du labelle ; la fonction garde
# les deux noms et body/senses.py les met dans les deux canaux.
FAFB_SUB_CLASS = {
    ("gustatory", "sugar/water"): ("taste_bristle_gustatory_neuron", "labellum_taste_bristle_gustatory_neuron", "labellum",
                                   "gustatory,sugar/water"),
    ("gustatory", "low-salt"): ("taste_bristle_gustatory_neuron", "labellum_taste_bristle_gustatory_neuron", "labellum",
                                "gustatory,low_salt"),
    ("gustatory", "bitter"): ("taste_bristle_gustatory_neuron", "labellum_taste_bristle_gustatory_neuron", "labellum",
                              "bitter,gustatory"),
    ("thermosensory", "cold"): ("thermosensory_receptor_neuron", "antenna_thermosensory_receptor_neuron", "antenna",
                                "cooling,thermosensory"),
    ("thermosensory", "heating"): ("thermosensory_receptor_neuron", "antenna_thermosensory_receptor_neuron", "antenna",
                                   "heating,thermosensory"),
    ("visual", "ocellar"): ("photoreceptor_neuron", "ocellus_photoreceptor_neuron", "ocellus", "visual"),
    ("visual", None): ("photoreceptor_neuron", "retina_photoreceptor_neuron", "retina", "visual_achromatic"),
    ("mechanosensory", "auditory"): ("chordotonal_organ_neuron", "johnstons_organ_other_neuron", "antenna", "auditory"),
    ("mechanosensory", "wind_gravity"): ("chordotonal_organ_neuron", "johnstons_organ_other_neuron", "antenna", "position"),
    ("mechanosensory", "grooming"): ("chordotonal_organ_neuron", "johnstons_organ_other_neuron", "antenna", "position"),
    ("mechanosensory", "eye bristle"): ("bristle_neuron", "interommatidial_bristle_neuron", "interommatidial", "tactile"),
    ("mechanosensory", "head bristle"): ("bristle_neuron", "head_bristle_neuron", "head", "tactile"),
}
COLUMNS = ("root_id", "super_class", "cls", "sub_class", "function", "body_part", "side", "cell_type", "region",
           "hemilineage", "nt_pred", "nt_verified", "flow", "nerve", "labels", "bridge_type")


def _majority(x: pd.Series) -> str | float:
    x = x.dropna()
    return x.value_counts().index[0] if len(x) else np.nan


def transfer_sensory(f: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """cls / sub_class / body_part / function des afférents FAFB = annotation BANC majoritaire de leur type ;
    à défaut, table explicite FAFB_SUB_CLASS ; à défaut, classe FAFB seule."""
    bs = b[b.super_class.fillna("").str.startswith("sensory") & b.cell_type.notna()]
    ref = bs.groupby("cell_type").agg(cls=("cls", _majority), sub_class=("sub_class", _majority),
                                      body_part=("body_part", _majority), function=("function", _majority))
    f = f.copy()
    is_sens = f.super_class.fillna("").str.startswith("sensory").to_numpy()
    t = f.loc[is_sens, "cell_type"]
    got = t.isin(ref.index)
    for c in ("cls", "sub_class", "body_part", "function"):
        f.loc[is_sens, c] = t.map(ref[c])
    keys = zip(f.loc[is_sens, "cell_class"], f.loc[is_sens, "cell_sub_class"])
    table = [FAFB_SUB_CLASS.get((cc, sc if isinstance(sc, str) else None)) for cc, sc in keys]
    for i, c in enumerate(("cls", "sub_class", "body_part", "function")):
        vals = pd.Series([v[i] if v else np.nan for v in table], index=t.index)
        f.loc[is_sens, c] = f.loc[is_sens, c].where(got, vals)
    f.loc[is_sens, "cls"] = f.loc[is_sens, "cls"].fillna(f.loc[is_sens, "cell_class"].map(CLASS))
    return f


def bridge_types(b: pd.DataFrame, fafb_types: set[str], manc_types: set[str]) -> dict[str, str]:
    """Type FAFB -> type MANC via les neurones pontés du BANC portant les deux noms (voir doc du module)."""
    bb = b[b.super_class.isin(("descending", "ascending", "sensory_ascending", "ascending_visceral_circulatory"))
           & b.cell_type.notna() & b.cell_type.isin(manc_types)]
    names = []
    for ct, alt, lab in zip(bb.cell_type, bb["Alternative Cell Type(s)"], bb.labels):
        found = ct if ct in fafb_types else np.nan
        for v in (alt, lab):
            if not isinstance(found, str) and isinstance(v, str):
                found = next((tok for tok in re.split(r"[,;]\s*", v) if tok in fafb_types), np.nan)
        names.append(found)
    pairs = pd.DataFrame({"fafb": names, "manc": bb.cell_type.to_numpy()}).dropna()
    cnt = pairs.groupby(["fafb", "manc"]).size().reset_index(name="n")
    tot = cnt.groupby("fafb").n.transform("sum")
    top = cnt[cnt.n / tot > BRIDGE_PURITY].sort_values("n", ascending=False).drop_duplicates("fafb")
    out = {t: t for t in fafb_types & manc_types}
    out.update({k: v for k, v in zip(top.fafb, top.manc) if k not in out})
    return out


def load_neurons() -> pd.DataFrame:
    n = data.neuron_table().rename(columns={"primary_type": "cell_type", "class": "cls_fafb", "sub_class": "sub_class_fafb",
                                            "nt_type": "nt_pred", "group": "region", "name": "labels"})
    ann = data.load_annotations()[["root_id", "cell_class", "cell_sub_class", "ito_lee_hemilineage"]]
    n = n.merge(ann, on="root_id", how="left").rename(columns={"ito_lee_hemilineage": "hemilineage"})
    n["root_id"] = n.root_id.astype(np.int64)
    n["super_class"] = n.super_class.map(lambda s: SUPER_CLASS.get(s, s))
    n["cls"] = n.cell_class.map(lambda s: CLASS.get(s, s))
    for c in ("sub_class", "body_part", "function", "nt_verified", "region"):
        if c not in n:
            n[c] = pd.Series(np.nan, index=n.index, dtype=object)
    b = banc.load_neurons()
    n = transfer_sensory(n, b)
    types = set(n.cell_type.dropna())
    tmap = bridge_types(b, types, set(manc.load_neurons().cell_type.dropna()))
    n["bridge_type"] = n.cell_type.map(tmap)
    return n[[c for c in COLUMNS if c in n.columns]]


def load_positions() -> pd.DataFrame:
    """Position de référence de chaque neurone (export Codex `coordinates`, en nm), en µm et dans les axes du
    volume BANC utilisés par body/vision.py (x droite -> gauche de la mouche, y antérieur -> postérieur,
    z ventral -> dorsal). Dans le volume FAFB, X croît vers la droite de la mouche, Y vers le ventre et Z vers
    l'arrière (vérifié sur les centroïdes annotés : rétine gauche X~220 / droite X~820 µm, ocelles Y~90 /
    labellum Y~350, ORN Z~30 / calyx Z~170) : (x, y, z)_BANC = (-X, Z, -Y)_FAFB."""
    if POSITIONS_FILE.exists():
        return pd.read_csv(POSITIONS_FILE)
    c = pd.read_csv(data.CODEX / "coordinates.csv.gz").drop_duplicates("root_id")
    xyz = np.array([[float(v) for v in re.findall(r"-?\d+", s)[:3]] for s in c.position]) / 1000.0
    pos = pd.DataFrame({"root_id": c.root_id.astype(np.int64), "x": -xyz[:, 0], "y": xyz[:, 2], "z": -xyz[:, 1]})
    data.PROCESSED.mkdir(parents=True, exist_ok=True)
    pos.to_csv(POSITIONS_FILE, index=False)
    return pos


def build(min_synapses: int = 5) -> Network:
    """Réseau FAFB signé (network.build) ; comme dans manc.build, les motoneurones (110 MN de la tête : trompe,
    cou, antennes) sont excitateurs quel que soit leur NT prédit (glutamate excitateur à la jonction
    neuromusculaire des insectes)."""
    net = network.build(min_synapses)
    n = data.neuron_table().set_index("root_id").reindex(net.root_ids)
    mot = np.flatnonzero((n.super_class == "motor").to_numpy())
    flip = mot[net.sign[mot] < 0]
    net.sign[mot] = 1.0
    W = net.W.tocsc(copy=True)
    for j in flip:
        W.data[W.indptr[j]:W.indptr[j + 1]] = np.abs(W.data[W.indptr[j]:W.indptr[j + 1]])
    return Network(root_ids=net.root_ids, W=W, sign=net.sign, nt_type=net.nt_type)
