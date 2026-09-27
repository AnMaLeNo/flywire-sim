from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CODEX = RAW / "codex783"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

MN9 = {"left": 720575940618238523, "right": 720575940660219265}


def load_neurons() -> pd.DataFrame:
    return pd.read_csv(CODEX / "neurons.csv.gz")


def load_connections() -> pd.DataFrame:
    return pd.read_csv(
        CODEX / "connections.csv.gz",
        dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "syn_count": np.int32},
    )


def load_classification() -> pd.DataFrame:
    return pd.read_csv(CODEX / "classification.csv.gz")


def load_labels() -> pd.DataFrame:
    return pd.read_csv(CODEX / "labels.csv.gz")


def load_names() -> pd.DataFrame:
    return pd.read_csv(CODEX / "names.csv.gz")


def load_cell_types() -> pd.DataFrame:
    return pd.read_csv(CODEX / "consolidated_cell_types.csv.gz")


def load_annotations() -> pd.DataFrame:
    """Annotations systématiques Schlegel et al. 2024 (flyconnectome/flywire_annotations)."""
    return pd.read_csv(RAW / "Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False)


def neuron_table() -> pd.DataFrame:
    """Table par neurone : NT prédit, classification hiérarchique, nom et type cellulaire."""
    n = load_neurons()[["root_id", "nt_type", "nt_type_score"]]
    c = load_classification()[["root_id", "flow", "super_class", "class", "sub_class", "side", "nerve"]]
    nm = load_names()[["root_id", "name"]]
    ct = load_cell_types()[["root_id", "primary_type"]]
    return n.merge(c, on="root_id", how="left").merge(nm, on="root_id", how="left").merge(ct, on="root_id", how="left")


def root_ids_with_label(pattern: str, side: str | None = None) -> np.ndarray:
    lab = load_labels()
    ids = lab.loc[lab.label.str.contains(pattern, case=False, regex=True), "root_id"].unique()
    if side is not None:
        cls = load_classification().set_index("root_id")
        ids = np.array([i for i in ids if cls.side.get(i) == side], dtype=np.int64)
    return np.asarray(ids, dtype=np.int64)
