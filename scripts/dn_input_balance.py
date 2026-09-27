"""Entrées excitatrices / inhibitrices des DN dans le cerveau BANC vs FAFB v783 (même LIF, mêmes signes).

Question : la boucle AN -> cerveau -> DN du réseau hybride tient-elle à un biais E/I des entrées des DN
dans l'export BANC ? Pour chaque type de DN présent dans les deux jeux : synapses entrantes E et I par
neurone (min 5 synapses), et ratio BANC / FAFB."""
import argparse

import numpy as np
import pandas as pd

from flywire_sim import banc, data, network


def ei(W, idx) -> tuple[np.ndarray, np.ndarray]:
    Wr = W.tocsr()[idx]
    return np.asarray(Wr.maximum(0).sum(axis=1)).ravel(), -np.asarray(Wr.minimum(0).sum(axis=1)).ravel()


def table(W, n: pd.DataFrame, brain_cols: np.ndarray) -> pd.DataFrame:
    dn = n.super_class.eq("descending") & n.cell_type.notna()
    idx = np.flatnonzero(dn.to_numpy())
    W = W.tocsc()[:, brain_cols] if brain_cols is not None else W
    e, i = ei(W, idx)
    return pd.DataFrame({"type": n.cell_type.to_numpy()[idx], "E": e, "I": i}).groupby("type").mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=["DNg100", "DNa03", "DNa06", "DNg33", "DNb02", "DNg75", "DNa02",
                                                   "DNp31", "DNut040", "DNp26", "DNb06", "DNg49"])
    a = ap.parse_args()

    bn = banc.build()
    b = banc.load_neurons().set_index("root_id").reindex(bn.root_ids).reset_index()
    brain = ~(b.super_class.fillna("").isin(["ventral_nerve_cord_intrinsic", "motor"])
              | b.super_class.fillna("").str.startswith("sensory") & ~b.region.fillna("").str.startswith("brain"))
    tb = table(bn.W, b, np.flatnonzero(brain.to_numpy()))

    fn = network.build()
    f = data.neuron_table().rename(columns={"primary_type": "cell_type"}).set_index("root_id").reindex(fn.root_ids).reset_index()
    tf = table(fn.W, f, None)

    t = tb.join(tf, lsuffix="_banc", rsuffix="_fafb", how="inner")
    t["E_ratio"] = t.E_banc / t.E_fafb
    t["I_ratio"] = t.I_banc / t.I_fafb
    t["EI_banc"] = t.E_banc / t.I_banc.clip(lower=1)
    t["EI_fafb"] = t.E_fafb / t.I_fafb.clip(lower=1)
    pd.set_option("display.width", 200)
    print(f"types de DN communs : {len(t)} ; médiane E_ratio {t.E_ratio.median():.2f}, I_ratio {t.I_ratio.median():.2f} ; "
          f"E/I médian BANC {t.EI_banc.median():.2f} vs FAFB {t.EI_fafb.median():.2f}")
    print(t.loc[[x for x in a.types if x in t.index]].round(2))
    out = data.RESULTS / "dn_input_balance.csv"
    t.round(3).to_csv(out)
    print("->", out)


if __name__ == "__main__":
    main()
