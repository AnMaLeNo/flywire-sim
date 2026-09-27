"""Localisation : la moelle MANC seule produit-elle un rythme de marche sous DNg100 ?

Pugliese et al. 2025 (bioRxiv 2025.09.12.675944) : dans un modèle de taux du MANC avec seuil ∝ taille et gain
∝ 1/taille (volume, médiane-normalisé), DNg100 à lui seul produit des oscillations des MN de patte à 7–15 Hz
via un circuit E1 (IN17A001) -> E2 (INXXX466) -> I1 (IN16B036) -| E1, E2. On mesure ici, avec notre LIF
(CALIBRATED_V2 = paramètres de Shiu 2024, PSP homogène) et avec une normalisation par le volume officiel
(`Volume (nm^3)` de l'export MANC) d'exposant gamma, l'activité et la rythmicité des MN de patte et de E1/E2/I1.
Aucune connexion n'est modifiée : seul le PSP reçu par chaque neurone est divisé par (V / V_REF)^gamma
(flywire_sim.size). Avec --stim-stop, l'activité après l'arrêt de DNg100 mesure l'auto-entretien du réseau.
"""
import argparse
import sys
from dataclasses import replace

import numpy as np

from flywire_sim import banc, manc
from flywire_sim.lif import LIFNetwork, Stimulus
from flywire_sim.size import V_REF_NM3, relative_size

CORE = ["IN17A001", "INXXX466", "IN16B036", "IN19A007", "IN19B012", "IN03A006"]
LEGS = ["front_leg", "middle_leg", "hind_leg"]


def rhythmicity(train: np.ndarray, bin_ms: float, lag_min: float = 40, lag_max: float = 250) -> tuple[float, float]:
    """Autocorrélation d'une série de taux (binée) ; retourne (score, période_ms) : score = hauteur du premier
    pic d'autocorrélation dans [lag_min, lag_max] ms (0 si aucun pic), à la façon du score de Pugliese 2025."""
    x = train - train.mean()
    if x.std() == 0:
        return 0.0, np.nan
    ac = np.correlate(x, x, "full")[x.size - 1:]
    ac = ac / ac[0]
    lo, hi = int(lag_min / bin_ms), int(lag_max / bin_ms)
    seg = ac[lo:hi]
    k = np.argmax(seg)
    if 0 < k < seg.size - 1 and seg[k] > seg[k - 1] and seg[k] > seg[k + 1]:
        return float(seg[k]), float((lo + k) * bin_ms)
    return 0.0, np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=2000)
    ap.add_argument("--gammas", type=float, nargs="*", default=[0.0, 1.0, 2.0])
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--bin", type=float, default=5.0)
    ap.add_argument("--w", type=float, default=banc.CALIBRATED_V2.w_syn, help="PSP (mV/synapse) du neurone médian")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapt", type=float, default=0.0, help="adaptation de fréquence (mV/spike, tau 100 ms)")
    ap.add_argument("--refr", type=float, default=banc.CALIBRATED_V2.t_refractory, help="période réfractaire (ms)")
    ap.add_argument("--std", type=float, default=0.0, help="dépression synaptique globale U (Tsodyks-Markram)")
    ap.add_argument("--stim-stop", type=float, default=np.inf, help="fin de la stimulation DNg100 (ms)")
    a = ap.parse_args()

    n, net = manc.load_neurons(), manc.build()
    n = n.set_index("root_id").loc[net.root_ids].reset_index()
    size = relative_size(n)
    print(f"[w={a.w}, adapt={a.adapt}, refr={a.refr}, std={a.std}] V_REF {V_REF_NM3:.2e} nm^3 ; taille rel. MN patte médiane "
          f"{np.median(size[(n.super_class == 'motor') & n.body_part.isin(LEGS)]):.2f}, IN VNC "
          f"{np.median(size[n.super_class == 'ventral_nerve_cord_intrinsic']):.2f}, "
          f"E1 {size[n.cell_type == 'IN17A001'].mean():.2f} E2 {size[n.cell_type == 'INXXX466'].mean():.2f} "
          f"I1 {size[n.cell_type == 'IN16B036'].mean():.2f}", flush=True)
    stim = np.flatnonzero(n.cell_type.eq("DNg100").to_numpy())
    mn = (n.super_class.eq("motor") & n.body_part.isin(LEGS)).to_numpy()
    nb = int(a.ms / a.bin)

    for g in a.gammas:
        res = LIFNetwork(net.W, replace(banc.CALIBRATED_V2, w_syn=a.w, size_norm=g, adapt_b=a.adapt, t_refractory=a.refr,
                                        std_U=a.std), seed=a.seed, size=size).run(a.ms, [Stimulus(stim, a.rate, t_stop=a.stim_stop)])
        rates = res.rates_hz()
        sc = n.super_class.to_numpy()
        inn = rates[sc == 'ventral_nerve_cord_intrinsic']
        print(f"\n[gamma={g}] total {rates.sum():.0f} spk/s | IN >100 Hz {(inn > 100).sum()} ({inn[inn > 100].sum() / max(inn.sum(), 1):.0%} des spikes IN) | IN VNC {rates[sc == 'ventral_nerve_cord_intrinsic'].mean():.1f} Hz "
              f"({(rates[sc == 'ventral_nerve_cord_intrinsic'] > 0).mean():.0%}) | MN patte {rates[mn].mean():.1f} Hz "
              f"({(rates[mn] > 0).mean():.0%} actifs) | MN aile {rates[(sc == 'motor') & (n.body_part == 'wing')].mean():.1f} Hz")
        if np.isfinite(a.stim_stop):
            post = res.spike_times >= a.stim_stop + 100
            print(f"   après l'arrêt (+100 ms) : {post.sum() / (a.ms - a.stim_stop - 100) * 1000:.0f} spk/s "
                  f"(pendant : {(res.spike_times < a.stim_stop).sum() / a.stim_stop * 1000:.0f} spk/s)")
        for t in CORE:
            m = n.cell_type.eq(t).to_numpy()
            print(f"   {t}: {np.round(rates[m], 1).tolist()} Hz ({n.nt_pred[m].iloc[0]})")
        # rythmicité : population de MN par patte et par côté, puis MN individuels actifs
        t_bin = np.minimum((res.spike_times / a.bin).astype(int), nb - 1)
        for side in ["left", "right"]:
            for leg in LEGS:
                m = mn & (n.side == side).to_numpy() & (n.body_part == leg).to_numpy()
                sel = m[res.spike_neurons]
                train = np.bincount(t_bin[sel], minlength=nb)[int(300 / a.bin):]   # après 300 ms de transitoire
                s, per = rhythmicity(train.astype(float), a.bin)
                print(f"   {side[0].upper()}{leg[0].upper()} MN pop: {train.sum() / (a.ms - 300) * 1000:.0f} spk/s, "
                      f"rythmicité {s:.2f}, période {per:.0f} ms")
        scores = []
        for i in np.flatnonzero(mn & (rates > 2)):
            train = np.bincount(t_bin[res.spike_neurons == i], minlength=nb)[int(300 / a.bin):]
            scores.append(rhythmicity(train.astype(float), a.bin))
        if scores:
            s = np.array([x[0] for x in scores])
            p = np.array([x[1] for x in scores])
            print(f"   MN individuels (>2 Hz, n={len(s)}): rythmicité médiane {np.median(s):.2f}, >0.3 : {(s > 0.3).mean():.0%}, "
                  f"période médiane {np.nanmedian(p):.0f} ms", flush=True)
        for t in CORE[:3]:
            for i in np.flatnonzero(n.cell_type.eq(t).to_numpy() & (rates > 2)):
                train = np.bincount(t_bin[res.spike_neurons == i], minlength=nb)[int(300 / a.bin):]
                s, per = rhythmicity(train.astype(float), a.bin)
                print(f"   {t} ({n.side[i][0]}) rythmicité {s:.2f} période {per:.0f} ms")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
