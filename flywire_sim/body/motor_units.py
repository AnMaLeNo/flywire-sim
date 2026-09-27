"""Unités motrices des pattes : du volume officiel du motoneurone à sa classe, sa force et sa cinétique.

Mesures primaires (D. melanogaster, patte avant, fléchisseurs du tibia ; Azevedo et al. 2020, eLife 9:e56754) :
  - trois classes le long d'un gradient de taille : rapide (soma 13-21 µm, R_in ~150 MΩ, V_rest ~-68 mV,
    muet au repos), intermédiaire (8-12 µm, ~300 MΩ, ~-60 mV, muet), lent (5-10 µm, ~700 MΩ, ~-48 mV,
    ~30 Hz au repos) ; recrutement lent -> intermédiaire -> rapide (principe de taille) ;
  - force par spike : ~10 µN (rapide), ~1 µN (intermédiaire), 0,013 µN/spike (lent, pente Fig. 4D), mesurée
    au bout du tibia ; le pool fléchisseur entier produit ~100 µN (Fig. 1) ;
  - sommation : 2 spikes -> 1,6 x un spike ; plateau vers ~10 spikes à ~3 x un spike (rapide/intermédiaire) ;
    le MN lent somme linéairement sur >= 50 spikes ;
  - cinétique d'une secousse rapide/intermédiaire : pic ~20 ms après le spike, retour à la base en ~60-70 ms
    (Fig. 4A-B) ; le MN lent intègre sur >= 500 ms et relaxe en 200-300 ms (Fig. 4C).

Le MANC ne porte aucune annotation lent/intermédiaire/rapide : on utilise le **volume officiel** du MN
(`Volume (nm^3)`), qui ordonne les MN d'un pool comme la taille du soma d'Azevedo (R_in ∝ V^-0,6 sur les trois
classes) et comme la surface de Lesser et al. 2024 (poids prémoteurs ∝ taille du MN). Le fléchisseur du tibia
de la patte avant du MANC (5 MN principaux + 10 accessoires) sert de règle : son plus gros MN (1,39e12 nm³) est le
rapide, les MN à 0,5-0,9e12 les intermédiaires, les plus petits accessoires (~1,4e11) les lents.

Lois (ajustées sur ces trois points, puis appliquées à toutes les unités de patte — extrapolation documentée
dans docs/calibration.md § 10) :
  couple par spike   T1(V) = T_FAST x min(V / V_FAST, 1) ** K          (K = 3 ; ajustement 2,8, plage 2,5-3,3)
  saturation         T(a)  = T1 x (1 - exp(-a / A)) / (1 - exp(-1 / A)) : un spike isolé donne T1, deux spikes
                     1,67 x T1, plateau (tétanos) T1 / (1 - e^-1/A) = 3 x T1 ; A : 2,5 (rapide) -> 50 (lent, ~linéaire)
  secousse           noyau (e^-t/τd - e^-t/τr) normé, τr/τd : 12/15 ms (rapide : pic à 13 ms, < 10 % à 70 ms)
                     -> 20/300 ms (lent)
  potentiel de repos V_rest(V) = -8,2 mV x ln(V / 1e11 nm³) - 44,6 mV, borné à [-68, -48]
  résistance d'entrée R_in(V) ∝ V^-0,67 (700 MΩ lent -> 150 MΩ rapide pour un rapport de volume 10), appliquée
                     comme gain des entrées synaptiques du MN, = 1 au volume médian du MANC (size.V_REF_NM3)
Vérification : la somme des couples tétaniques du pool fléchisseur (principal + accessoire) de la patte avant
gauche vaut 48 µN·mm ≈ 100 µN x 0,52 mm (longueur du tibia du modèle), la force maximale mesurée par Azevedo.
"""
import numpy as np

V_FAST_NM3 = 1.39e12     # plus gros fléchisseur du tibia de la patte avant (MANC, moyenne G/D)
V_SLOW_NM3 = 1.4e11      # plus petits fléchisseurs accessoires distaux (MANC), classe lente d'Azevedo
TIBIA_MM = 0.52          # longueur du tibia du modèle (bras de levier de la mesure d'Azevedo)
T_FAST = 10.0 * TIBIA_MM  # µN·mm par spike du MN rapide (10 µN au bout du tibia)
K = 3.0
A_FAST, A_SLOW = 2.5, 50.0
TAU_FAST_MS, TAU_SLOW_MS = (12.0, 15.0), (20.0, 300.0)   # (montée, relaxation)
V_REST_FAST, V_REST_SLOW = -68.0, -48.0                   # mV
R_IN_EXP = np.log(700.0 / 150.0) / np.log(V_FAST_NM3 / V_SLOW_NM3)   # R_in ∝ V^-0,67
CLASS_BOUNDS = (0.2, 0.7)   # fraction de V_FAST : < 0,2 lent, < 0,7 intermédiaire, sinon rapide (étiquette)


def _frac(volume_nm3: np.ndarray) -> np.ndarray:
    """Position log-linéaire du MN entre la classe lente (0) et la classe rapide (1)."""
    v = np.clip(np.asarray(volume_nm3, dtype=np.float64), V_SLOW_NM3, V_FAST_NM3)
    return (np.log(v) - np.log(V_SLOW_NM3)) / (np.log(V_FAST_NM3) - np.log(V_SLOW_NM3))


def torque_per_spike(volume_nm3: np.ndarray) -> np.ndarray:
    """Couple (µN·mm) d'une secousse isolée."""
    v = np.asarray(volume_nm3, dtype=np.float64)
    return T_FAST * np.minimum(v / V_FAST_NM3, 1.0) ** K


def twitch_tetanus_ratio(volume_nm3: np.ndarray) -> np.ndarray:
    """A : échelle de saturation (en secousses) ; T(a) = T1·(1 - e^{-a/A}) / (1 - e^{-1/A})."""
    return np.exp(np.log(A_SLOW) + _frac(volume_nm3) * (np.log(A_FAST) - np.log(A_SLOW)))


def twitch_tau_ms(volume_nm3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constantes de montée et de relaxation (ms) de la secousse."""
    f = _frac(volume_nm3)
    tr = np.exp(np.log(TAU_SLOW_MS[0]) + f * (np.log(TAU_FAST_MS[0]) - np.log(TAU_SLOW_MS[0])))
    td = np.exp(np.log(TAU_SLOW_MS[1]) + f * (np.log(TAU_FAST_MS[1]) - np.log(TAU_SLOW_MS[1])))
    return tr, td


def rest_potential(volume_nm3: np.ndarray) -> np.ndarray:
    """Potentiel de repos (mV) du MN, gradient mesuré -68 (rapide) -> -48 (lent)."""
    v = np.asarray(volume_nm3, dtype=np.float64)
    return np.clip(-8.2 * np.log(v / 1e11) - 44.6, V_REST_FAST, V_REST_SLOW)


def input_gain(volume_nm3: np.ndarray, ref_nm3: float) -> np.ndarray:
    """Résistance d'entrée relative (V / ref)^-0,67 : facteur appliqué aux entrées synaptiques du MN."""
    return (np.asarray(volume_nm3, dtype=np.float64) / ref_nm3) ** -R_IN_EXP


def mn_class(volume_nm3: np.ndarray) -> np.ndarray:
    """Étiquette lent / intermédiaire / rapide (proxy par volume, pour les rapports)."""
    r = np.asarray(volume_nm3, dtype=np.float64) / V_FAST_NM3
    return np.where(r < CLASS_BOUNDS[0], "slow", np.where(r < CLASS_BOUNDS[1], "intermediate", "fast"))


def tetanic_torque(volume_nm3: np.ndarray) -> np.ndarray:
    """Couple maximal (µN·mm) d'une unité motrice (tétanos), T1 / (1 - e^{-1/A})."""
    return torque_per_spike(volume_nm3) / (1.0 - np.exp(-1.0 / twitch_tetanus_ratio(volume_nm3)))
