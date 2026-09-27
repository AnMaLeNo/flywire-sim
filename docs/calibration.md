# Calibration de la dynamique BANC (cerveau + ganglion ventral)

Objectif : que la commande descendante « marche » recrute les motoneurones (MN) de patte du BANC v888 à
des taux physiologiques, sans emballement du réseau, **sans toucher à la connectivité officielle**
(FlyWire/BANC, Codex). Tout ce qui suit est mesuré avec `scripts/banc_calibrate.py` et
`scripts/banc_walk_analysis.py` (fichiers `results/calib_*.txt`, `results/banc_walk*.png`).

## 1. Valeurs mesurées dans la littérature (D. melanogaster)

| Grandeur | Valeur | Source |
|---|---|---|
| Classes de MN de patte (tibia) | lents / intermédiaires / rapides ; recrutés dans cet ordre | Azevedo et al. 2020 (Nature) |
| Potentiel de repos | rapides ≈ −68 mV, intermédiaires ≈ −60 mV, lents ≈ −48 mV | idem |
| Activité au repos | rapides et intermédiaires muets ; lents ≈ 30 Hz | idem |
| Résistance d'entrée | ≈ 150 MΩ (rapides), 300 MΩ (interm.), 700 MΩ (lents) | idem |
| Force par spike | croissante des lents vers les rapides (≈ ×10 par classe) ; demi-max en ≈ 8,5 ms, saturation après ≈ 10 spikes (interm./rapides) | idem |
| Taux max des neurones du VNC | \(r_{max}\) ≈ 200 Hz ; constante de temps ≈ 20 ms | Pugliese et al. 2025 (modèle MANC) |
| DN qui déclenchent le rythme de marche | DNg100, DNb08 (2 et 4 neurones dans le BANC) | Pugliese et al. 2025 ; Cheong et al. 2024 |
| Fréquence de pas | ≈ 7–15 Hz selon la vitesse | Mendes et al. 2013 ; Wosnitza et al. 2013 |
| Vitesse de marche | ≈ 7–45 mm/s (typ. ≈ 28 mm/s) | Mendes et al. 2013 |
| GABA et glutamate | inhibiteurs dans le SNC (GABA-A/B, GluCl) | Liu & Wilson 2013 |
| Histamine des photorécepteurs | inhibitrice (canaux HisCl / Ort) | Gengs et al. 2002 |
| Afférents primaires | tirent depuis leur courant récepteur ; les synapses centrales sur leurs terminaisons sont de l'inhibition présynaptique GABA-B, pas une source de spikes | Root et al. 2008 ; Clarke et al. 2015 |

## 2. Hypothèses du modèle (fixées à partir de la littérature)

1. **Signe des synapses.** NT vérifié quand il existe (BANC, minuscules, éventuellement composé), sinon NT
   prédit (Eckstein et al. 2024). GABA, glutamate et histamine → inhibiteurs ; le reste → excitateur ;
   MN forcés excitateurs (leurs prédictions sont peu fiables et leurs cibles sont hors réseau).
   *Correction importante* : la version v0 ne reconnaissait pas les NT vérifiés en minuscules
   (`gaba`, `glutamate`, `gaba,nitric_oxide`…) → ~18 000 neurones inhibiteurs vérifiés étaient traités
   comme excitateurs. Fraction de synapses inhibitrices : 23 % (v0) → **43 %** après correction.
2. **Afférents clampés** (`banc.clamp_afferents`) : les neurones sensoriels ne reçoivent pas d'entrée
   centrale ; seuls les capteurs du corps (`senses.py`) ou une stimulation expérimentale les font tirer.
   Sans cela, une avalanche « VNC → sensoriels → ascendants → cerveau → lobes optiques » embrasait
   ~70 000 neurones (`results/calib_ignition.txt`).
3. **LIF de Shiu et al. 2024** conservé (repos −52, seuil −45, τm 20 ms, τsyn 5 ms, réfractaire 2,2 ms,
   délai 1,8 ms) : dt 0,1 ms commun avec la physique.
4. **Normalisation par la taille** (`size_norm` = γ) : le poids effectif reçu par un neurone est divisé par
   \(\max(1, n_{in}/\tilde n)^{\gamma}\) (\(n_{in}\) = synapses d'entrée, \(\tilde n\) = médiane du réseau).
   Justification : la résistance d'entrée décroît avec la taille (Azevedo 2020 : 700 → 150 MΩ) ; Pugliese
   et al. 2025 normalisent gain et seuil par la taille. Approximation : la taille est estimée par
   \(n_{in}\) et un même γ sert à tout le réseau.

## 3. Balayage et régime retenu

Commande : DNg100 (2 neurones) à 50 Hz, 500–1000 ms, réseau `min5` (≥ 5 synapses).

| γ | w (mV/synapse) | MN de patte actifs | taux moyen / médian / max des actifs | VNC actifs (taux) | cerveau actifs | stable ? |
|---|---|---|---|---|---|---|
| 0 | 0,275 (Shiu) | 73/391 | 11 / 6 / 76 Hz | 506 (25 Hz) | 837 | oui |
| 0 | 0,40 | 107/391 | 26 / 10 / 160 Hz | 1258 (35 Hz) | 1998 | oui |
| 0,5 | 1,6 | 22/391 | 3 / 2 / 7 Hz | 247 (5 Hz) | 44 | oui (quasi silencieux) |
| **0,5** | **2,0** | **104/391** | **34 / 19 / 148 Hz** | **2600 (33 Hz)** | **6100** | **oui** |
| 0,5 | 2,2 | 117/391 | 29 / 10 / 132 Hz | 3096 (40 Hz) | 7435 | oui |
| 1 | 5–7 | 0/391 | – | 80–200 (10 Hz) | 43–174 | oui (silencieux) |

Avant la correction des signes, *tous* les réglages donnaient soit le silence, soit l'embrasement de
tout le SNC (390/391 MN à 300–400 Hz, 70 000 neurones du cerveau actifs) ; adaptation et dépression
synaptique ne faisaient que transformer l'embrasement en bouffées.

Régime retenu (`banc.CALIBRATED` : w = 2,0, γ = 0,5) :

- 104/391 MN de patte actifs, médiane ≈ 19 Hz (ordre de grandeur des MN lents au repos, 30 Hz), max 148 Hz
  (< \(r_{max}\) ≈ 200 Hz) ; tous les groupes musculaires des 6 pattes recrutés, symétrie gauche/droite ;
- réponse graduée à la commande : 20 Hz de DNg100 → 26 MN à 2 Hz ; 50 Hz → 104 MN à 34 Hz ; 100 Hz → 128 MN
  à 26 Hz ; population « marche » de 43 DN → 127–131 MN ;
- VNC stable (~3000 neurones, 33–44 Hz), cerveau ~6000 neurones actifs ; pas d'emballement sur 1 s ;
- en boucle fermée avec le corps MuJoCo (`scripts/body_brain_loop.py`, `results/body_loop_dng100.txt`) :
  129/391 MN actifs à 14 Hz, activation musculaire moyenne 0,025 (v0 : 0,54 avec co-contraction), la
  mouche tient debout ; sans stimulation : 0 MN actif, 1100 neurones sensoriels/VNC actifs (5 spikes/ms).

## 4. Ce qui n'est pas encore reproduit

- **Principe de taille** : les MN recrutés en premier sont les gros (taille relative 0,68 vs 0,42 pour les
  muets ; corrélation taux ~ taille +0,3), alors qu'in vivo les lents (petits) sont recrutés d'abord.
  γ = 1 corrige la tendance mais éteint le réseau ; il faudra une hétérogénéité explicite des MN
  (repos −68/−60/−48 mV, Rin) plutôt qu'un γ global.
- **Rythme de pas** : la sortie motrice montre des bouffées à ~16–19 Hz mais peu de puissance dans la
  bande 5–20 Hz (12–28 %) et pas d'alternance fléchisseur/extenseur nette. Chez Pugliese et al. le rythme
  émerge dans un sous-réseau MANC de la patte avant ; ici le corps et la proprioception doivent
  contribuer (Mendes 2013 : le tripode persiste sans proprioception, mais la précision se dégrade).
- **Muscle** : activation générique (τ 30 ms, +0,05/spike) ; force par spike, twitch de 8,5 ms et
  saturation à ~10 spikes par classe de MN restent à implémenter.
- Pas de marche en boucle fermée : déplacement 0,13 mm en 1 s.
