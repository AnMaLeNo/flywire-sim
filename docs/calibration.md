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

## 5. Deuxième passe (ciblée) : afférents actifs, lobe antennaire, corps pédonculé

Constat de départ (capteurs v1, PR #3) : avec `CALIBRATED` (w = 2, γ = 0,5) le seul bruit spontané des
afférents embrase le cerveau entier. Méthode imposée : (a) vérifier d'abord que les capteurs et le corps
n'introduisent pas de signal faux, (b) localiser l'étage responsable par élimination, (c) confronter à la
littérature, (d) ne toucher qu'à des grandeurs dont la valeur réelle est incertaine, jamais au nombre de
neurones ni au support des connexions (`tests/test_calibration.py` vérifie que le support de la matrice
est inchangé). Outil : `scripts/banc_al_gain.py` (paramètres, dépression, gain eLN, DN stimulés, odeur,
ablation de groupes de capteurs, taux par population et par fenêtre de 100 ms).

### 5.1 Audit des capteurs (côté corps)

- Effectifs BANC re-comptés depuis les annotations officielles (`banc.antennal_lobe_populations`) :
  ORN 3 006 (2 811 antennes, 195 palpes), PN 699, LN 429 (eLN 123 / iLN 306 d'après le signe NT ; les
  eLN sont en majorité `nt_verified = acetylcholine`, les iLN GABA/glutamate), KC 4 553, MBON 104.
- Taux spontanés forcés : ORN 8 Hz (de Bruyne 2001), hygro/thermo 3 Hz toniques, soies phasico-toniques,
  chordotonaux ~10 Hz au repos. Sans aucun canal actif : 0 spike dans tout le réseau (pas de fuite).
- Bug corrigé : les 35 cellules `Ir40a,cooling` du sacculus (glomérules VP1d/VP1l) étaient pilotées
  comme des cellules « sèches » toniques. Frank et al. 2017 : la triade hygrosensorielle comprend une
  cellule sèche (Ir40a, VP4), une cellule **froide** (Ir40a, VP1) et une cellule humide (Ir68a, VP5) ;
  elles reçoivent maintenant −dT/dt (canal `hygro_cooling`).
- Tactile tarsal séparé par tarsomère (`tactile1..5`, contact réel du tarsomère), vibro-chordotonaux (`club`)
  et `ppk23` (phéromone de contact) branchés ; test : chaque canal ne tire que si son tarsomère touche.

### 5.2 Localisation par élimination (500 ms, mouche posée)

| Capteurs actifs | Paramètres | Total (spk/s) | PN | KC | DN | MN patte |
|---|---|---|---|---|---|---|
| aucun | CALIBRATED | 0 | 0 | 0 | 0 | 0 |
| ORN seuls (3 006) | CALIBRATED | 1 871 000 (croissant) | 158 | 160 | 26 | 5,8 |
| hygro/thermo seuls (121) | CALIBRATED | 1 860 000 (croissant) | 151 | 151 | 25 | 4,0 |
| soies, JO, labellum, propriocepteurs (~9 000) | CALIBRATED_V2 | 14 400 (décroissant) | 0 | 0 | 0 | 0 |
| hygro/thermo seuls | CALIBRATED_V2 | 10 800 | 2,0 | 0 | 0,2 | 0,4 |
| ORN seuls | CALIBRATED_V2 | 30 100 | 1,5 | 0 | 0 | 0 |
| tous | CALIBRATED_V2 | 49 200 | 2,5 | 0 | 0,2 | 0,3 |

(CALIBRATED_V2 inclut les mécanismes de § 5.3 ; fichiers `results/al_gain_*.txt`.)

Conclusions : (1) le corps et les mécanorécepteurs ne sont pas en cause (pas de propagation au-delà du
premier relais) ; (2) l'entrée qui embrase est le **lobe antennaire** (ORN, ou même les seuls 121 hygro/thermo à 3 Hz :
leurs 41 000 synapses sortantes ciblent PN et LN), puis PN → KC → tout le cerveau ; (3) l'embrasement
n'existe qu'avec le gain global de `CALIBRATED` (w × 7 par rapport à Shiu) : avec les paramètres de
Shiu et al. 2024 les mêmes entrées ne se propagent pas. `CALIBRATED` avait été réglé pour compenser un
défaut situé ailleurs (DN → MN trop faible, § 4) en amplifiant tout le cerveau : c'est ce réglage global
qui est fautif, pas les capteurs.

### 5.3 Mécanismes retenus (littérature) et grandeurs réglées

Le LIF homogène n'a ni synapses électriques ni dépression ; les deux sont documentés précisément dans le
lobe antennaire et réglés **par type de neurone / de synapse annoté**, sans changer le support :

1. **eLN → PN et eLN → eLN sont électriques** (Yaksi & Wilson 2010 : transmission bidirectionnelle,
   insensible au Cd²⁺, abolie par `shakB²` ; les coefficients de couplage mesurés au soma sont faibles ;
   Huang 2010). Le BANC les compte comme synapses chimiques et le simulateur les traitait comme des
   synapses excitatrices pleines, d'où eLN à 320 Hz et PN à 100 Hz spontanés (Shiu, sans dépression).
   Réglage : efficacité `ELN_ELECTRICAL_GAIN` sur ces seules sorties (`banc.synaptic_efficacy`).
   Balayage (Shiu, U_ORN = 0,5) : gain 1 → PN 104 Hz, KC 36 Hz ; 0,5 → PN 52, KC 13 ; 0,25 → PN 15, KC
   0,7 ; 0,1 → PN 3–4, KC 0. Retenu 0,1 (le seul qui ramène les KC spontanées à ~0 Hz, Turner 2008) ;
   la valeur réelle du couplage est inconnue → paramètre explicitement incertain.
2. **Dépression ORN → PN** (Kazama & Wilson 2008 : probabilité de libération élevée, ~40 % de dépression
   déjà à 7 Hz, réponses PN transitoires). Modèle Tsodyks-Markram par neurone présynaptique
   (`LIFNetwork(std_U=vecteur)`), U ≠ 0 seulement pour les ORN (`banc.depression_U`). Balayage (odeur
   « levure », ORN à 80 Hz, PN cibles par fenêtre de 100 ms, début → régime) :
   U = 0 : 75 → 55 Hz ; U = 0,15/τ 200 ms : 46 → 16 ; U = 0,3/τ 200 : 30 → 8 ; U = 0,5/τ 200 : 19 → 5.
   Retenu **U = 0,3, τ_rec = 200 ms** : dépression ~30 % à 8 Hz (Kazama : ~40 %, dont une part
   d'inhibition présynaptique non modélisée), réponse PN transitoire (×3,7 début/régime).
3. Paramètres globaux : retour à Shiu et al. 2024 (`CALIBRATED_V2 = LIFParams(tau_rec=200)`), qui
   reproduit une activité spontanée de base ~0 Hz hors afférents.

Non implémentés (documentés) : inhibition présynaptique GABA-B des terminaisons ORN (Olsen & Wilson 2008,
Root 2008), seuil KC / rétroaction APL (Lin 2014), hétérogénéité intrinsèque des LN (Seki 2010).

### 5.4 Résultats avec tous les capteurs actifs (CALIBRATED_V2, U = 0,3, τ 200 ms, gain eLN 0,1)

| Condition | Total (spk/s) | ORN | PN | eLN | iLN | KC | DN | MN patte |
|---|---|---|---|---|---|---|---|---|
| spontané | 49 000 | 7,9 | 2,5 | 28 | 9 | 0 | 0,2 | 0,3 |
| DNg100 à 50 Hz | 58 000 | 7,8 | 3,7 | 38 | 12 | 0 | 0,4 | 1,4 (13 % actifs) |
| 20 types DN « marche » à 50 Hz | 148 000 (stable sur 500 ms) | 7,9 | 3,0 | 34 | 10 | 0 | 9,9 | 2,7 (20 % actifs) |
| odeur levure (ORN cibles 80 Hz) | 99 000 | 21,6 | 4,0 (levure 30 → 8) | 56 | 15 | 0,2 | 0,2 | 0,4 |
| comparaison : CALIBRATED + mêmes mécanismes, DNg100 | 496 000 et croissant | 7,8 | 8,9 | 47 | 20 | 3,6 | 16,8 | 4,9 |

Stable, sans emballement, activité de fond ~0 Hz hors afférents et lobe antennaire ; le corps pédonculé
est silencieux au repos et répond faiblement à l'odeur (KC 0,2 Hz ; in vivo : réponses clairsemées de
quelques KC). Le recrutement DN → MN est réel mais faible (MN à 1–3 Hz, 13–20 % actifs) : c'est le
problème de § 4 (gain DN → MN dans le ganglion ventral, propriétés intrinsèques des MN), à traiter
**dans le VNC** et non par un gain global — c'est exactement l'erreur de la v1.

### 5.5 Ce qui reste incertain

- Le gain eLN 0,1 et U = 0,3 sont des grandeurs incertaines réglées sur des cibles qualitatives
  (KC ~0 Hz au repos, PN transitoires) ; ils devront être revus si l'inhibition présynaptique GABA-B ou
  l'APL sont implémentées.
- La dépression s'applique à toutes les sorties des ORN (dont ORN → LN), pas seulement à ORN → PN.
- Les hygro/thermorécepteurs à 3 Hz toniques dominent l'activité de fond du lobe (canal VP) : valeur à
  confronter à des enregistrements (Enjin 2016 n'en donne pas de taux au repos).
- La réponse PN à l'odeur (30 Hz début) est faible par rapport à in vivo (100–200 Hz, Bhandawat 2007) :
  la force unitaire ORN → PN de Shiu (w = 0,275 par synapse) est probablement sous-estimée pour cette
  synapse « très forte » (Kazama & Wilson) ; à régler par type de synapse, pas globalement.
- La classe `mushroom_body_extrinsic_neuron` (4 neurones, dont les 2 APL) tire à ~30 Hz au repos : ce sont
  les APL (GABA vérifié), excitées par leurs entrées non-KC. In vivo l'APL est **non impulsionnelle**
  (Papadopoulou 2011, potentiels gradués) ; en LIF ses spikes ne font qu'inhiber les KC (sens conservé),
  mais elle devra passer en neurone gradué comme les photorécepteurs quand l'APL sera traitée.
