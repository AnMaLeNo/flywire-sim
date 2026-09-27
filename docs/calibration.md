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

## 6. Signal moteur faible : enquête à la source et correction de complétude de la moelle

Question posée : avec CALIBRATED_V2, DNg100 à 50 Hz ne recrute les MN de patte qu'à 1–3 Hz. Erreur de
notre côté (corps, capteurs, conversion), limite des données, ou limite du modèle ?

### 6.1 Côté corps : pas de bug trouvé

- Mapping : 391/391 MN de patte du BANC reliés à un muscle (`body/muscles.py`), aucun non relié.
- Unités MuJoCo cohérentes (mm, mg, µN·mm) : masse 0,94 mg, couples actifs max des actuateurs 1–6 µN·mm
  par muscle, posture tenue passivement sous gravité (thorax à 0,79 mm), tripode scripté fonctionnel (PR #1).
- Conversion spikes → activation (`per_spike` 0,05, τ 30 ms, saturation à 1) : linéaire jusqu'à ~10
  spikes cumulés — cohérent avec la sommation de twitchs d'un muscle d'insecte ; pas le maillon faible.
- Capteurs : au repos (mouche posée, sans cerveau) les seuls afférents actifs sont ORN (8 Hz, Hallem &
  Carlson 2006), organe de Johnston (2–7 Hz, déviation gravitaire), soies tarsales du segment en contact
  (7,9 Hz soutenu sur 615 neurones), claw (3,7 Hz) et hair plates (4,4 Hz) ; total 35 000 spk/s dont
  24 000 ORN. Rien d'aberrant.

### 6.2 Localisation : l'étage DN → interneurones de la moelle est sous-alimenté (données)

Comparaison statique des exports officiels Codex, synapses ≥ 5, par super-classe postsynaptique :

| Étage | BANC / référence |
|---|---|
| MN ← interneurones VNC | 0,92 (réf. MANC) |
| MN ← DN | 1,02 (réf. MANC) |
| MN ← afférents | 0,99 (réf. MANC) |
| IN VNC ← IN, ← DN, ← afférents, ← AN | 0,21–0,23 (réf. MANC), homogène sur 36 hémilignées (0,08–0,36) |
| sorties des DN dans la moelle | 0,30 (réf. MANC) |
| entrées des neurones du cerveau central | ~0,3 (réf. FAFB) |

Les entrées des motoneurones sont complètes ; c'est la couche prémotrice qui manque de synapses : un IN
VNC du BANC reçoit 319 synapses (24,5 partenaires) contre 1 309 (75,7 partenaires) dans le MANC. Le
papier BANC le dit : 18 % des synapses ont leurs deux extrémités identifiées, contre 42–44 % pour
FAFB/MANC. L'export BANC est déjà seuillé à 3 synapses par paire : aucune connexion « cachée » à
récupérer. Le poids unitaire de Shiu 2024 (0,275 mV) a été calibré sur les comptes FAFB : appliqué aux
comptes BANC de la moelle, la couche prémotrice est sous-alimentée ~4×. Ce n'est ni un bug du corps ni un
paramètre LIF : c'est une limite de complétude de la reconstruction.

Vérifications dynamiques (LIF homogène, mêmes paramètres, sans corps) :

| Réseau | DNg100 à 50 Hz → IN VNC | → MN patte |
|---|---|---|
| BANC brut | 0,1 Hz | 1,9 Hz (13 % actifs) |
| MANC (référence, même LIF) | 7,5 Hz (14 % actifs) | 12,3 Hz (19 % actifs) |

Le même modèle, avec les comptes de la moelle d'un connectome plus complet, produit le recrutement
attendu : le déficit est bien dans les comptes.

### 6.3 Correction retenue : rééchelonnage par neurone des entrées de la moelle (`completeness.py`)

Principe : pour chaque interneurone de la moelle (super-classe `ventral_nerve_cord_intrinsic`), ratio = entrées BANC / médiane des entrées de son type cellulaire dans le MANC (repli :
hémilignée, puis super-classe) ; facteur = 1/ratio borné à [1, 3] ; toutes les entrées existantes du
neurone sont multipliées par ce facteur. **Aucune connexion ajoutée ni retirée** (support de la matrice
identique, signes conservés — `tests/test_completeness.py`) ; motoneurones et afférents inchangés
(entrées complètes), neurones ascendants inchangés (§ 6.5). Le cerveau n'est pas corrigé (§ 6.4). Le MANC ne sert que de référence de comptes ;
le réseau simulé reste le BANC.

Choix du plafond (3) : les synapses manquantes sont surtout des **partenaires manquants** (0,32× de
paires) plus que des paires sous-comptées (0,75× de synapses par paire). Multiplier les paires
récupérées reporte tout le poids manquant sur elles : avec le facteur statique complet (médiane 5,3,
plafond 10) 88 % du poids entrant d'un IN est porté par des paires ≥ 26 synapses-équivalents (une seule
présynaptique suffit à faire tirer le neurone) contre 51 % dans le MANC. Le réseau s'embrase alors dès
qu'un afférent tire. On a donc cherché le plafond qui reproduit la dynamique de la référence MANC dans
les deux régimes (essais avec IN **et** AN corrigés, sans corps) (drive sensoriel de repos synthétique : 20 % des soies de patte à 8 Hz, hair plates 4 Hz,
40 % des chordotonaux 4 Hz ; et DNg100 à 50 Hz) :

| Plafond | repos sensoriel → IN VNC / MN | DNg100 → IN VNC / MN |
|---|---|---|
| 1 (brut) | 0,0 / 0,0 Hz | 0,1 / 1,9 Hz |
| 1,5 | 0,0 / 0,0 | 0,3 / 1,1 |
| 2 | 0,0 / 0,0 | 4,1 (13 %) / 3,9 (27 %) |
| **3** | **0,2 (6 %) / 0,1 (10 %)** | **6,1 (20 %) / 6,4 (39 %)** |
| 4 | embrasement à 800 ms | 7,8 / 8,9 |
| 10 | embrasement (16 / 21 Hz) | embrasement |
| MANC (référence) | 0,3 (6 %) / 0,6 (12 %) | 7,5 (14 %) / 12,3 (19 %) |

Le plafond est une grandeur de calibration explicite (incertaine, 2–3 acceptables, 4 instable), pas une
mesure biologique : c'est documenté comme tel.

### 6.4 Pourquoi le cerveau n'est pas corrigé

La même méthode avec le FAFB comme référence (facteurs médians 2,0 DN, 2,8 cerveau central) rend le
cerveau instable sous DNg100 seul (DN 39 % actifs), alors que le FAFB lui-même, simulé avec le même LIF,
reste silencieux dans cette condition. Deux raisons mesurées : (1) la récupération du BANC est biaisée
vers l'excitation (entrées inhibitrices récupérées 0,87× moins que les excitatrices : part inhibitrice
des entrées du cerveau central 0,35 contre 0,39 dans le FAFB) ; (2) la médiane par type surestime la
cible pour des types très divergents entre datasets (APL : 3 900 entrées BANC contre 44 000–50 000 FAFB →
saturation). Le cerveau BANC brut est stable avec tous les capteurs (§ 5.4) et sa voie DNg100 → moelle
est complète côté sorties directes sur les MN ; on le laisse tel quel. Dans la moelle, la récupération
n'est pas biaisée vers l'excitation (part inhibitrice 0,51 contre 0,46 dans le MANC).

### 6.5 Avec le corps et tous les capteurs : les ascendants ne doivent pas être corrigés

Boucle complète (corps MuJoCo, tous les capteurs, 1 s, `banc_al_gain.py`) :

| Correction (plafond 3) | Repos : total / DN / MN patte | DNg100 à 50 Hz : DN / MN patte |
|---|---|---|
| aucune (§ 5.4) | 49 000 spk/s / 0,2 Hz (1 %) / 0,3 Hz (3 %) | 0,4 Hz / 1,4 Hz (13 %) |
| IN VNC + ascendants | 253 000 / 12,7 Hz (21 %) / 6,8 Hz (32 %) | 14,6 Hz (24 %) / 6,9 Hz (34 %) |
| **IN VNC seulement** | **56 000 / 0,2 Hz (1 %) / 2,0 Hz (23 %)** | **0,4 Hz (3 %) / 2,5 Hz (25 %)** |

Avec les ascendants corrigés ×3, les afférents de repos (soies tarsales, JO, ORN) suffisent à fermer une
boucle moelle → AN → cerveau → DN → moelle : 21 % des DN tirent à 12 Hz au repos et DNg100 n'ajoute
plus rien (ablations : pattes seules ou tout sauf les pattes donnent le même régime). Le « 6,4 Hz » du
tableau § 6.3 était donc surtout ce recrutement de tout le cerveau descendant, pas la voie DNg100 → MN.
In vivo les DN sont majoritairement silencieux au repos ; on ne corrige donc que les interneurones
intrinsèques de la moelle. Résultat : repos silencieux côté DN, MN de patte toniques à 2 Hz sur 23 %
(compatible avec l'activité tonique des MN lents de posture, Azevedo 2020), DNg100 → MN 2,5 Hz (25 %)
au lieu de 1,4 Hz. Le gain reste modeste, et la raison est mesurable : les sorties de DNg100 dans le
BANC comptent 407 interneurones cibles (8 500 synapses) contre 834 (22 400) dans le MANC, alors que ses
cibles motrices directes sont complètes (136 MN / 2 270 synapses contre 111 / 1 680). Le déficit est fait
de **partenaires absents** ; rééchelonner les entrées existantes des cibles récupérées ne les recrée pas,
et le faire plus fort (plafond ≥ 4, ou ascendants) embrase le reste. On est à la limite de ce que la
complétude du BANC permet pour cette voie.

Boucle corps-cerveau (`body_brain_loop.py`, 1 s, IN VNC corrigés) : repos 88/391 MN de patte actifs à
9 Hz ; DNg100 97/391 à 10 Hz ; 43 types DN « marche » 148/391 à 20 Hz (activation musculaire moyenne
0,04, tibia extensor dominant). Posture tenue, pas de rythme de pas ni de déplacement : la faiblesse
neuronale (§ 6.5) et l'absence de classes de MN / de twitch (§ 6.6) restent à traiter avant la marche.

### 6.6 Ce qui reste (dans l'ordre)

1. **Décision données** : garder la moelle BANC (voie DNg100 → prémoteurs à ~0,4× de la référence, non
   récupérable sans ajouter de connexions) ou brancher la moelle MANC (mâle, complète) sous le cerveau
   BANC — c'est un changement de source, pas un réglage.
2. Classes de MN lents / intermédiaires / rapides (Azevedo 2020 : R_in 700 / 300 / 150 MΩ, recrutement
   par taille, MN lents toniques ~30 Hz) : le LIF homogène ne les distingue pas ; les MN annotés par
   muscle dans le BANC permettent de les typer.
3. Inhibition présynaptique des afférents : 92–99 % des synapses reçues par les axones sensoriels de patte
   sont GABAergiques (BANC) ; elles sont pour l'instant supprimées (`clamp_afferents`) au lieu de moduler
   la libération (Dallmann 2025 : inhibition présynaptique sélective des propriocepteurs pendant la marche).
4. Muscle : twitch avec montée physiologique, sommation, saturation et force par classe (Azevedo 2020),
   à ne régler qu'une fois l'étage neuronal validé.

## 7. Moelle MANC sous le cerveau BANC (réseau hybride, `flywire_sim/hybrid.py`)

Décision prise après § 6 : la moelle simulée n'est plus celle du BANC (interneurones à ~0,2× de leurs
synapses) mais le **MANC v1.2.1** (Janelia, export officiel Codex `flywire-data/codex/data/manc/1.2.1`,
`scripts/download_data.sh`), connectome **complet** du ganglion ventral d'un **mâle**. Le cerveau reste
le **BANC v888** (femelle). C'est un changement de source de données, pas un réglage, et l'assemblage
est un **hybride de deux animaux** (sexes différents, individus différents) : il ne doit pas être présenté
comme le connectome d'une seule mouche. La correction de complétude (§ 6.3, `completeness.py`) disparaît.

### 7.1 Règles d'assemblage (`hybrid.build`)

| Étape | Règle | Effectif |
|---|---|---|
| Cerveau | tous les neurones BANC sauf ceux résidant dans la moelle (`vnc_resident` : IN du VNC, MN, afférents et neurones dont la région annotée est le VNC, hors classes pontées) | 133 070 gardés (dont 403 DN, 253 AN, 80 sensoriels ascendants sans homologue MANC : ils gardent leurs connexions cérébrales, aucune moelle) |
| Moelle | le MANC entier (`manc.py` : colonnes renommées au schéma BANC, super-classes harmonisées, côté depuis `Sub Class`/`Nerve`/`Soma side`, `body_part` des MN depuis `Class`, des afférents depuis `Sub Class`) | 23 665 neurones, 1 372 588 paires ≥ 5 synapses |
| Pont | DN, AN, sensoriels ascendants et efférents ascendants appariés par **(super_class, cell_type, side)** ; k = min(n_BANC, n_MANC) paires fusionnées en un noeud (entrées cérébrales BANC + sorties/entrées de la moelle MANC) | **2 949 fusions** : 913 DN, 1 596 AN, 436 sensoriels ascendants, 4 efférents |
| Non appariés MANC | restent dans la moelle **sans aucune connexion cérébrale** (`bridge == "unpaired"`) | 784 : 415 DN, 266 AN, 99 sensoriels ascendants, 4 ; 488 sont des surnuméraires d'un type présent des deux côtés, 296 d'un type absent du BANC |
| Connexions | somme des deux matrices officielles ré-indexées ; **aucune synapse ajoutée** — `tests/test_hybrid.py` tire 4 000 entrées de W et vérifie que chacune est une paire (pré, post) de l'export BANC ou de l'export MANC | 156 735 noeuds, 2 452 560 entrées |
| Signes | chaque connexion garde le signe du jeu dont elle vient (NT prédit BANC pour les synapses cérébrales, MANC pour la moelle ; MN forcés excitateurs) ; 153/2 949 fusionnés ont un NT prédit différent dans les deux jeux — conservés tels quels, pas d'arbitrage silencieux | |
| Afférents MANC | annotés au niveau classe seulement (`chordotonal_organ`, `hair_plate`, `taste_bristle`…) ; l'organe et la fonction (claw/hook/club, plaques pilifères, soies, sucre/amer/…) sont **transférés depuis le BANC par `cell_type`** (annotation majoritaire du type ; pureté médiane 0,99) | 3 650 afférents de patte MANC branchés au corps par organe |
| MN de patte MANC | mêmes noms de muscle que le BANC pour 17 types → reliés aux actionneurs sans changer `muscles.py` ; les 36 types `MNfl10`, `MNml76–86`, `MNhl59–75`… n'ont pas de muscle nommé dans l'export | **330/396 reliés**, 66 non reliés (listés au démarrage de `body_brain_loop.py`), aucun mapping inventé |

Un premier essai copiait aux DN/AN surnuméraires les entrées ou sorties cérébrales d'un frère du même type
(« copy_in/copy_out »). Retiré : c'est une connexion dérivée, non présente dans un export, et son effet
mesuré était < 5 % sur tous les taux.

### 7.2 Résultats, réseau seul (`scripts/hybrid_regimes.py`, CALIBRATED_V2, 500 ms, sans capteurs)

| Réseau | total spk/s | DN | AN | IN VNC | MN patte | MN aile |
|---|---|---|---|---|---|---|
| MANC seul, DNg100 (2) à 50 Hz | 112 000 | 0,6 Hz | 3,3 Hz (8 %) | 6,6 Hz (14 %) | **10,8 Hz (19 %)** | 23,7 Hz (48 %) |
| Hybride, repos | 0 | 0 | 0 | 0 | 0 | 0 |
| Hybride, DNg100 à 50 Hz | 237 000 | **11,3 Hz (17 %)** | 7,9 Hz (14 %) | 5,5 Hz (18 %) | 7,8 Hz (24 %) | **109 Hz (73 %)** |
| idem, AN → cerveau/DN coupé | 55 000 | 0,2 Hz | 1,9 Hz | 3,3 Hz | 4,9 Hz (24 %) | 13 Hz |
| idem, cerveau/AN → DN coupé | 139 000 | 0,4 Hz | 3,5 Hz | 6,6 Hz | 10,6 Hz (24 %) | 20 Hz |

Lecture :
- La voie **DNg100 → moelle → MN de patte** est maintenant celle de la référence (10,8 Hz, 19–24 % des
  396 MN, contre 1,4–2,5 Hz avec la moelle BANC, § 6) : le problème du § 6 est réglé par les données.
- Un nouveau régime apparaît, **la boucle moelle → AN → cerveau → DN → moelle** : DNg100 seul recrute
  17 % des DN à 11 Hz en moyenne, avec des types à 250–300 Hz (DNa03, DNa06, DNg33, DNb02, DNg75, DNa02 :
  non physiologique), et 73 % des MN d'aile à 109 Hz (les MN des muscles de vol tirent à ~5 Hz en vol :
  pathologique ; ils sont alimentés surtout par IN19B043/IN19B040/IN19B067 et vMS12 de la moelle, puis par
  DNp31/DNut040). Couper AN → cerveau supprime la boucle (DN 0,2 Hz) ; couper cerveau → DN aussi (0,4 Hz),
  en retrouvant exactement le régime du MANC seul.
- Origine mesurée côté données (`scripts/dn_input_balance.py`, `results/dn_input_balance.csv`) : sur les
  344 types de DN communs, l'export BANC récupère 0,68× les synapses **excitatrices** reçues par les DN
  dans le FAFB v783 mais seulement 0,58× les **inhibitrices** ; E/I médian 1,43 (BANC) contre 1,17 (FAFB),
  et pour les DN de la boucle : DNa03 1,70 vs 1,25, DNa06 1,03 vs 0,64, DNb02 1,37 vs 0,73, DNg33 4,86 vs
  2,26. Le cerveau BANC est donc **biaisé vers l'excitation sur ses DN** par rapport à la référence ; ce
  biais était masqué tant que la moelle BANC ne renvoyait presque rien par les AN.

### 7.3 Résultats avec le corps et tous les capteurs (`scripts/body_brain_loop.py`, 500 ms)

| Régime | spikes totaux/ms | MN de patte actifs (sur 330 reliés) | taux des actifs | activation musculaire moy. / max | hauteur thorax |
|---|---|---|---|---|---|
| repos (capteurs seuls) | 260 | 66 | 27 Hz | 0,024 / 0,72 | 0,80 mm |
| DNg100 à 50 Hz | 301 | 79 | 27 Hz | 0,027 / 0,89 | 0,78 mm |
| 43 types DN « marche » à 50 Hz | 291 | 71 | 37 Hz | 0,036 / 0,90 | 0,77 mm |

Elle tient debout (aucune chute, hauteur stable), les MN toniques de posture tirent (extenseur du tibia,
réducteur du fémur, rotateurs sternaux — cohérent avec Azevedo 2020), mais **repos et commande sont
presque indiscernables** : le fond sensoriel suffit à amorcer la boucle du § 7.2, qui sature ce que la
commande peut ajouter. Pas de déplacement (0,1 mm en 500 ms, dû à la pose initiale), pas d'alternance
des six pattes : **la marche n'est pas validée**.

### 7.4 Ce qui reste (dans l'ordre)

1. **Boucle AN → cerveau → DN** : c'est maintenant le verrou. Sa source mesurée est un déficit
   d'inhibition sur les DN du BANC (§ 7.2) ; à traiter sans gain global et sans toucher aux connexions —
   options à évaluer : (a) FAFB v783 comme cerveau (complet, femelle, E/I de référence) avec le même
   pont par type, (b) MN d'aile : le corps n'a pas d'ailes et leurs 66 MN ne pilotent rien, mais leur
   embrasement recrute des AN ; (c) inhibition présynaptique des afférents (§ 6.6, point 3) qui alimentent
   les AN.
2. Classes de MN lents / intermédiaires / rapides et muscle (§ 6.6, points 2 et 4), une fois la boucle réglée.
3. Les 66 MN de patte MANC sans nom de muscle : chercher leur muscle dans Lesser et al. 2024 (table S1)
   plutôt que de les laisser muets.
