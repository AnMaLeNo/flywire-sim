# Capteurs de la mouche (v1) : du corps MuJoCo et de l'environnement aux neurones sensoriels du BANC

Objectif : que **tous les afférents sensoriels annotés dans le BANC v888** reçoivent un signal issu du
corps simulé (MuJoCo) ou de l'environnement 3D, avec une physiologie tirée des études sur
*D. melanogaster*, avant toute expérience comportementale et avant les ailes. La connectivité officielle
n'est pas modifiée : les capteurs ne font que **forcer les spikes** (ou injecter un courant gradué pour les
neurones non impulsionnels) des neurones sensoriels, qui restent clampés vis-à-vis du réseau
(`banc.clamp_afferents`, voir `calibration.md`).

Code : `flywire_sim/body/senses.py` (tous les sens sauf la vision), `flywire_sim/body/vision.py` (œil
composé), `flywire_sim/body/environment.py` (odeurs, goûts, température, humidité, vent, son, hypoxie),
`flywire_sim/body/model.py` (caméras oculaires, antennes articulées, soies de contact),
`flywire_sim/body/sim.py` (boucle, forces du vent et du son sur les antennes),
`scripts/experiments_senses.py` (expériences en scène 3D), `tests/test_senses.py`.

## 1. Populations du BANC branchées

Colonnes officielles utilisées : `Class`, `Sub Class`, `Function`, `Body Part`, `Side`, `Cell Type`.

| Organe / modalité | Population BANC (`Sub Class` / `Function`) | n | Grandeur lue | Physiologie retenue |
|---|---|---|---|---|
| Pattes : hair plates, claw, hook, club, campaniformes, soies tactiles | `*_hair_plate_neuron`, `*_claw/hook/club_chordotonal_organ_neuron`, `*_campaniform_sensillum_neuron`, `*_bristle_neuron` (tactile) | 4 300 | angles/vitesses articulaires, charge, contact tarsal | Tuthill & Wilson 2016 ; Mamiya 2018 (claw = position, hook = direction, club = vibration) |
| Pattes : goût | `*_taste_bristle_gustatory_neuron` × {sugar, bitter, water, low_salt, contact_pheromone} | 1 000 | goût du sol sous les tarses en contact | Hiroi 2002 ; Ling 2014 (r_max 100 Hz sucre, 60 amer/eau, 40 sel/phéromone) |
| Labelle et taste pegs | `labellum_taste_bristle_gustatory_neuron`, `labellum_taste_peg_gustatory_neuron` × 9 classes | 344 | goût au point de contact du labelle | idem ; taste pegs = sucre et acides gras (Steck 2018) |
| Soies tactiles tête / thorax / abdomen / trompe / antennes | `bristle_neuron` par `Body Part` (1 400 tête dont 1 206 interommatidiales, 298 thorax, 781 abdomen, 256 trompe, 42 antennes) | 2 800 | force de contact des géométries correspondantes | phasico-toniques, adaptation ~200 ms (Tuthill & Wilson 2016) |
| Organe de Johnston A/B (vibration) | `johnstons_organ_A/B_neuron` | 511 | vitesse angulaire passe-haut des charnières antennaires (son, marche) | Kamikouchi 2009 ; Yorozu 2009 : A basses fréquences, B hautes ; réf. 1° à 200 Hz |
| Organe de Johnston C/D/E/F (position) | `johnstons_organ_C/D/E/F_neuron`, scindés en deux moitiés opposées par axe | 660 | déviation statique des antennes (gravité, vent), réf. 3° | Kamikouchi 2009 : cellules toniques directionnelles ; Yorozu 2009 : vent |
| ORN antennaires et palpaux | `antenna/maxillary_palp_olfactory_receptor_neuron` × classe d'odeur (`Function`) | 3 006 (dont 256 sans classe) | concentration de la classe d'odeur à l'ariste / au palpe | Hallem & Carlson 2006 : spontané ~8 Hz, max ~200 Hz ; adaptation ~0,5 s (Nagel & Wilson 2011) |
| Thermorécepteurs de l'ariste | `Ir21a,cooling` (6), `Gr28a,heating` (6), `Ir68a,evaporation` (16) | 28 | dérivée de la température à l'ariste (inertie 50 ms) | Budelli 2019 : phasiques, seuil ~0,5 °C |
| Hygrorécepteurs du sacculus | `Ir40a,dry` (70), `Ir68a,humid` (20) | 90 | humidité relative ambiante (tonique) | Enjin 2016 ; Knecht 2017 |
| Cou | `neck_chordotonal_organ_neuron`, `prosternal_hair_plate_neuron` | 75 | angles du cou | proprioception (Paulk & Gilbert 2006, calliphore) |
| Organe de Wheeler | `wheelers_chordotonal_organ_neuron` | 52 | accélération passe-haut du thorax (vibration du substrat) | chordotonal thoracique |
| Oxygénation | `oxygenation_neuron` (abdomen) | 20 | hypoxie ambiante `Environment.hypoxia` | entrée réservée (voir conception_corps.md) |
| Œil composé | `R7`, `R8`, `L1`-`L5`, `Mi1` (types colonnaires) | 9 815 | luminance/couleur dans la direction de chaque colonne | § 2 |
| **Silencieux** | halteres, aile, tegula, marge de l'aile (mouche sans ailes au repos), multidendritiques (nociception), goût interne du pharynx (ingestion non modélisée), tractus reproducteur | ~1 400 | — | — |

Tous les canaux : Poisson à `r_spont + r_max × grandeur ∈ [0, 1]`, adaptation optionnelle par
soustraction d'une moyenne glissante (`Channel.rate`).

## 2. Vision

### 2.1 Physiologie retenue (D. melanogaster)

| Grandeur | Valeur | Source |
|---|---|---|
| Ommatidies par œil | ~750-800, champ ~160° par œil | Ready 1976 ; Heisenberg & Wolf 1984 |
| Angle inter-ommatidial | ~4,6° (jusqu'à ~5,5° en périphérie) | Götz 1964 ; Gonzalez-Bellido 2011 |
| Acceptance angle Δρ | ~4,5° (3,5-5°) | Götz 1965 |
| Photorécepteurs | R1-6 (Rh1, pic ~480 nm + UV) : potentiels gradués, pas de spikes ; R7 (Rh3 345 nm / Rh4 375 nm), R8 (Rh5 437 nm / Rh6 508 nm), 30 % d'ommatidies « pale » | Salcedo 1999 ; Rister 2013 |
| Cinétique R1-6 | temps au pic 20-40 ms, adaptation sur 0,1-10 s | Juusola & Hardie 2001 |
| Lamina | L1-L4 hyperpolarisées par l'augmentation de lumière (histamine), dépolarisées par la baisse ; L1 → voie ON, L2/L3 → voie OFF ; L3 code la luminance | Joesch 2010 ; Clark 2011 ; Ketkar 2020 |
| Résolution temporelle | fusion > 100 Hz | — |

### 2.2 Modèle

1. **Rendu** : 5 caméras par œil (`{l,r}_eye_{front,lateral,back,up,down}`, fov 90°, 32×32) au centre de
   chaque œil du scan micro-CT, rendues toutes les 5 ms (MUJOCO_GL=egl, ~10 ms par image ; le pas
   physique de 0,1 ms est inchangé). Ombres et reflets désactivés.
2. **Échantillonnage colonnaire** : chaque neurone colonnaire du BANC (`R7`, `R8`, `L1`-`L5`, `Mi1`)
   reçoit la couleur moyenne (7 rayons, gaussienne Δρ = 4,5°) dans **sa direction de vue**.
3. **Rétinotopie** : direction de vue = rayon vecteur de la position du neurone (export officiel Codex
   `neuron_attributes`, voxels 4×4×45 nm) depuis le centre d'une sphère de 100 µm ajustée de façon
   robuste sur les types denses (R7/R8/L1/L2/L5, `robust_sphere_center`). Cette coquille couvre ~±80°
   autour de sa normale moyenne, identifiée à l'axe optique de l'œil (±60° d'azimut dans le repère tête),
   le dorsal du volume au dorsal de la mouche. Résultat sur le BANC : œil droit 5-95 % en azimut −115°…+30°,
   élévation ±65° ; œil gauche symétrique. **Approximation documentée** : l'ordre rétinotopique
   colonnaire exact (et l'inversion du chiasma externe) n'est pas démontré ici ; le lobe optique gauche
   est moins complet dans les annotations (3 262 vs 6 553 neurones colonnaires), sans signification
   biologique.
4. **Photorécepteurs R1-6** : log-luminance du canal Rh1 (RGB → 0,15 R + 0,55 G + 0,30 B), deux filtres
   passe-bas de 8 ms (pic ~20 ms), adaptation lente 500 ms. Potentiel gradué (pas de spikes).
5. **Lamina** : courants injectés dans les LIF `L1`/`L2` ∝ baisse de contraste (référence lente 150 ms)
   et `L3` ∝ obscurité relative au niveau adapté (`Eyes.currents`, via `LIFStepper.step(currents=…)`) ;
   ils sont donc dépolarisés à l'OFF et muets à l'ON, comme observé.
6. **R7/R8** : spikes forcés au taux `150 Hz × canal spectral / (0,3 + niveau adapté)` (canal UV ≈ B pour
   R7 ; R8 : B en pale, G en yellow). Histamine → inhibiteurs dans le réseau.
7. Ocelles et DRA (polarisation) : non modélisés.

## 3. Antennes, vent, son

Chaque antenne du scan reçoit deux charnières faibles (funicule/pédicelle, `{l,r}_antenna_{pitch,yaw}`)
avec masse 1,5 µg à la base de l'ariste : raideur 0,1 µN·mm/rad, inertie → résonance ~400 Hz (Göpfert &
Robert 2002 : ~420 Hz passive), déviation gravitaire ~1°, amortissement ζ ≈ 0,3.
Le vent (`Environment.wind`, mm/s) est une **traînée** F = 1,7·10⁻⁴ µN/(mm/s) × (vent − vitesse de l'ariste)
appliquée à la base de l'ariste (0,5 m/s → ~5°, Yorozu 2009) ; le son (`sound` ∈ [0, 1] à `sound_hz`) une
force sinusoïdale équivalente à ~1° d'oscillation. Gravité, vent, son et vibrations de la marche passent
donc **tous par la physique** avant d'être lus par l'organe de Johnston.

## 4. Résultats (300 ms, mouche posée, régime `banc.SENSES`, `scripts/experiments_senses.py`)

Fichiers : `results/senses_*_senses.txt`. Taux en Hz par neurone, moyennés sur la population et le côté.

| Expérience | Témoin | Stimulus | Lecture |
|---|---|---|---|
| Posture | thorax à 0,79 mm, 6 tarses en contact (soies tactiles 30 Hz, charge lue par patte) ; antennes déviées de ~0,2° par la gravité | — | proprioception + contact fonctionnels |
| Vision : boîte noire 3×3 mm à 3 mm à gauche | log-luminance Rh1 −0,55 / −0,54 (G/D) ; lamina 1,0 / 0,7 Hz | Rh1 gauche −0,95 (droite inchangée) ; lamina gauche 4,0 Hz, droite 0,7 ; R7/R8 gauche 80 → 77 Hz ; Tm gauche 0,1 → 0,3 Hz | réponse **OFF latéralisée** du bon côté ; T4/T5, LC muets (scène statique, régime dépressif) |
| Olfaction : source « levure » à 1 mm à gauche | ORN 8 Hz | ORN levure 97 / 86 Hz (G/D, concentration 0,65 / 0,54), autres ORN 8 Hz ; PN +3 Hz ; KC +0,5 Hz | ORN spécifiques et latéralisés ; réponse centrale faible (§ 5) |
| Goût : sol sucré | GRN 0 Hz | GRN sucre des tarses 76 / 78 Hz, GRN amers 0 ; DN +0,3 Hz | GRN sélectifs ; pas d'extension de trompe (MN9 muet) |
| Vent 0,5 m/s venant de la gauche | JO C-F 2,9 / 2,7 Hz ; antennes −0,17 / +0,17° (lacet) | antennes déviées à −1,1 / −0,6° (lacet) et ±0,5° (tangage) ; JO C-F 7,6 / 7,2 Hz ; JO A/B inchangé | déviation statique lue par les cellules de position, pas par celles de vibration (Yorozu 2009) |
| Chant 200 Hz | JO A/B 2,1 / 2,5 Hz | JO A/B 28,5 / 29,1 Hz, JO C-F 7 Hz (redressement de l'oscillation ±0,5°) | vibration lue par A/B (Kamikouchi 2009) |

Les réponses centrales (PN, KC, DN) restent faibles : voir § 5.

## 5. Limites et suite

### 5.1 L'activité spontanée des afférents embrase le lobe antennaire (constat de calibration)

Avec `banc.CALIBRATED` (PR #2, réglé sur DNg100 → MN avec afférents silencieux), le simple bruit
spontané physiologique des ORN (8 Hz × 3 000) — ou même les seuls hygro/thermorécepteurs toniques (90
neurones) — suffit à embraser le lobe antennaire (LN 180 Hz, PN 150 Hz), le corps pédonculé (KC 155 Hz,
au lieu de < 1 Hz) puis tout le cerveau (1,8 M spikes/s, DN 23 Hz, MN de patte 5 Hz). Balayage
(`results/` et journal de session) : `size_norm=0.75` éteint le lobe (PN 3 Hz, KC 1 Hz) mais aussi la
voie DNg100 → MN ; la dépression synaptique (`std_U=0.2, tau_rec=500`, Kazama & Wilson 2008) stabilise
(PN 26 Hz, KC 11 Hz, DN 1 Hz, 20 MN toniques à 3 Hz) au prix d'un recrutement DNg100 plus faible
(28/391 MN au lieu de 106). Aucun réglage **global** ne satisfait les deux : le lobe antennaire et le corps
pédonculé reposent sur des mécanismes spécifiques absents du LIF homogène —

- inhibition présynaptique GABA-B des terminaisons ORN par les LN (Olsen & Wilson 2008 ; Root 2008),
  perdue par le clamp des afférents ;
- forte dépression ORN → PN (Kazama & Wilson 2008) ;
- seuil élevé des cellules de Kenyon et rétroaction APL (Lin 2014) → codage clairsemé.

`banc.SENSES` (= CALIBRATED + dépression) est donc le régime **provisoire** des expériences sensorielles ;
la prochaine passe de calibration doit introduire ces mécanismes (par type de synapse / de neurone, à
partir des annotations officielles) puis re-régler DN → MN.

### 5.2 Autres limites

- Rétinotopie colonnaire et chiasma externe approximés (§ 2.2) ; L1-L5 modélisés en LIF (réels : gradués).
- Le gain lamina (30 mV par unité log de contraste) est fixé pour qu'un contraste de ~25 % atteigne le seuil.
- Déviation antennaire au vent ~1° à 0,5 m/s (ordre de grandeur de Yorozu 2009, non ajusté finement).
- Hygrorécepteurs toniques (pas de dérivée), thermorécepteurs purement phasiques.
- L'organe de Johnston A/B ne sépare pas encore les bandes de fréquence.
- Goût interne (pharynx), nociception (multidendritiques), ocelles, DRA et vol : silencieux ou différés.
