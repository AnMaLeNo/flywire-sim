# Étape 2 — Donner un corps au cerveau : recherche & conception

*Espèce : **Drosophila melanogaster**, femelle adulte (c'est l'animal des jeux de données FAFB/FlyWire
et BANC). Toutes les études citées ci-dessous portent sur cette espèce.*

## 0. Résumé des décisions proposées

| Sujet | Décision | Pourquoi |
|---|---|---|
| Système nerveux | Passer de FAFB v783 (cerveau seul) au **BANC v888** (cerveau **+ moelle** = ganglion ventral, même mouche, publié par FlyWire/Harvard en 2025-2026, export officiel Codex) | Les motoneurones des pattes, du cou, de l'abdomen sont dans le ganglion ventral, absent de FAFB. Le BANC contient les 805 motoneurones **annotés muscle par muscle** et 17 000 neurones sensoriels **annotés organe par organe** : l'interface cerveau↔corps est donnée par les chercheurs, pas inventée. |
| Moteur physique | **MuJoCo** (C, Apache-2.0, Google DeepMind ; wheels Python Linux/macOS/Windows) | Corps articulés + contacts stables au pas de 0,1 ms à l'échelle du millimètre, actionneurs d'adhésion (pattes qui collent), tendons, muscles. Moteur générique, ce n'est pas un simulateur de mouche. |
| Corps | **Notre propre modèle MJCF** (arbre cinématique, degrés de liberté, muscles, adhésion, capteurs, collisions écrits par nous), mais **formes et dimensions issues d'un vrai scan micro-CT** : maillages et origines de segments de NeuroMechFly (Lobato-Rios et al. 2022, femelle adulte de *D. melanogaster*, licence Apache-2.0, `flywire_sim/body/meshes/`). Aucun code, contrôleur ou paramètre de simulation de ce projet n'est repris. | Décision validée : un scan micro-CT réel est plus fidèle que toute reconstruction à partir de primitives (longueurs, masses par segment, positions exactes des articulations). |
| Langage | **Python** (numpy/scipy + mujoco) pour la v0, comme le cerveau. Portage Rust/GPU des boucles chaudes plus tard si nécessaire | Une seule pile pour cerveau + corps + capteurs ; itération rapide. Le « jeu » (rendu 3D, caméra, interactions) viendra par-dessus (viewer MuJoCo d'abord, moteur de jeu ensuite via socket). |
| Ailes | **Amputées** : ailes retirées du corps, les ~950 neurones sensoriels des ailes/tegula/base d'aile ne reçoivent rien, les MN des ailes sont libres mais n'actionnent rien (haltères conservés, ils ne battent qu'en vol) | Correspond à une manipulation expérimentale classique (« wing clipping ») dont les effets comportementaux sont documentés ; les circuits de vol restent intacts dans le cerveau mais sans effet mécanique. |

Décisions validées : MuJoCo ; géométrie micro-CT autorisée si on ne peut pas faire mieux (c'est le cas).

## 1. L'animal : ordres de grandeur

- Longueur ~2,5–2,8 mm, masse **~1 mg** ; répartition mesurée : tête 0,125 mg, thorax 0,31 mg,
  abdomen 0,45 mg, pattes 0,11 mg, ailes 0,005 mg. Le femur de la patte (~0,6 mm) sert de proxy de
  taille corporelle.
- 6 pattes, **5 articulations / 7 degrés de liberté par patte** : thorax–coxa (3 DoF), coxa–trochanter
  (1), trochanter–fémur (1, quasi fixe), fémur–tibia (1), tibia–tarse (1) ; le tarse a 5 tarsomères
  passifs + pré-tarse (griffes + **pulvilles adhésifs**).
- **14 muscles intrinsèques par patte** (18 pour la patte avant en comptant le thorax), innervés par
  **63–70 motoneurones par patte** (BANC : 69/70 avant, 63 milieu, 63 arrière), regroupés en unités
  motrices lentes / intermédiaires / rapides (« size principle » : recrutement des petits MN lents
  d'abord, puis des grands MN rapides pour les mouvements balistiques ; Azevedo et al. 2020).
  Les MN de mouche ne sont jamais inhibiteurs (la prédiction « GABA » de nombreux MN dans le BANC est
  un artefact connu : on les force excitateurs vers les muscles).
- Marche : **tripode** à toutes les vitesses (tétrapode/onde à très basse vitesse), la vitesse est
  contrôlée quasi uniquement par la **fréquence de pas (jusqu'à 10–20 Hz)** ; fémur–tibia oscille
  de plusieurs milliers de °/s ; vitesses de marche typiques 5–30 mm/s ; patte du milieu la plus rapide
  à revenir ; démarrage souvent en tripode + virage (Strauss & Heisenberg 1990, Mendes 2013,
  Wosnitza 2013, DeAngelis 2019, Chun 2021).
- Adhésion : les pulvilles (coussins poilus + sécrétion) et les griffes permettent de marcher sur
  murs et plafonds ; décollement par traction / torsion / levée (Niederegger & Gorb 2003). À 1 mg,
  l'adhésion domine la gravité : indispensable même sur sol plat pour un rendu réaliste.
- Proprioception de la patte : organe chordotonal fémoral (FeCO) avec 3 sous-types — **claw**
  (angle statique fémur–tibia, flexion vs extension), **hook** (direction du mouvement), **club**
  (mouvement bidirectionnel + vibrations 100–1 600 Hz) (Mamiya 2018) ; + plaques de poils (hair
  plates, angles proximaux), sensilles campaniformes (contrainte dans la cuticule = charge),
  soies tactiles (~1 000 par paire de pattes dans le BANC).

## 2. Ce que le BANC nous donne comme « prises » sur le corps

Comptages dans l'export officiel `neurons.csv.gz` BANC v888 (158 262 neurones, 3,04 M connexions ≥ 3 synapses).

**Sorties (805 motoneurones, `Super Class = motor`, colonnes `Body Part` / `Function`)**

| Partie du corps | MN | Fonctions annotées (exemples) |
|---|---|---|
| pattes avant / milieu / arrière | 139 / 126 / 126 | `flex_femur_tibia_joint` (33/patte), `flex_coxa_trochanter_joint` (20), `extend_coxa_trochanter_joint` (16), `pull_long_tendon` (griffes/tarse, 16), `extend_tibia_tarsus_joint` (8), `move_coxa_anterior/posterior/medial/lateral`, `extend_femur_tibia_joint` (4), `femur_reductor`… avec le **nom du muscle** (`tibia_flexor`, `sternotrochanter`, `tergopleural_promotor`…) |
| abdomen | 168 | muscles abdominaux (respiration, ponte, posture) |
| cou | 49 | `neck_yaw` / `neck_pitch` / `neck_roll` |
| proboscis / pharynx / glandes | 35 + 22 + 2 | positionnement, ouverture labellaire, pompage pharyngien, salivation |
| antennes | 10 | mouvement du scape |
| ailes / haltères | 64 / 27 | puissance, steering (inactifs après amputation) |
| yeux | 4 | muscles rétiniens |

**Entrées (17 086 neurones sensoriels, annotés par organe et modalité)**

| Sens | Neurones (BANC) | Ce que le corps doit calculer |
|---|---|---|
| Vision | rétine 1 839 + inter-ommatidiaux 1 205 + ocelles 5 (**~750–800 ommatidies/œil**, 8 photorécepteurs R1–R8 chacune, résolution ~5°, temps de réponse ~10 ms) | rendu d'une image par œil (caméras hémisphériques ~800 pixels), chromaticité (UV/bleu/vert), mouvement |
| Olfaction | antennes ~2 600 + palpes 226 ; classes annotées : `yeasty`, `fruity`, `alcoholic_fermentation`, `decaying_fruit`, `aversive`, `CO2`, phéromones volatiles… | champs de concentration d'odeurs dans la scène, par classe |
| Ouïe / vent / gravité | organe de Johnston ~1 100 (`auditory low/high frequency`, `position`, `direction`) | vibration de l'air, direction du vent, déflection des antennes par la gravité |
| Thermo / hygro | `Ir21a cooling`, `Gr28a heating`, `Ir40a dry`, `Ir68a humid` (~120) | température / humidité locales |
| Goût | labellum 521, pattes ~900, marge d'aile 594, pharynx ~100 : `sugar (Gr5a/Gr64f)`, `bitter (Gr66a/Gr33a)`, `water (ppk28)`, `low salt`, `fatty acids`, `amino acids`, `contact pheromone (ppk23/25)`, `heavy metal (Ir47a)` | composition chimique de la surface touchée par chaque organe |
| Toucher | 6 367 soies tactiles (pattes, thorax, abdomen, tête), nociception 153 | contacts par zone de cuticule (MuJoCo donne les forces de contact) |
| Proprioception | FeCO claw/club/hook (~700), hair plates (~230), campaniformes (~60), `mechanical_strain`, `joint_angle`, `position`, `direction` | angles, vitesses, charge articulaires |
| Viscéral | `oxygenation` (7), `hemolymph` (8), tube digestif 64, tractus reproducteur 34 | état interne (voir §4) |

Tout ça est déjà **connecté** dans le graphe : GRN → interneurones → MN, FeCO → prémoteurs → MN,
etc. Notre travail est de fournir le corps qui ferme la boucle.

## 3. Le corps à construire (sans vol)

1. **Squelette rigide** (MJCF généré par `flywire_sim/body/model.py`) : tête (proboscis 2 segments,
   antennes 3 segments, yeux), cou (3 DoF), thorax, abdomen (5 segments, 4 DoF), 6 pattes ×
   (coxa 3 DoF, trochantérofémur 2 DoF, tibia 1 DoF, tarse 1 DoF + 4 tarsomères passifs), haltères
   conservés, **ailes absentes**. Géométrie et masses : scan micro-CT (v0 réalisée : 68 segments,
   82 DoF, 0,94 mg, unités mm/g/s). Collisions : capsules sur les pattes, ellipsoïdes sur le corps.
2. **Muscles** : un actionneur par muscle annoté dans le BANC (≈ 14–18 par patte, ~40 tête/cou/
   trompe, abdominaux). Modèle : chaque spike de MN ajoute un « twitch » (montée ~5–10 ms,
   décroissance 20–40 ms pour les lents, plus rapide pour les rapides) ; l'activation somme les MN
   du muscle pondérés par leur taille (size principle) → force = activation × F_max × courbe
   force-longueur-vitesse (Hill simplifié). F_max estimée pour qu'un tripode soutienne ~1 mg avec
   marge (≈ 50–100 µN par muscle fléchisseur).
3. **Adhésion** : actionneur d'adhésion MuJoCo au pré-tarse, commandé par les MN `pull_long_tendon`
   (griffes) + contact ; détachement quand la patte se lève (règle documentée par la cinématique).
4. **Contacts / friction / gravité** : sol, murs, objets ; friction cuticule–verre/bois ; gravité
   9,81 m/s². Échelle réelle en mètres (2,8 mm) avec pas de 0,1–0,2 ms.
5. **Capteurs** (encodeurs « corps → spikes ») : pour chaque classe sensorielle du tableau §2, une
   fonction physique → taux de décharge Poisson des neurones concernés (ex. : angle fémur–tibia →
   claw neurons tuned à des angles ; contact d'un tarsomère → soies de ce segment ; image → R1–R8
   des ommatidies concernées ; concentration sucre au labellum → GRN sucre du côté touché).
6. **Comportements attendus pour valider** (sans rien programmer d'autre que le corps) : posture
   debout stable, initiation de marche par stimulation des DN (DNp09 avant, MDN arrière, DNa01/02
   virage, BDN2), coordination tripode émergente, extension du proboscis sur sucre, toilettage
   (séquence tête → ailes → thorax → abdomen par suppression hiérarchique, Seeds 2014) quand on
   « salit » la cuticule (activation des soies), évitement d'odeurs aversives et de CO₂.

## 4. États internes (« hémolymphe ») — minimal mais réel

Le connectome contient les capteurs, pas les variables. On tient un petit modèle physiologique :
- **Sucre circulant** (glucose/tréhalose/fructose) : monte quand la pompe pharyngienne ingère du
  sucre (MN `pharyngeal_pumping` actifs + labellum sur sucre), baisse avec l'activité (coût ∝
  activation musculaire). Il alimente les capteurs internes annotés (`hemolymph`, neurones Gr43a du
  cerveau = capteur de fructose, IPC = capteur de glucose) : c'est ainsi que faim/satiété modulent la
  réponse au sucre (Miyamoto 2012, Musso 2021).
- **Eau** (soif → GRN `ppk28 water`), **O₂/CO₂** : capteurs `oxygenation` (7) et `CO2 volatile`
  (91) existent ; chez la drosophile la respiration est passive (diffusion trachéenne, spiracles),
  la coma anoxique survient < 2 % O₂ ; un modèle ventilation-abdominale ↔ O₂ n'est justifié que si
  on veut des atmosphères hypoxiques → **optionnel, v2**.
- **Température** (préférence ~25 °C) via `Ir21a/Gr28a/Ir68a`, **fatigue musculaire** simple.
- Sommeil/circadien : circuits présents dans le cerveau, horloge externe (heure de la scène) → **v2**.

## 5. Ailes retirées : ce qu'on sait

Classique depuis Morgan/McEwen (1918) et Benzer (1967) : couper les ailes réduit fortement la
**phototaxie** et la **géotaxie** des mouches qui marchent, indépendamment du temps de récupération,
et l'effet se mime en collant les ailes (donc c'est l'absence de retour sensoriel/moteur des ailes
qui compte, pas la blessure) (Brembs & Gorostiza). Elles marchent normalement et toilettent plus.
Dans notre modèle : ailes absentes du corps, entrées sensorielles d'aile nulles → on pourra tester
si cet effet **émerge** du connectome (bon test de réalisme).

## 6. Architecture logicielle

```
 flywire_sim/           cerveau+moelle (BANC, LIF numpy/scipy, dt=0.1 ms)   ← existe
 flywire_sim/lif.py           LIFStepper : intégration pas à pas pour boucler avec le corps   ← fait
 flywire_sim/body/model.py    MJCF généré (micro-CT + DoF + muscles + capteurs), sans ailes     ← fait (v0)
 flywire_sim/body/muscles.py  MN BANC (391 MN de patte, nommés par muscle) → twitch → actionneurs ← fait (v0)
 flywire_sim/body/senses.py   capteurs → 4 305 neurones sensoriels de patte (proprio, charge, toucher, sucre) ← fait (v0)
 flywire_sim/body/sim.py      boucle fermée : capteurs → spikes → LIF → spikes MN → muscles → MuJoCo ← fait (v0)
 flywire_sim/interoception/   hémolymphe (sucre, eau, O2), température                          ← v1
 scripts/body_stand_test.py   posture debout sous gravité + rendu
 scripts/body_muscle_test.py  tripode scripté sur les muscles (sans cerveau) → avance
 scripts/body_brain_loop.py   expérience cerveau+corps : stimulation des DN de marche
```
Boucle temps : cerveau dt = 0,1 ms (158 k neurones ≈ 4–5 s de calcul / s simulée sur CPU),
physique dt = 0,1–0,2 ms, échange cerveau↔corps toutes les 1 ms. Cible v0 : ~5–10× plus lent que
le temps réel sur un portable ; accélération ensuite (Rust/GPU pour le LIF, `mjx` pour MuJoCo).
Rendu : viewer MuJoCo (mac/win/linux) pour la v0 ; le « jeu » (caméra libre, injection d'odeurs,
de sucre, d'obstacles) sera une couche au-dessus, moteur de jeu à choisir quand la simulation tient.

## 7. Résultats préliminaires sur le BANC (aujourd'hui)

- Réseau BANC construit (158 262 neurones, 3,04 M connexions) et simulé avec le même LIF.
- **Stimuler MDN (4 neurones « marche arrière ») ou DNa02 (virage) recrute bien les ~390 MN des
  6 pattes** via le ganglion ventral — le chemin cerveau → moelle → muscles existe et fonctionne.
- Mais avec les paramètres FAFB (poids 0,275 mV, seuil 3 synapses) le réseau **s'emballe**
  (22 000 neurones actifs à 300 Hz) : la moelle est plus récurrente/excitatrice que le cerveau et
  le seuil 3 synapses ajoute beaucoup de petites connexions. À calibrer (seuil 5–10 synapses, poids
  plus faible, inhibition) — c'est la première tâche de la v0. Premier balayage : le réseau est
  **tout-ou-rien** (poids 0,275 → emballement ; 0,15 → MDN n'active plus que 17 neurones), donc il
  faudra un mécanisme de régulation absent du LIF minimal (adaptation / inhibition tonique /
  saturation synaptique) plutôt qu'un simple réglage de poids. Le corps aidera aussi (les MN
  déchargent sur des muscles, pas dans le réseau ; la proprioception ferme la boucle).
- Sucre → MN9 ne se reproduit pas encore dans le BANC (0 Hz) alors qu'il marche dans FAFB : la
  zone gustative du BANC est moins relue (proofread) que FAFB, ou l'annotation MN9 diffère. À creuser ;
  option de repli : greffer la moelle BANC sous le cerveau FAFB via les neurones descendants
  (le BANC fournit les correspondances `fafb_cell_type`).

### État v0 (corps)

- Debout sous gravité sans muscle (ressorts articulaires passifs) : 6 tarses au sol, force de contact
  totale = poids, hauteur du thorax ≈ 0,8 mm.
- Tripode scripté sur les muscles nommés BANC (boucle ouverte) : avance de 5 mm en 2 s (2,5 mm/s ;
  réel 10–30 mm/s) sans basculer → les signes/couples des muscles sont plausibles, à affiner.
- Boucle fermée BANC ↔ corps (200 ms, 43 DN « marche » à 20 Hz, w=0,21, adaptation 3 mV) : les
  391 MN de patte sont tous reliés à un actionneur, 364 tirent (~180 Hz) → co-contraction
  généralisée (activation moyenne 0,54) : la mouche se raidit et se dresse, ne marche pas encore.
  Cause : dynamique VNC non calibrée (§7), pas le corps. Tests : `tests/test_body.py`.

## 8. Plan

| Étape | Contenu | Estimation |
|---|---|---|
| v0 « elle tient debout et bouge les pattes » | ~~MJCF corps (pattes 7 DoF, tête, abdomen) ; muscles ← MN ; adhésion ; proprio + toucher pattes → spikes ; scène sol plat ; stimulation des DN de marche~~ **fait** ; reste : calibration BANC (rythme de marche dans le VNC), viewer interactif | 1 session |
| v1 « elle se nourrit et explore » | goût (labellum/pattes), hémolymphe (faim), proboscis, olfaction (champs d'odeur), antennes/JO (vent, gravité), thermo | 1 session |
| v2 « elle voit » | rendu par œil → photorécepteurs (2 × ~800 ommatidies), ocelles ; sommeil/horloge ; O₂/CO₂ optionnel | 1–2 sessions |
| v3 « jeu » | caméra libre, outils d'expérience (déposer sucre/odeur/obstacle, pincer une patte), export mac/win | 1 session |

## 9. Sources principales

- Bates, Phelps, Kim, Yang et al., *Distributed control circuits across a brain-and-cord connectome*, Nature 2026 (BANC) ; export Codex `gs://flywire-data/codex/data/banc/888/` ; documentation des colonnes `sjcabs/fly_connectome_data_tutorial`.
- Azevedo, Lesser, Phelps, Mark et al., *Connectomic reconstruction of a female Drosophila ventral nerve cord*, Nature 2024 (atlas MN → muscles).
- Lesser et al., *Synaptic architecture of leg and wing premotor control networks*, Nature 2024 (7 DoF, 18 muscles, 69 MN, pas de MN GABA).
- Azevedo et al., *A size principle for recruitment of Drosophila leg motor neurons*, eLife 2020.
- Mamiya, Gurung, Tuthill, *Neural coding of leg proprioception in Drosophila*, Neuron 2018 ; Mamiya et al. 2023 (biomécanique FeCO).
- Strauss & Heisenberg 1990 ; Mendes et al. 2013 ; Wosnitza et al. 2013 ; DeAngelis et al. 2019 ; Chun et al., eLife 2021 (tripode, ARSLIP).
- Niederegger & Gorb 2003 (pulvilles/griffes) ; Langer, Ruppersberg & Gorb 2004 (adhésion des setae).
- Rayshubskiy et al. 2020/2023 (DNa01/DNa02 steering) ; Bidaye et al. 2014 & 2026 (MDN, marche arrière) ; Braun et al., Nature 2024 (réseaux descendants) ; Cheong et al. 2024.
- Seeds et al., eLife 2014 (hiérarchie du toilettage) ; Hampel et al. 2015.
- Miyamoto et al., Cell 2012 (Gr43a, fructose) ; Musso et al. 2021 ; Oh et al. 2019 (IPC glucose).
- Brembs & Gorostiza (SfN 2009 / 2011), McEwen 1918, Benzer 1967 (ailes coupées : photo-/géotaxie).
- Masses par segment : mesures reprises dans Lobato-Rios et al. 2022 (tête 0,125 mg, thorax 0,31, abdomen 0,45, pattes 0,11, ailes 0,005 ; longueur 2,8 mm).
- Vision : ~750–800 ommatidies/œil, 8 photorécepteurs (Ready 1976 ; Currea et al. 2018).
