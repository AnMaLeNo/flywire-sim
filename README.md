# flywire-sim — faire tourner le cerveau d'une mouche (FlyWire v783)

Objectif final : un jeu / simulation 3D (PC & Mac) où une mouche est pilotée par une simulation de
son vrai cerveau. **Étape 1 (ce dépôt)** : installer et faire tourner le cerveau seul, à partir
uniquement des données officielles publiées par les chercheurs du projet FlyWire.

## C'est quoi FlyWire ?

FlyWire (Princeton, Seung & Murthy labs + consortium international) est le premier **connectome
complet** du cerveau d'un animal adulte complexe : une drosophile femelle (*Drosophila
melanogaster*, jeu de données EM « FAFB »).

- 139 255 neurones reconstruits et vérifiés (proofread) manuellement (~33 années-personnes).
- ~50 millions de synapses chimiques détectées automatiquement, dont ~34 M dans les 2,7 M de
  connexions neurone→neurone qui comptent ≥ 5 synapses (seuil recommandé par les auteurs).
- Pour chaque neurone, un **neurotransmetteur dominant prédit** depuis les images EM
  (acétylcholine ≈ excitateur ; GABA, glutamate ≈ inhibiteurs ; dopamine, sérotonine, octopamine
  = modulateurs).
- Des annotations hiérarchiques : flux (afférent / intrinsèque / efférent), super-classe
  (sensoriel, central, optique, descendant, moteur…), classe, type cellulaire, côté, nerf.

Publications de référence (Nature, oct. 2024) :
- Dorkenwald *et al.*, « Neuronal wiring diagram of an adult brain » — le connectome (v783).
- Schlegel *et al.*, « Whole-brain annotation and multi-connectome cell typing… » — les annotations.
- Eckstein, Bates *et al.* (Cell 2024) — prédiction des neurotransmetteurs.

Important : le connectome ne contient **pas** de « poids » appris. Il contient la structure
(qui est connecté à qui, avec combien de synapses, et quel neurotransmetteur). C'est nous qui
choisissons un modèle de neurone et qui en déduisons des poids : `poids = signe(NT) × nb_synapses`.

## Données utilisées (sources officielles uniquement)

| Fichier | Source | Contenu |
|---|---|---|
| `codex783/neurons.csv.gz` | Codex (Princeton), bucket public `gs://flywire-data/codex/data/fafb/783/` | 139 255 neurones, NT dominant + probabilités |
| `codex783/connections.csv.gz` | idem | connexions pre→post par neuropile, ≥ 5 synapses, `syn_count`, NT |
| `codex783/classification.csv.gz` | idem (annotations Schlegel *et al.*) | flow, super_class, class, sub_class, side, nerve |
| `codex783/labels.csv.gz` | idem | labels communautaires (ex. « Motor neuron 9; MN9 » par P. Shiu, Scott lab) |
| `codex783/names, consolidated_cell_types, coordinates, …` | idem | noms, types cellulaires, positions |
| `Supplemental_file1_neuron_annotations.tsv` | dépôt `flyconnectome/flywire_annotations` (Schlegel *et al.*) | annotations systématiques, `top_nt` par neurone |
| `proofread_connections_783.feather` (optionnel, 850 Mo) | Zenodo 10676866, FlyWire Consortium | toutes les connexions proofread **sans seuil**, moyennes NT par paire |

Aucun code de simulation tiers n'est réutilisé : le réseau et le simulateur sont écrits ici
(numpy / scipy).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/download_data.sh                     # ~80 Mo (ajouter WITH_ZENODO_CONNECTIONS=1 pour le feather)
PYTHONPATH=. .venv/bin/python scripts/build_network.py          # matrice sparse signée 139k × 139k (5 s)
PYTHONPATH=. .venv/bin/python scripts/run_sugar_experiment.py   # 1 s de cerveau en ~5 s
PYTHONPATH=. .venv/bin/python scripts/run_controls.py
```

## Le modèle

`flywire_sim/network.py` : matrice `W[post, pre] = signe(pre) × syn_count`, format CSC, 2,7 M
d'entrées. Le signe vient du NT dominant du neurone présynaptique (GABA / glutamate → −1, sinon +1 ;
les ~20 k neurones sans NT dans Codex sont complétés par le `top_nt` de Schlegel *et al.*).

`flywire_sim/lif.py` : neurones **leaky integrate-and-fire** (le modèle biologiquement plausible le
plus simple), vectorisés sur tout le cerveau :

```
dv/dt = (g − (v − V_rest)) / τ_m        V_rest = V_reset = −52 mV, V_seuil = −45 mV, τ_m = 20 ms
dg/dt = −g / τ_syn                      τ_syn = 5 ms, délai 1,8 ms, réfractaire 2,2 ms
spike de j  →  g_i += W[i, j] × w_syn    w_syn = 0,275 mV (seul paramètre libre)
```

Ces constantes sont celles de l'électrophysiologie de la drosophile (publiées, réutilisées telles
quelles). Pas d'activité spontanée : le cerveau est silencieux tant qu'on ne stimule pas d'entrée
sensorielle. Un stimulus = forcer des neurones à émettre des spikes selon un processus de Poisson
(équivalent d'une activation optogénétique).

Coût : 139 255 neurones × 10 000 pas (dt = 0,1 ms) ≈ **4–5 s de calcul par seconde simulée** sur
un CPU (pas de GPU nécessaire).

## Premier résultat : le circuit « sucre → extension du proboscis »

On active à 100 Hz les 23 neurones gustatifs « sucre » du labellum gauche (labels officiels) et on
regarde ce qui s'allume dans le cerveau, notamment **MN9**, le motoneurone qui soulève le rostre
(extension du proboscis = la mouche « sort sa langue » pour manger) :

- 460 neurones actifs sur 139 255, presque tous dans le ganglion sous-œsophagien (GNG = centre
  gustatif/moteur de la tête) ;
- MN9 gauche ≈ 157 Hz, MN9 droit ≈ 218 Hz, et les autres motoneurones d'ingestion / proboscis
  (MN10, CB0700, MNx01…) sont activés ;
- contrôles : aucun stimulus → 0 spike ; 23 neurones centraux au hasard ou GRN « amer » → MN9 muet.

Autrement dit, la structure brute du connectome + un modèle de neurone minimal suffit à reproduire
une transformation sensorimotrice complète : goût sucré → commande motrice d'alimentation.

![raster](results/sugar_left_100Hz_seed0.png)

Limites à garder en tête : rates absolus non calibrés, pas de modulation (dopamine…), pas de
synapses électriques, pas d'états internes (faim), le côté contra > ipsi n'est pas attendu
biologiquement (à creuser : conventions gauche/droite de l'imagerie FAFB, qui est en miroir).

## Étape 2 (à discuter) : brancher un corps

Le cerveau FlyWire s'arrête au cou : **les motoneurones des pattes et des ailes sont dans le
ganglion ventral (VNC)**, qui n'est pas dans FAFB. Ce qui sort du cerveau vers le corps :
1 303 **neurones descendants** (DN) → VNC, et 110 motoneurones de la tête (proboscis, antennes,
cou, yeux). Ce qui rentre : ~17 k neurones sensoriels (vision 77 k neurones optiques, olfaction,
goût, mécanosensation, …).

Pistes concrètes pour le corps :
- Interface capteurs → cerveau : caméra virtuelle → photorécepteurs (R1-R8, 2 × ~800 colonnes) ;
  odeurs → ORN ; contact patte / labellum → GRN ; vent / son → JON (organe de Johnston).
- Interface cerveau → corps : lire les DN connus (ex. DNa02 = virage, DNp09 = avance,
  MDN = recul, DNge = tête) et les motoneurones de la tête, les mapper sur un contrôleur de
  locomotion du corps 3D (le VNC réel n'est pas simulé ; à terme, FlyWire publie aussi
  BANC/MANC = cerveau + VNC, ce qui permettrait de descendre jusqu'aux muscles des pattes).
- Moteur 3D : Python (ce simulateur) + moteur de jeu séparé (Godot ou Unity/Bevy) qui parle au
  cerveau via socket / mémoire partagée, ou tout en Rust (Bevy) avec le simulateur porté
  (matrice sparse + LIF sont ~200 lignes, portables facilement, et un backend GPU est possible).
  Le choix se fera à l'étape 2.
