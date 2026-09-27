#!/usr/bin/env bash
# Télécharge UNIQUEMENT les données officielles FlyWire (FAFB v783 puis BANC v888) :
#  - export statique Codex (Princeton, bucket public flywire-data) : neurones, connexions (>=5 synapses),
#    classification hiérarchique (Schlegel et al.), labels communautaires, coordonnées.
#  - dépôt Zenodo du FlyWire Consortium (Dorkenwald et al. 2024) : connexions proofread non seuillées.
#  - annotations systématiques (Schlegel et al. 2024, dépôt flyconnectome/flywire_annotations).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW/codex783"

CODEX="https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
for f in neurons.csv.gz connections.csv.gz classification.csv.gz labels.csv.gz names.csv.gz \
         consolidated_cell_types.csv.gz coordinates.csv.gz cell_stats.csv.gz processed_labels.csv.gz \
         neuropil_synapse_table.csv.gz connectivity_tags.csv.gz; do
  [ -s "$RAW/codex783/$f" ] || curl -sSL -o "$RAW/codex783/$f" "$CODEX/$f"
done

ZEN="https://zenodo.org/api/records/10676866/files"
[ -s "$RAW/proofread_root_ids_783.npy" ] || curl -sSL -o "$RAW/proofread_root_ids_783.npy" "$ZEN/proofread_root_ids_783.npy/content"
if [ "${WITH_ZENODO_CONNECTIONS:-0}" = "1" ]; then
  [ -s "$RAW/proofread_connections_783.feather" ] || curl -sSL -o "$RAW/proofread_connections_783.feather" "$ZEN/proofread_connections_783.feather/content"
fi

ANN="https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
[ -s "$RAW/Supplemental_file1_neuron_annotations.tsv" ] || curl -sSL -o "$RAW/Supplemental_file1_neuron_annotations.tsv" "$ANN"
# BANC v888 (cerveau + moelle, export statique Codex, même bucket public)
mkdir -p "$RAW/banc888"
BANC="https://storage.googleapis.com/flywire-data/codex/data/banc/888"
for f in neurons.csv.gz connections_princeton.csv.gz neuron_attributes.pickle.gz; do
  [ -s "$RAW/banc888/$f" ] || curl -sSL -o "$RAW/banc888/$f" "$BANC/$f"
done
echo "OK -> $RAW"
