"""Contrôles : (a) aucun stimulus -> le cerveau doit rester silencieux (pas d'activité spontanée dans
ce modèle) ; (b) 23 neurones tirés au hasard à 100 Hz -> MN9 ne doit pas s'activer ; (c) GRN amers
(bitter) à 100 Hz -> MN9 ne doit pas (ou peu) s'activer ; (d) GRN sucre gauche -> MN9 actif."""
import numpy as np

from flywire_sim import data
from flywire_sim.lif import LIFNetwork, LIFParams, Stimulus
from flywire_sim.network import Network

net = Network.load(data.PROCESSED / "network_783.npz")
mn9 = {s: int(net.index_of([rid])[0]) for s, rid in data.MN9.items()}
sugar = net.index_of(data.root_ids_with_label(r"Sugar Gustatory Receptor Neuron", side="left"))
bitter = net.index_of(data.root_ids_with_label(r"Bitter Gustatory Receptor Neuron", side="left"))
rng = np.random.default_rng(1)
cls = data.load_classification().set_index("root_id").reindex(net.root_ids)
central = np.flatnonzero((cls.super_class == "central").to_numpy())

cases = {
    "aucun stimulus": [],
    f"{len(sugar)} neurones centraux aléatoires @100Hz": [Stimulus(rng.choice(central, len(sugar), replace=False), 100)],
    f"GRN amers gauche ({len(bitter)}) @100Hz": [Stimulus(bitter, 100)],
    f"GRN sucre gauche ({len(sugar)}) @100Hz": [Stimulus(sugar, 100)],
    f"GRN sucre gauche ({len(sugar)}) @50Hz": [Stimulus(sugar, 50)],
    f"GRN sucre gauche ({len(sugar)}) @25Hz": [Stimulus(sugar, 25)],
}
print(f"{'cas':50s} {'spikes':>8s} {'actifs':>7s} {'MN9 G':>7s} {'MN9 D':>7s}")
for name, stim in cases.items():
    res = LIFNetwork(net.W, LIFParams(), seed=0).run(1000.0, stim)
    r = res.rates_hz()
    print(f"{name:50s} {len(res.spike_times):8d} {int((r > 0).sum()):7d} {r[mn9['left']]:7.1f} {r[mn9['right']]:7.1f}")
