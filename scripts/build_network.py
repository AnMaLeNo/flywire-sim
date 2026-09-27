"""Construit et sauvegarde la matrice de connectivité signée du cerveau FlyWire v783."""
import time

import numpy as np

from flywire_sim import data, network

t0 = time.time()
net = network.build(min_synapses=5)
out = data.PROCESSED / "network_783.npz"
net.save(out)
W = net.W
print(f"neurones : {net.n:,}")
print(f"connexions (paires pre->post, >=5 synapses) : {W.nnz:,}")
print(f"synapses totales dans les connexions : {int(np.abs(W.data).sum()):,}")
print(f"neurones inhibiteurs (GABA/GLUT) : {(net.sign < 0).sum():,}  excitateurs : {(net.sign > 0).sum():,}")
vals, cnt = np.unique(net.nt_type, return_counts=True)
print("NT dominants :", dict(zip(vals, cnt)))
print(f"sauvé -> {out}  ({time.time()-t0:.1f}s)")
