"""Champs d'environnement lus par les capteurs non mécaniques (senses.py).

Tout ce qui n'est pas simulé par MuJoCo (odeurs, goût, température, humidité, vent, son) est décrit ici
par des sources ponctuelles et des valeurs ambiantes ; les capteurs interrogent `Environment` à la
position de l'organe concerné (antenne, labelle, tarses). Unités : mm, °C, humidité relative [0, 1],
vent en mm/s, son en déplacement normalisé de l'ariste [0, 1].
"""
from dataclasses import dataclass, field

import numpy as np

# classes d'odeurs = mots-clés du champ `Function` des neurones récepteurs olfactifs du BANC
ODOR_CLASSES = ("yeasty", "alcoholic_fermentation", "decaying_fruit", "fruity", "plant_matter",
                "animal_matter", "aversive", "carbon_dioxide", "pheromone")
# modalités gustatives = mots-clés du champ `Function` des neurones gustatifs du BANC
TASTE_CLASSES = ("sugar", "bitter", "water", "low_salt", "fatty_acids", "amino_acids", "contact_pheromone")


@dataclass
class OdorSource:
    pos: np.ndarray                 # mm
    odors: dict                     # classe -> concentration normalisée à la source (0-1)
    radius: float = 5.0             # mm : décroissance gaussienne (panache diffusif simplifié, sans vent)

    def concentration(self, p: np.ndarray) -> dict:
        w = float(np.exp(-0.5 * np.sum((p - self.pos) ** 2) / self.radius ** 2))
        return {k: v * w for k, v in self.odors.items()}


@dataclass
class TastePatch:
    pos: np.ndarray                 # mm, sur le sol
    tastes: dict                    # modalité -> intensité normalisée (0-1)
    radius: float = 1.0             # mm : tache uniforme

    def taste(self, p: np.ndarray) -> dict:
        return dict(self.tastes) if np.linalg.norm(p[:2] - self.pos[:2]) <= self.radius else {}


@dataclass
class Environment:
    odor_sources: list = field(default_factory=list)
    taste_patches: list = field(default_factory=list)
    temperature_c: float = 25.0     # ambiante (Drosophila préfère ~24-25 °C)
    hot_spot: tuple | None = None   # (pos, delta_c, radius) : source de chaleur gaussienne
    humidity: float = 0.5           # humidité relative ambiante
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))   # mm/s, repère monde
    sound: float = 0.0              # amplitude normalisée de vibration de l'ariste (chant, 0-1)
    sound_hz: float = 200.0         # fréquence dominante (pulse song ~200 Hz)
    hypoxia: float = 0.0            # 0 = air normal

    def odors_at(self, p: np.ndarray) -> dict:
        out = {k: 0.0 for k in ODOR_CLASSES}
        for s in self.odor_sources:
            for k, v in s.concentration(p).items():
                out[k] = max(out[k], v)
        return out

    def tastes_at(self, p: np.ndarray) -> dict:
        out = {k: 0.0 for k in TASTE_CLASSES}
        for t in self.taste_patches:
            for k, v in t.taste(p).items():
                out[k] = max(out[k], v)
        return out

    def temperature_at(self, p: np.ndarray) -> float:
        t = self.temperature_c
        if self.hot_spot is not None:
            pos, delta, radius = self.hot_spot
            t += delta * float(np.exp(-0.5 * np.sum((p - np.asarray(pos)) ** 2) / radius ** 2))
        return t
