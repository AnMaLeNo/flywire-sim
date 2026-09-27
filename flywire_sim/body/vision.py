"""Œil composé : rendu MuJoCo par œil -> photorécepteurs -> neurones visuels du BANC.

Approximations documentées (voir docs/capteurs.md) :
- Optique (Götz 1964) : angle interommatidial Δφ ≈ 4,6°, champ d'une ommatidie Δρ ≈ 3,5-5° (gaussien).
  Chaque colonne échantillonne une cubemap rendue depuis la surface de l'œil (5 faces à 90°).
- Rétinotopie : la direction de vue de chaque neurone colonnaire du BANC (R7, R8, L1-L5, Mi1) est
  déduite de la position officielle de son soma/arbre dans le lobe optique (`neuron_attributes`) :
  sphère ajustée par côté, rayon vecteur = direction de vue, axe optique de l'œil à 60° d'azimut.
  Hypothèse (à tester) : la position dans la médulla est inversée avant/arrière par le chiasma.
- R1-R6 (Rh1, large bande 478 nm + UV) : potentiel gradué log(I) adapté (Juusola & Hardie 2001 :
  temps au pic ~20 ms à forte lumière, ~40 ms à faible), deux filtres passe-bas en cascade.
  Ils n'existent pas dans le BANC (hors volume EM) : leur sortie histaminergique est injectée en
  courant dans L1, L2 (contraste, transitoire, dépolarisés par l'assombrissement) et L3 (luminance,
  soutenu, dépolarisé dans le noir) — Clark 2011, Ketkar 2022, Strother 2014.
- R7/R8 (dans le BANC) : forcés en Poisson selon leur canal spectral ; ommatidies pale/yellow
  tirées au sort 30/70 (Rh3/Rh4 UV pour R7 ≈ canal bleu du rendu ; Rh5 bleu / Rh6 vert-rouge pour R8,
  Sharkey 2020). Pas de rendu UV : le bleu sert de substitut, la région dorsale (polarisation) n'est
  pas modélisée. Les ocelles ne sont pas dans le BANC.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

COLUMNAR = ("R7", "R8", "L1", "L2", "L3", "L4", "L5", "Mi1")
SHELL_TYPES = ("R7", "R8", "L1", "L2", "L5")   # types denses (~1 par colonne) servant à ajuster la coquille
SHELL_RADIUS_UM = 100.0                          # rayon de la coquille des colonnes (ajustement œil droit)
EYE_AXIS_AZ_DEG = 60.0      # azimut de l'axe optique de chaque œil par rapport à l'avant (± selon côté)
FACES = ("front", "lateral", "back", "up", "down")
# quaternions (w x y z) des caméras MuJoCo : la caméra regarde vers -z local, +y local = haut
# face « front » regarde vers +x du corps ; « lateral » vers ±y ; « back » -x ; « up » +z ; « down » -z


def _face_quat(face: str, side: str) -> np.ndarray:
    lat = 1.0 if side == "l" else -1.0
    if face == "front":
        return _quat_look(np.array([1, 0, 0]), np.array([0, 0, 1]))
    if face == "lateral":
        return _quat_look(np.array([0, lat, 0]), np.array([0, 0, 1]))
    if face == "back":
        return _quat_look(np.array([-1, 0, 0]), np.array([0, 0, 1]))
    if face == "up":
        return _quat_look(np.array([0, 0, 1]), np.array([-1, 0, 0]))
    return _quat_look(np.array([0, 0, -1]), np.array([1, 0, 0]))


def _quat_look(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Quaternion (w x y z) tel que -z local = forward et +y local = up."""
    z = -forward / np.linalg.norm(forward)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def eye_cameras_xml(side: str, offset: np.ndarray, fovy: float = 90.0) -> str:
    return "".join(
        f'<camera name="{side}_eye_{f}" pos="{offset[0]:.4g} {offset[1]:.4g} {offset[2]:.4g}" '
        f'quat="{" ".join(f"{v:.6g}" for v in _face_quat(f, side))}" fovy="{fovy}"/>'
        for f in FACES)


@dataclass
class Retina:
    side: str
    idx: np.ndarray          # indices réseau des neurones colonnaires
    types: np.ndarray        # type de chaque neurone
    dirs: np.ndarray         # direction de vue unitaire, repère tête (x avant, y gauche, z haut)


def sphere_fit(P: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + c @ c))


def robust_sphere_center(P: np.ndarray, radius: float = SHELL_RADIUS_UM, iters: int = 5) -> np.ndarray:
    """Centre d'une sphère de rayon fixé ajusté aux points, en écartant les points hors coquille
    (positions de neurones non colonnaires ou mal placées ; lobe optique gauche moins complet)."""
    c, _ = sphere_fit(P)
    keep = np.ones(len(P), dtype=bool)
    for _ in range(iters):
        c = least_squares(lambda cc, k=keep: np.linalg.norm(P[k] - cc, axis=1) - radius, c).x
        d = np.abs(np.linalg.norm(P - c, axis=1) - radius)
        keep = d < max(15.0, 2.0 * float(np.median(d)))
    return c


def build_retinas(neurons: pd.DataFrame, positions: pd.DataFrame, ap_sign: float = 1.0) -> list[Retina]:
    """Direction de vue de chaque neurone colonnaire à partir de sa position BANC (µm ; repère volume :
    x = droite -> gauche de la mouche, y = antérieur -> postérieur de l'axe neural, z = ventral -> dorsal).
    La coquille des colonnes (R7/R8/L1/L2/L5) couvre ~±80° autour de sa normale moyenne, ce que l'on
    identifie au champ de l'œil (~160°) : rayon vecteur = direction de vue, normale moyenne = axe optique."""
    n = neurons.reset_index(drop=True)
    pos = positions.set_index("root_id")
    out = []
    for side in ("l", "r"):
        on_side = (n.side == {"l": "left", "r": "right"}[side]) & n.root_id.isin(pos.index)
        sel = n.cell_type.isin(COLUMNAR) & on_side
        idx = np.flatnonzero(sel.to_numpy())
        if idx.size < 50:
            continue
        shell = np.flatnonzero((n.cell_type.isin(SHELL_TYPES) & on_side).to_numpy())
        c = robust_sphere_center(pos.loc[n.root_id.iloc[shell], ["x", "y", "z"]].to_numpy())
        v = pos.loc[n.root_id.iloc[idx], ["x", "y", "z"]].to_numpy() - c
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        lat = 1.0 if side == "l" else -1.0
        # repère œil : m = axe optique (normale moyenne de la coquille), d = dorsal, a = avant
        m = v.mean(axis=0)
        m /= np.linalg.norm(m)
        d = np.array([0.0, 0.0, 1.0])
        d -= (d @ m) * m
        d /= np.linalg.norm(d)
        a = np.cross(d, m) * lat
        if a[1] > 0:            # y du volume croît vers l'arrière : `a` doit pointer vers -y
            a = -a
        a *= ap_sign            # chiasma externe : la médulla est inversée avant/arrière
        e_lat, e_dor, e_ant = v @ m, v @ d, v @ a
        # axe optique à ±EYE_AXIS_AZ_DEG d'azimut dans le repère tête (x avant, y gauche, z haut)
        th = np.deg2rad(EYE_AXIS_AZ_DEG)
        x = e_lat * np.cos(th) + e_ant * np.sin(th)
        y = lat * (e_lat * np.cos(th) - e_ant * np.sin(th))
        z = e_dor
        dirs = np.stack([x, y, z], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        out.append(Retina(side, idx, n.cell_type.to_numpy()[idx], dirs))
    return out


class Eyes:
    """Rendu cubemap par œil et modèle de photorécepteurs. `update(data)` à appeler toutes les
    `period_ms` ; `currents()` et `spikes(dt_ms, rng)` fournissent l'entrée au réseau à chaque pas."""

    def __init__(self, model: mujoco.MjModel, retinas: list[Retina], period_ms: float = 5.0,
                 size: int = 32, rho_deg: float = 4.5, r_max: float = 150.0, gain_mv: float = 12.0,
                 seed: int = 0):
        self.model, self.retinas, self.period_ms, self.size = model, retinas, period_ms, size
        self.r_max, self.gain_mv = r_max, gain_mv
        self.renderer = mujoco.Renderer(model, height=size, width=size)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                     mujoco.mjtRndFlag.mjRND_SKYBOX):
            self.renderer.scene.flags[flag] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[0] = self.opt.geomgroup[1] = self.opt.geomgroup[2] = 1
        self.cam = {(r.side, f): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{r.side}_eye_{f}")
                    for r in retinas for f in FACES}
        self.head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "c_head")
        rng = np.random.default_rng(seed)
        # échantillons gaussiens du champ Δρ (centre + 6 points à ~sigma)
        sig = np.deg2rad(rho_deg) / 2.355
        ang = np.linspace(0, 2 * np.pi, 7)[:-1]
        self.offsets = np.c_[[0.0, *(sig * np.cos(ang))], [0.0, *(sig * np.sin(ang))]]
        # état par neurone colonnaire : luminance R1-6 (log), filtres, adaptation
        self.n_cols = sum(r.idx.size for r in retinas)
        self.pale = rng.random(self.n_cols) < 0.3
        self.lum = np.zeros(self.n_cols)        # log-luminance instantanée (canal Rh1)
        self.p1 = np.zeros(self.n_cols)         # filtre passe-bas 1 (photorécepteur)
        self.p2 = np.zeros(self.n_cols)         # filtre passe-bas 2 -> potentiel R1-6
        self.adapt = np.zeros(self.n_cols)      # adaptation lente (niveau moyen)
        self.slow = np.zeros(self.n_cols)       # référence lente pour le contraste (L1/L2)
        self.rgb = np.zeros((self.n_cols, 3))
        self.tau_p, self.tau_adapt, self.tau_slow = 8.0, 500.0, 150.0
        self._init = False
        self.types = np.concatenate([r.types for r in retinas])
        self.idx = np.concatenate([r.idx for r in retinas])
        self.dirs = np.concatenate([r.dirs for r in retinas])
        self.side = np.concatenate([np.full(r.idx.size, r.side) for r in retinas])
        self.is_l12 = np.isin(self.types, ["L1", "L2"])
        self.is_l3 = self.types == "L3"
        self.is_r7 = self.types == "R7"
        self.is_r8 = self.types == "R8"
        self.images = {}

    # ---- rendu -------------------------------------------------------------------------------
    def render(self, data: mujoco.MjData) -> None:
        for (side, face), cid in self.cam.items():
            self.renderer.update_scene(data, camera=cid, scene_option=self.opt)
            self.images[(side, face)] = self.renderer.render().astype(np.float32) / 255.0

    def sample(self, data: mujoco.MjData) -> np.ndarray:
        """Couleur RGB moyenne (champ gaussien) vue par chaque colonne."""
        out = np.zeros((self.n_cols, 3))
        for side in ("l", "r"):
            sel = self.side == side
            if not sel.any():
                continue
            d = self.dirs[sel]
            for dx, dy in self.offsets:
                # petites rotations du regard autour de la direction de la colonne
                u = np.cross(d, np.array([0, 0, 1.0]))
                u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
                w = np.cross(d, u)
                dd = d + dx * u + dy * w
                dd /= np.linalg.norm(dd, axis=1, keepdims=True)
                out[sel] += self._lookup(side, dd)
        return out / len(self.offsets)

    def _lookup(self, side: str, d: np.ndarray) -> np.ndarray:
        """Couleur de la cubemap de l'œil `side` dans les directions `d` (repère tête)."""
        ax, ay, az = np.abs(d[:, 0]), np.abs(d[:, 1]), np.abs(d[:, 2])
        lat = 1.0 if side == "l" else -1.0
        rgb = np.zeros((len(d), 3))
        s = self.size
        # face avant / arrière (x dominant)
        for face, cond, u, v in (
            ("front", (ax >= ay) & (ax >= az) & (d[:, 0] > 0), -d[:, 1] / np.maximum(ax, 1e-9), d[:, 2] / np.maximum(ax, 1e-9)),
            ("back", (ax >= ay) & (ax >= az) & (d[:, 0] <= 0), d[:, 1] / np.maximum(ax, 1e-9), d[:, 2] / np.maximum(ax, 1e-9)),
            ("lateral", (ay > ax) & (ay >= az) & (lat * d[:, 1] > 0), lat * d[:, 0] / np.maximum(ay, 1e-9), d[:, 2] / np.maximum(ay, 1e-9)),
            ("up", (az > ax) & (az > ay) & (d[:, 2] > 0), -d[:, 1] / np.maximum(az, 1e-9), -d[:, 0] / np.maximum(az, 1e-9)),
            ("down", (az > ax) & (az > ay) & (d[:, 2] <= 0), -d[:, 1] / np.maximum(az, 1e-9), d[:, 0] / np.maximum(az, 1e-9)),
        ):
            img = self.images.get((side, face))
            if img is None or not cond.any():
                continue
            col = np.clip(((u[cond] + 1) * 0.5 * s).astype(int), 0, s - 1)
            row = np.clip(((1 - v[cond]) * 0.5 * s).astype(int), 0, s - 1)
            rgb[cond] = img[row, col]
        return rgb

    # ---- photorécepteurs ---------------------------------------------------------------------
    def update(self, data: mujoco.MjData) -> None:
        """Rendu + intégration des photorécepteurs sur `period_ms`."""
        self.render(data)
        self.rgb = self.sample(data)
        rh1 = 0.1 * self.rgb[:, 0] + 0.5 * self.rgb[:, 1] + 0.4 * self.rgb[:, 2]
        self.lum = np.log(rh1 + 0.02)
        dt = self.period_ms
        if not self._init:
            self.p1[:] = self.p2[:] = self.adapt[:] = self.slow[:] = self.lum
            self._init = True
        a = np.exp(-dt / self.tau_p)
        self.p1 += (1 - a) * (self.lum - self.p1)
        self.p2 += (1 - a) * (self.p1 - self.p2)
        self.adapt += (1 - np.exp(-dt / self.tau_adapt)) * (self.p2 - self.adapt)
        self.slow += (1 - np.exp(-dt / self.tau_slow)) * (self.p2 - self.slow)

    @property
    def photoreceptor_mv(self) -> np.ndarray:
        """Potentiel R1-6 adapté (unités log : +1 = luminance x e)."""
        return self.p2 - self.adapt

    def currents(self) -> list[tuple[np.ndarray, float]]:
        """Entrées (indices, mV) pour les LMC : L1/L2 dépolarisés par l'assombrissement (contraste),
        L3 par la luminance faible (soutenu)."""
        out = []
        contrast = np.clip(self.slow - self.p2, 0, None)           # >0 quand ça s'assombrit
        dark = np.clip(self.adapt - self.p2 + 0.3, 0, None)        # niveau soutenu sous la moyenne
        for mask, val in ((self.is_l12, contrast), (self.is_l3, dark * 0.5)):
            if mask.any():
                for i in np.flatnonzero(mask & (val > 0.02)):
                    out.append((self.idx[i:i + 1], self.gain_mv * float(val[i])))
        return out

    def rates(self) -> np.ndarray:
        """Taux (Hz) des R7/R8 selon leur canal spectral et l'adaptation (zéro pour les autres)."""
        r = np.zeros(self.n_cols)
        uv = self.rgb[:, 2]
        blue = self.rgb[:, 2]
        green_red = 0.5 * self.rgb[:, 1] + 0.5 * self.rgb[:, 0]
        r[self.is_r7] = uv[self.is_r7]
        r[self.is_r8 & self.pale] = blue[self.is_r8 & self.pale]
        r[self.is_r8 & ~self.pale] = green_red[self.is_r8 & ~self.pale]
        # adaptation : gain réduit quand la scène est globalement lumineuse (Weber)
        return self.r_max * r / (0.3 + np.exp(self.adapt) - 0.02)

    def spikes(self, dt_ms: float, rng: np.random.Generator) -> np.ndarray:
        p = self.rates() * dt_ms / 1000.0
        hit = (rng.random(self.n_cols) < p) & (self.is_r7 | self.is_r8)
        return self.idx[hit]
