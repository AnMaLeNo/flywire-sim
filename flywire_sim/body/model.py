"""Corps articulé de Drosophila melanogaster (femelle adulte) pour MuJoCo, sans ailes.

Unités : mm, g, s (=> force en µN, couple en µN·mm) ; gravité 9810 mm/s².

Géométrie : maillages micro-CT et origines de segments de NeuroMechFly (Lobato-Rios et al. 2022,
Apache-2.0, voir meshes/LICENSE-*). Tout le reste est défini ici : arbre cinématique, degrés de liberté
(7 par patte : thorax-coxa yaw/pitch/roll, coxa-trochantérofémur pitch/roll, fémur-tibia pitch,
tibia-tarse pitch ; tarsomères passifs), cou 3 DoF, trompe 2 DoF, abdomen 4 DoF, géométries de collision,
adhésion tarsale, actionneurs « muscle » nommés d'après l'atlas des muscles de patte du BANC/FANC
(Azevedo et al. 2024 ; Lesser et al. 2024), capteurs (angles, vitesses, contacts, charge).

Convention d'axes dans le repère de chaque segment de patte (patte tendue vers -z, x vers l'avant) :
yaw = rotation autour de x (abduction/adduction), pitch = autour de y (avant/arrière ou flexion/extension),
roll = autour de z (rotation propre du segment).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESHES = HERE / "meshes"
RIG = json.loads((HERE / "nmf_rigging.json").read_text())
RIGGING = RIG["rigging"]
NEUTRAL_DEG = RIG["neutral_pose_deg"]

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
LEG_NAME = {"f": "front_leg", "m": "middle_leg", "h": "hind_leg"}
SIDE = {"l": "left", "r": "right"}

# degrés de liberté articulaires actifs par segment de patte (axes dans l'ordre des charnières)
LEG_DOF = {"coxa": ["yaw", "pitch", "roll"], "trochanterfemur": ["pitch", "roll"], "tibia": ["pitch"], "tarsus1": ["pitch"]}
AXIS = {"yaw": "1 0 0", "pitch": "0 1 0", "roll": "0 0 1"}
# amplitudes (deg) autour de la pose neutre, d'après la cinématique de marche publiée (Mendes 2013, Lobato-Rios 2022)
RANGE_DEG = {("coxa", "yaw"): 40, ("coxa", "pitch"): 45, ("coxa", "roll"): 40,
             ("trochanterfemur", "pitch"): 60, ("trochanterfemur", "roll"): 25,
             ("tibia", "pitch"): 70, ("tarsus1", "pitch"): 45}
LEG_CHAIN = ["coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5"]
LEG_RADIUS = {"coxa": 0.06, "trochanterfemur": 0.055, "tibia": 0.04, "tarsus1": 0.028, "tarsus2": 0.025,
              "tarsus3": 0.022, "tarsus4": 0.02, "tarsus5": 0.02}

# Muscles de patte (atlas BANC/FANC `Primary Cell Type` des motoneurones) -> (articulation, axe, signe).
# Le signe donne le sens du couple pour +1 d'activation ; « ~side » : inversé pour les pattes droites.
# Couple max en µN·mm (ordre de grandeur : poids du corps 9.8 µN x bras de levier 0.5 mm / 3 pattes ~ 1.6 ;
# on donne 3-6 aux muscles principaux, 1-2 aux accessoires) — hypothèses à calibrer.
LEG_MUSCLES = {
    "sternal_anterior_rotator":     ("coxa", "pitch", -1, 3.0),
    "tergopleural_promotor":        ("coxa", "pitch", -1, 3.0),
    "sternal_posterior_rotator":    ("coxa", "pitch", +1, 3.0),
    "pleural_remotor_abductor":     ("coxa", "pitch", +1, 2.0),
    "sternal_adductor":             ("coxa", "yaw", "~side", 2.0),
    "trochanter_flexor":            ("trochanterfemur", "pitch", -1, 6.0),
    "accessory_trochanter_flexor":  ("trochanterfemur", "pitch", -1, 2.0),
    "trochanter_extensor":          ("trochanterfemur", "pitch", +1, 4.0),
    "sternotrochanter":             ("trochanterfemur", "pitch", +1, 3.0),
    "tergotrochanter":              ("trochanterfemur", "pitch", +1, 6.0),
    "TTMn":                         ("trochanterfemur", "pitch", +1, 6.0),
    "femur_reductor":               ("trochanterfemur", "roll", +1, 1.0),
    "tibia_flexor":                 ("tibia", "pitch", +1, 6.0),
    "accessory_tibia_flexor":       ("tibia", "pitch", +1, 2.0),
    "tibia_extensor":               ("tibia", "pitch", -1, 4.0),
    "tarsus_depressor":             ("tarsus1", "pitch", -1, 1.5),
    "tarsus_levator":               ("tarsus1", "pitch", +1, 1.0),
    "long_tendon_muscle":           ("tarsus1", "pitch", -1, 1.0),   # + adhésion (griffes/pulvilles), voir muscles.py
}


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _neutral(leg, seg_parent, seg, axis):
    key = f"{seg_parent}-{seg}-{axis}"
    side, pos = leg[0], leg[1]
    val = NEUTRAL_DEG.get(f"{'l' + pos}_{seg_parent}-{'l' + pos}_{seg}-{axis}".replace("lc_thorax", "c_thorax"), 0.0)
    if seg_parent == "c_thorax":
        val = NEUTRAL_DEG.get(f"c_thorax-l{pos}_{seg}-{axis}", 0.0)
    if side == "r" and axis in ("yaw", "roll"):
        val = -val
    return val, key


def _leg_xml(leg, stiffness, damping):
    prefix = f"{leg}_"
    out = []
    parent = "c_thorax"
    for i, seg in enumerate(LEG_CHAIN):
        name = prefix + seg
        rig = RIGGING[name]
        child = LEG_CHAIN[i + 1] if i + 1 < len(LEG_CHAIN) else None
        seg_len = -RIGGING[prefix + child]["pos"][2] if child else 0.08
        out.append(f'<body name="{name}" pos="{_fmt(rig["pos"])}" quat="{_fmt(rig["quat"])}">')
        if seg in LEG_DOF:
            for axis in LEG_DOF[seg]:
                n0, _ = _neutral(leg, parent if i else "c_thorax", seg, axis)
                rng = RANGE_DEG[(seg, axis)]
                out.append(f'<joint name="{name}_{axis}" axis="{AXIS[axis]}" range="{n0 - rng:.4g} {n0 + rng:.4g}" '
                           f'springref="{n0:.4g}" stiffness="{stiffness}" damping="{damping}"/>')
        else:  # tarsomères passifs (ressorts)
            n0 = -5.0
            out.append(f'<joint name="{name}_pitch" axis="0 1 0" range="{n0 - 40:.4g} {n0 + 40:.4g}" '
                       f'springref="{n0:.4g}" stiffness="{stiffness * 0.5}" damping="{damping}"/>')
        out.append(f'<geom class="visual" mesh="{name}"/>')
        r = LEG_RADIUS[seg]
        out.append(f'<geom class="collision" type="capsule" fromto="0 0 0 0 0 {-seg_len:.4g}" size="{r}" mass="{rig["mass"]:.4g}"/>')
        if seg == "tarsus5":
            out.append(f'<site name="{leg}_claw" pos="0 0 {-seg_len:.4g}" size="0.03"/>')
            out.append(f'<geom class="adhesion" type="sphere" pos="0 0 {-seg_len:.4g}" size="0.03" mass="1e-8"/>')
        if seg.startswith("tarsus"):
            out.append(f'<site name="{name}_touch" type="capsule" fromto="0 0 0 0 0 {-seg_len:.4g}" size="{r + 0.005}"/>')
        parent = seg
    out.append("</body>" * len(LEG_CHAIN))
    return "\n".join(out)


def build_mjcf(stiffness: float = 20.0, damping: float = 0.05, adhesion_gain: float = 20.0) -> str:
    # les segments droits sont le miroir (y -> -y) des maillages gauches du scan
    meshes = "\n".join(f'<mesh name="{n}" file="{"l" + n[1:] if n[0] == "r" else n}.stl" '
                       f'scale="1000 {-1000 if n[0] == "r" else 1000} 1000"/>'
                       for n in RIGGING if "wing" not in n)
    thorax = RIGGING["c_thorax"]
    head = RIGGING["c_head"]
    legs = "\n".join(_leg_xml(leg, stiffness, damping) for leg in LEGS)

    def simple(name, extra=""):
        r = RIGGING[name]
        return f'<body name="{name}" pos="{_fmt(r["pos"])}" quat="{_fmt(r["quat"])}">{extra}<geom class="visual" mesh="{name}"/>'

    abd = ""
    for seg in ["c_abdomen12", "c_abdomen3", "c_abdomen4", "c_abdomen5", "c_abdomen6"]:
        r = RIGGING[seg]
        abd += (f'<body name="{seg}" pos="{_fmt(r["pos"])}" quat="{_fmt(r["quat"])}">'
                f'<joint name="{seg}_pitch" axis="0 1 0" range="-25 25" stiffness="{stiffness*3}" damping="{damping*3}"/>'
                f'<geom class="visual" mesh="{seg}"/>'
                f'<geom class="collision" type="ellipsoid" size="0.16 0.3 0.28" pos="-0.1 0 0" mass="{r["mass"]:.4g}"/>')
    abd += "</body>" * 5

    actuators = []
    for leg in LEGS:
        for muscle, (seg, axis, sign, tmax) in LEG_MUSCLES.items():
            if muscle == "tergopleural_promotor" and leg[1] != "f":
                continue  # muscle propre à la patte avant (BANC)
            if muscle == "TTMn" and leg[1] != "m":
                continue  # muscle tergotrochantéral du saut : patte du milieu
            s = sign if sign != "~side" else (-1 if leg[0] == "l" else 1)
            actuators.append(f'<motor name="{leg}_{muscle}" joint="{leg}_{seg}_{axis}" gear="{s * tmax:.4g}" ctrlrange="0 1"/>')
        actuators.append(f'<adhesion name="{leg}_adhesion" body="{leg}_tarsus5" ctrlrange="0 1" gain="{adhesion_gain}"/>')

    sensors = []
    for leg in LEGS:
        for seg, axes in LEG_DOF.items():
            for axis in axes:
                sensors.append(f'<jointpos name="{leg}_{seg}_{axis}_pos" joint="{leg}_{seg}_{axis}"/>'
                               f'<jointvel name="{leg}_{seg}_{axis}_vel" joint="{leg}_{seg}_{axis}"/>')
        for k in range(1, 6):
            sensors.append(f'<touch name="{leg}_tarsus{k}_contact" site="{leg}_tarsus{k}_touch"/>')
        sensors.append(f'<force name="{leg}_load" site="{leg}_claw"/>')
    sensors.append('<framequat name="thorax_quat" objtype="body" objname="c_thorax"/>'
                   '<gyro name="thorax_gyro" site="thorax_site"/><accelerometer name="thorax_acc" site="thorax_site"/>')

    return f"""<mujoco model="drosophila_melanogaster_female">
<compiler angle="degree" meshdir="{MESHES}" autolimits="true" boundmass="1e-7" boundinertia="1e-11"/>
<option timestep="0.0001" gravity="0 0 -9810" integrator="implicitfast" cone="elliptic" noslip_iterations="3"/>
<statistic extent="4" center="0 0 1"/>
<visual><map force="0.1" zfar="200" znear="0.001"/><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  <global offwidth="1280" offheight="720"/></visual>
<default>
  <joint type="hinge" armature="1e-7"/>
  <geom solref="0.0005 1" solimp="0.95 0.99 0.001" friction="1.0 0.01 0.001"/>
  <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.55 0.4 0.25 1" mass="0"/></default>
  <default class="collision"><geom group="3" rgba="0.2 0.6 0.9 0.3"/></default>
  <default class="adhesion"><geom group="3" rgba="0.9 0.3 0.3 0.5" margin="0.01" gap="0.01"/></default>
</default>
<asset>
  <texture type="skybox" builtin="gradient" rgb1="0.6 0.75 0.95" rgb2="0.95 0.95 1" width="64" height="64"/>
  <texture name="sol" type="2d" builtin="checker" rgb1="0.85 0.85 0.8" rgb2="0.7 0.7 0.65" width="256" height="256"/>
  <material name="sol" texture="sol" texrepeat="40 40" reflectance="0.05"/>
  {meshes}
</asset>
<worldbody>
  <light pos="0 0 20" dir="0 0 -1" directional="true"/>
  <geom name="floor" type="plane" size="50 50 0.1" material="sol"/>
  <camera name="side" mode="targetbody" target="c_thorax" pos="0 -5 1.8"/>
  <camera name="front" mode="targetbody" target="c_thorax" pos="5 1.5 1.5"/>
  <camera name="top" mode="targetbody" target="c_thorax" pos="0.5 0 6"/>
  <body name="c_thorax" pos="{_fmt(thorax["pos"])}">
    <freejoint name="root"/>
    <site name="thorax_site" size="0.02"/>
    <geom class="visual" mesh="c_thorax"/>
    <geom class="collision" type="ellipsoid" size="0.55 0.4 0.4" pos="-0.15 0 -0.05" mass="{thorax["mass"]:.4g}"/>
    <body name="c_head" pos="{_fmt(head["pos"])}">
      <joint name="neck_yaw" axis="0 0 1" range="-40 40" stiffness="{stiffness*2}" damping="{damping*2}"/>
      <joint name="neck_pitch" axis="0 1 0" range="-40 40" stiffness="{stiffness*2}" damping="{damping*2}"/>
      <joint name="neck_roll" axis="1 0 0" range="-30 30" stiffness="{stiffness*2}" damping="{damping*2}"/>
      <geom class="visual" mesh="c_head"/>
      <geom class="collision" type="ellipsoid" size="0.3 0.35 0.35" pos="0.25 0 0" mass="{head["mass"]:.4g}"/>
      {simple("l_eye")}</body>{simple("r_eye")}</body>
      {simple("l_pedicel")}{simple("l_funiculus")}{simple("l_arista")}</body></body></body>
      {simple("r_pedicel")}{simple("r_funiculus")}{simple("r_arista")}</body></body></body>
      {simple("c_rostrum", f'<joint name="rostrum_pitch" axis="0 1 0" range="-10 50" stiffness="{stiffness}" damping="{damping}"/>')}
        {simple("c_haustellum", f'<joint name="haustellum_pitch" axis="0 1 0" range="-40 40" stiffness="{stiffness}" damping="{damping}"/>')}</body></body>
    </body>
    {simple("l_haltere")}</body>{simple("r_haltere")}</body>
    {abd}
    {legs}
  </body>
</worldbody>
<contact>
{chr(10).join(f'<exclude body1="c_thorax" body2="{leg}_trochanterfemur"/>' for leg in LEGS)}
</contact>
<actuator>
{chr(10).join(actuators)}
</actuator>
<sensor>
{chr(10).join(sensors)}
</sensor>
</mujoco>"""


def save_mjcf(path: Path) -> Path:
    path.write_text(build_mjcf())
    return path
