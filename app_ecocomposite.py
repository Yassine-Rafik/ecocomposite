import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="EcoComp-Opt | Optimisation Métaheuristique d'Éco-Composites",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. BASE DE DONNÉES MATÉRIAUX (Articles 49, 50 & 56 — Post-cuisson 80°C)
# ==============================================================================
MATERIALS = {
    0: {
        "name": "Jute UD / Époxy", "short": "Jute", "code": "J", "abaqus": "Jute_UD",
        "E1": 500.0, "E2": 300.0, "G12": 100.0, "nu12": 0.30,
        "XT": 50.0, "XC": 40.0, "YT": 15.0, "YC": 10.0, "SL": 5.0, "ST": 2.0,
        "t": 0.90, "areal_w": 233.0, "color": "#C4704F"
    },
    1: {
        "name": "Verre UD / Époxy", "short": "Verre", "code": "V", "abaqus": "Glass_UD",
        "E1": 12000.0, "E2": 4500.0, "G12": 2500.0, "nu12": 0.3045,
        "XT": 300.0, "XC": 150.0, "YT": 38.0, "YC": 50.0, "SL": 82.0, "ST": 82.0,
        "t": 0.22, "areal_w": 216.0, "color": "#3B6E8C"
    }
}

SYM_MAP = {
    0: [[1, 1, 1, 1, 1]],
    1: [[1, 1, 0, 1, 1]],
    2: [[0, 1, 1, 1, 0], [1, 0, 1, 0, 1]],
    3: [[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],
    4: [[0, 0, 1, 0, 0]],
    5: [[0, 0, 0, 0, 0]],
}


# ==============================================================================
# 2. MOTEUR MÉCANIQUE : CLT COMPLÈTE [A, B; B, D] + HASHIN 2D
# ==============================================================================
def get_Q_local(mat):
    E1, E2, G12, nu12 = mat["E1"], mat["E2"], mat["G12"], mat["nu12"]
    denom = 1.0 - (nu12 ** 2) * (E2 / E1)
    return np.array([
        [E1 / denom,        nu12 * E2 / denom, 0.0],
        [nu12 * E2 / denom, E2 / denom,        0.0],
        [0.0,               0.0,               G12]
    ])


def get_T_eps(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [c * c,        s * s,        c * s],
        [s * s,        c * c,       -c * s],
        [-2.0 * c * s, 2.0 * c * s,  c * c - s * s]
    ])


def hashin_safety_factor(sig1, sig2, tau12, mat):
    XT, XC = mat["XT"], mat["XC"]
    YT, YC = mat["YT"], mat["YC"]
    SL, ST = mat["SL"], mat["ST"]
    candidates = []

    if sig1 > 1e-12:
        val = (sig1 / XT) ** 2 + (tau12 / SL) ** 2
        if val > 0:
            candidates.append((1.0 / np.sqrt(val), "Fibre (Traction)"))
    elif sig1 < -1e-12:
        candidates.append((XC / abs(sig1), "Fibre (Compression)"))

    if sig2 > 1e-12:
        val = (sig2 / YT) ** 2 + (tau12 / SL) ** 2
        if val > 0:
            candidates.append((1.0 / np.sqrt(val), "Matrice (Traction)"))
    elif sig2 < -1e-12:
        a = (sig2 / (2.0 * ST)) ** 2 + (tau12 / SL) ** 2
        b = ((YC / (2.0 * ST)) ** 2 - 1.0) * (sig2 / YC)
        if a > 1e-18:
            lam = (-b + np.sqrt(b * b + 4.0 * a)) / (2.0 * a)
            if lam > 0:
                candidates.append((lam, "Matrice (Compression)"))
        elif b > 1e-18:
            candidates.append((1.0 / b, "Matrice (Compression)"))

    if not candidates and abs(tau12) > 1e-12:
        candidates.append((SL / abs(tau12), "Cisaillement"))

    return min(candidates, key=lambda x: x[0]) if candidates else (1e9, "Aucun")


def evaluate_laminate(mat_ids, angles_deg, detailed=False):
    n_plies = len(angles_deg)
    t_list = [MATERIALS[m]["t"] for m in mat_ids]
    H = sum(t_list)

    z = np.zeros(n_plies + 1)
    z[0] = -H / 2.0
    for i in range(n_plies):
        z[i + 1] = z[i] + t_list[i]

    A, B, D = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    Q_locals, T_eps_list = [], []

    for i in range(n_plies):
        Q = get_Q_local(MATERIALS[mat_ids[i]])
        T_e = get_T_eps(angles_deg[i])
        Q_bar = T_e.T @ Q @ T_e
        Q_locals.append(Q)
        T_eps_list.append(T_e)

        A += Q_bar * (z[i + 1] - z[i])
        B += 0.5 * Q_bar * (z[i + 1] ** 2 - z[i] ** 2)
        D += (1.0 / 3.0) * Q_bar * (z[i + 1] ** 3 - z[i] ** 3)

    ABD = np.block([[A, B], [B, D]])
    resp_x = np.linalg.solve(ABD, np.array([H, 0.0, 0.0, 0.0, 0.0, 0.0]))
    resp_y = np.linalg.solve(ABD, np.array([0.0, H, 0.0, 0.0, 0.0, 0.0]))
    resp_s = np.linalg.solve(ABD, np.array([0.0, 0.0, H, 0.0, 0.0, 0.0]))

    eps0, kappa = resp_x[:3], resp_x[3:]
    sigma_fpf, critical_ply, critical_mode = 1e9, -1, ""
    ply_details = []

    for i in range(n_plies):
        mat = MATERIALS[mat_ids[i]]
        ply_lam, ply_mode = 1e9, "Aucun"
        for z_pos in (z[i], z[i + 1]):
            sig_local = Q_locals[i] @ (T_eps_list[i] @ (eps0 + z_pos * kappa))
            lam, mode = hashin_safety_factor(sig_local[0], sig_local[1], sig_local[2], mat)
            if lam < ply_lam:
                ply_lam, ply_mode = lam, mode
            if lam < sigma_fpf:
                sigma_fpf, critical_ply, critical_mode = lam, i + 1, mode

        if detailed:
            z_mid = 0.5 * (z[i] + z[i + 1])
            sig_mid = Q_locals[i] @ (T_eps_list[i] @ (eps0 + z_mid * kappa))
            ply_details.append({
                "Pli": f"Pli {i+1}",
                "Matériau": mat["short"],
                "Orientation (°)": int(angles_deg[i]),
                "Épaisseur (mm)": mat["t"],
                "σ1 local / 1 MPa": round(float(sig_mid[0]), 3),
                "σ2 local / 1 MPa": round(float(sig_mid[1]), 3),
                "τ12 local / 1 MPa": round(float(sig_mid[2]), 3),
                "Seuil Hashin Pli (MPa)": round(float(ply_lam), 2),
                "Mode critique": ply_mode,
            })

    fiber_mass = sum(MATERIALS[m]["areal_w"] for m in mat_ids)
    out = {
        "Ex": float(1.0 / resp_x[0]),
        "Ey": float(1.0 / resp_y[1]),
        "Gxy": float(1.0 / resp_s[2]),
        "sigma_fpf": float(sigma_fpf),
        "crit_ply": critical_ply,
        "crit_mode": critical_mode,
        "B_norm": float(np.linalg.norm(B) / (12000.0 * H * H)),
        "A16_norm": float((abs(A[0, 2]) + abs(A[1, 2])) / (A[0, 0] + A[1, 1])),
        "H": float(H),
        "mass_g_m2": float(fiber_mass / 0.35),
        "jute_pct": float(sum(1 for m in mat_ids if m == 0) / n_plies * 100.0),
    }
    if detailed:
        out["ply_details"] = ply_details
    return out


# ==============================================================================
# 3. DÉCODAGE & MÉTAHEURISTIQUES (GA & PSO)
# ==============================================================================
def round5(val):
    a = np.round(np.clip(val, -90.0, 90.0) / 5.0) * 5.0
    return 0.0 if a == -0.0 else float(a)


def decode_solution(x, n_jute, symmetric=True, force_jute_skins=False):
    if symmetric:
        th1, th2, th3 = round5(x[0]), round5(x[1]), round5(x[2])
        angles = [th1, th2, th3, th2, th1]
        configs = SYM_MAP[n_jute]
        if force_jute_skins and n_jute in (2, 3):
            configs = [configs[0]]  # Garde les plis Jute en peau externe
        cfg_idx = int(np.clip(np.floor(x[3] * len(configs)), 0, len(configs) - 1))
        return configs[cfg_idx], angles
    else:
        angles = [round5(v) for v in x[:5]]
        keys = x[5:10]
        mat_ids = [1] * 5
        if n_jute > 0:
            for idx in np.argsort(keys)[:n_jute]:
                mat_ids[idx] = 0
        return mat_ids, angles


def build_objective(ref_metrics, n_jute, symmetric, force_jute_skins,
                    w_Ex, min_Ey_ratio, min_Gxy_ratio):
    Ex_ref, Ey_ref, Gxy_ref, fpf_ref = (
        ref_metrics["Ex"], ref_metrics["Ey"], ref_metrics["Gxy"], ref_metrics["sigma_fpf"]
    )
    w_fpf = 1.0 - w_Ex

    def obj(x):
        mat_ids, angles = decode_solution(x, n_jute, symmetric, force_jute_skins)
        res = evaluate_laminate(mat_ids, angles)
        perf = w_Ex * (res["Ex"] / Ex_ref) + w_fpf * (res["sigma_fpf"] / fpf_ref)
        pen_Ey = max(0.0, (min_Ey_ratio * Ey_ref - res["Ey"]) / Ey_ref)
        pen_Gxy = max(0.0, (min_Gxy_ratio * Gxy_ref - res["Gxy"]) / Gxy_ref)
        pen_B = 0.0 if symmetric else res["B_norm"]
        pen_A16 = res["A16_norm"]
        return -perf + 25.0 * pen_Ey + 25.0 * pen_Gxy + 30.0 * pen_B + 10.0 * pen_A16

    return obj


def run_GA(obj_func, lb, ub, pop_size=80, n_gen=80, pc=0.7, pm=0.2, seed=42):
    rng = np.random.default_rng(seed)
    dim = len(lb)
    pop = rng.uniform(lb, ub, size=(pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    best_idx = int(np.argmin(fitness))
    best_sol, best_fit = pop[best_idx].copy(), float(fitness[best_idx])
    history = [-best_fit]
    mut_scales = np.where((ub - lb) > 10.0, 15.0, 0.25)

    for _ in range(n_gen):
        new_pop = [best_sol.copy()]
        while len(new_pop) < pop_size:
            i1, i2 = rng.integers(0, pop_size, size=2)
            p1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
            j1, j2 = rng.integers(0, pop_size, size=2)
            p2 = pop[j1] if fitness[j1] < fitness[j2] else pop[j2]

            if rng.random() < pc:
                alpha = rng.random(dim)
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = (1.0 - alpha) * p1 + alpha * p2
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                mask = rng.random(dim) < pm
                if np.any(mask):
                    child[mask] += rng.normal(0.0, mut_scales[mask])
                np.clip(child, lb, ub, out=child)
                if len(new_pop) < pop_size:
                    new_pop.append(child)

        pop = np.array(new_pop)
        fitness = np.array([obj_func(ind) for ind in pop])
        idx = int(np.argmin(fitness))
        if fitness[idx] < best_fit:
            best_fit, best_sol = float(fitness[idx]), pop[idx].copy()
        history.append(-best_fit)

    return best_sol, best_fit, np.array(history)


def run_PSO(obj_func, lb, ub, swarm_size=80, n_iter=80, seed=42):
    rng = np.random.default_rng(seed)
    dim = len(lb)
    span = ub - lb
    X = rng.uniform(lb, ub, size=(swarm_size, dim))
    V = rng.uniform(-0.2 * span, 0.2 * span, size=(swarm_size, dim))
    pbest_X = X.copy()
    pbest_fit = np.array([obj_func(ind) for ind in X])
    gbest_idx = int(np.argmin(pbest_fit))
    gbest_X, gbest_fit = pbest_X[gbest_idx].copy(), float(pbest_fit[gbest_idx])
    history = [-gbest_fit]

    for t in range(n_iter):
        w = 0.8 - 0.4 * (t / max(1, n_iter - 1))
        r1, r2 = rng.random((swarm_size, dim)), rng.random((swarm_size, dim))
        V = np.clip(w * V + 1.5 * r1 * (pbest_X - X) + 1.5 * r2 * (gbest_X - X),
                    -0.5 * span, 0.5 * span)
        X = np.clip(X + V, lb, ub)
        fits = np.array([obj_func(ind) for ind in X])
        imp = fits < pbest_fit
        pbest_X[imp], pbest_fit[imp] = X[imp], fits[imp]
        idx = int(np.argmin(pbest_fit))
        if pbest_fit[idx] < gbest_fit:
            gbest_fit, gbest_X = float(pbest_fit[idx]), pbest_X[idx].copy()
        history.append(-gbest_fit)

    return gbest_X, gbest_fit, np.array(history)


# ==============================================================================
# 4. VISUALISATIONS & GÉNÉRATEUR DE SCRIPT ABAQUS
# ==============================================================================
def plot_laminate_schematic(ref_mats, ref_angs, opt_mats, opt_angs):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    titles = ["Stratifié de Référence [0/45/90/-45/0]", "Stratifié Optimisé (EcoComp-Opt)"]
    datasets = [(ref_mats, ref_angs), (opt_mats, opt_angs)]

    for ax, title, (mats, angs) in zip(axes, titles, datasets):
        y_curr = 0.0
        for idx, (m_id, ang) in enumerate(zip(mats, angs)):
            mat = MATERIALS[m_id]
            h_vis = 0.65 if m_id == 0 else 0.42  # Proportion visuelle lisible
            rect = patches.Rectangle(
                (0.0, y_curr), 1.0, h_vis,
                facecolor=mat["color"], edgecolor="white", linewidth=2.0, alpha=0.92
            )
            ax.add_patch(rect)
            ax.text(
                0.5, y_curr + h_vis / 2.0,
                f"Pli {idx+1} : {mat['short']} | {int(ang)}° ({mat['t']:.2f} mm)",
                ha="center", va="center", color="white", fontweight="bold", fontsize=9.5
            )
            y_curr += h_vis

        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_curr + 0.1)
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)

    plt.tight_layout()
    return fig


def generate_abaqus_script(ref_mats, ref_angs, opt_mats, opt_angs, mesh_size=0.5, u_max=5.0):
    def fmt_plies(mats, angs):
        lines = []
        for m_id, ang in zip(mats, angs):
            mat = MATERIALS[m_id]
            lines.append(f"    ('{mat['abaqus']}', {mat['t']:.2f}, {float(ang):.1f}),")
        return "\n".join(lines)

    return f"""# -*- coding: utf-8 -*-
# Script généré automatiquement par EcoComp-Opt (Streamlit DSS)
# Norme ASTM D638 Type 1 | Éléments S4R ({mesh_size} mm) | Critère de Hashin | Sortie tous plis
from __future__ import print_function
from abaqus import *
from abaqusConstants import *
import part, material, section, assembly, step, interaction, load, mesh, job, regionToolset

MESH_SIZE = {mesh_size}
U_MAX = {u_max}

LAYUP_REF = [
{fmt_plies(ref_mats, ref_angs)}
]

LAYUP_OPT = [
{fmt_plies(opt_mats, opt_angs)}
]

def build_composite_model(model_name, layup_plies, mesh_size=0.5, u_max=5.0):
    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    m = mdb.Model(name=model_name)

    x_f1, x_f2 = 32.857625, 132.142375
    s = m.ConstrainedSketch(name='Sketch_ASTM_D638', sheetSize=400.0)
    s.Line(point1=(0.0, 9.5), point2=(x_f1, 9.5))
    s.ArcByCenterEnds(center=(54.0, 82.5), point1=(x_f1, 9.5), point2=(54.0, 6.5), direction=COUNTERCLOCKWISE)
    s.Line(point1=(54.0, 6.5), point2=(111.0, 6.5))
    s.ArcByCenterEnds(center=(111.0, 82.5), point1=(111.0, 6.5), point2=(x_f2, 9.5), direction=COUNTERCLOCKWISE)
    s.Line(point1=(x_f2, 9.5), point2=(165.0, 9.5))
    s.Line(point1=(165.0, 9.5), point2=(165.0, -9.5))
    s.Line(point1=(165.0, -9.5), point2=(x_f2, -9.5))
    s.ArcByCenterEnds(center=(111.0, -82.5), point1=(x_f2, -9.5), point2=(111.0, -6.5), direction=COUNTERCLOCKWISE)
    s.Line(point1=(111.0, -6.5), point2=(54.0, -6.5))
    s.ArcByCenterEnds(center=(54.0, -82.5), point1=(54.0, -6.5), point2=(x_f1, -9.5), direction=COUNTERCLOCKWISE)
    s.Line(point1=(x_f1, -9.5), point2=(0.0, -9.5))
    s.Line(point1=(0.0, -9.5), point2=(0.0, 9.5))

    p = m.Part(name='Specimen', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseShell(sketch=s)

    dp1 = p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=25.0)
    dp2 = p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=140.0)
    p.PartitionFaceByDatumPlane(datumPlane=p.datums[dp1.id], faces=p.faces)
    p.PartitionFaceByDatumPlane(datumPlane=p.datums[dp2.id], faces=p.faces)

    mat_v = m.Material(name='Glass_UD')
    mat_v.Elastic(type=LAMINA, table=((12000.0, 4500.0, 0.3045, 2500.0, 2500.0, 1500.0),))
    mat_v.HashinDamageInitiation(table=((300.0, 150.0, 38.0, 50.0, 82.0, 82.0),))

    mat_j = m.Material(name='Jute_UD')
    mat_j.Elastic(type=LAMINA, table=((500.0, 300.0, 0.30, 100.0, 100.0, 100.0),))
    mat_j.HashinDamageInitiation(table=((50.0, 40.0, 15.0, 10.0, 5.0, 2.0),))

    csys = p.DatumCsysByThreePoints(
        name='CSYS_Layup', coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0), point2=(0.0, 1.0, 0.0)
    )
    region_all = regionToolset.Region(faces=p.faces)
    layup_name = 'Layup_' + model_name
    layup = p.CompositeLayup(
        name=layup_name, description='', elementType=SHELL,
        offsetType=MIDDLE_SURFACE, symmetric=False, thicknessAssignment=FROM_SECTION
    )
    layup.Section(
        preIntegrate=OFF, integrationRule=SIMPSON, thicknessType=UNIFORM,
        poissonDefinition=DEFAULT, temperature=GRADIENT, useDensity=OFF
    )
    layup.ReferenceOrientation(
        orientationType=SYSTEM, localCsys=p.datums[csys.id],
        fieldName='', additionalRotationType=ROTATION_NONE, angle=0.0, axis=AXIS_3
    )

    for idx, (m_name, t_ply, ang) in enumerate(layup_plies):
        layup.CompositePly(
            suppressed=False, plyName='Ply-%d' % (idx + 1), region=region_all,
            material=m_name, thicknessType=SPECIFY_THICKNESS, thickness=float(t_ply),
            orientationType=SPECIFY_ORIENT, orientationValue=float(ang),
            additionalRotationType=ROTATION_NONE, additionalRotationField='',
            axis=AXIS_3, angle=0.0, numIntPoints=3
        )

    a = m.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst = a.Instance(name='Specimen-1', part=p, dependent=ON)

    rp = a.ReferencePoint(point=(165.0, 0.0, 0.0))
    rp_set = a.Set(name='Set_RP_Load', referencePoints=(a.referencePoints[rp.id],))
    set_left = a.Set(name='Set_LeftGrip', faces=inst.faces.findAt(((12.5, 0.0, 0.0),)))
    surf_right = a.Surface(name='Surf_RightGrip', side1Faces=inst.faces.findAt(((152.5, 0.0, 0.0),)))

    m.Coupling(
        name='Constraint_Load', controlPoint=rp_set, surface=surf_right,
        influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, localCsys=None,
        u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON
    )

    m.StaticStep(
        name='Step-Traction', previous='Initial',
        timePeriod=1.0, initialInc=0.05, minInc=1e-5, maxInc=0.05, nlgeom=OFF
    )
    # Sortie complète sur le plan médian de chaque pli (S, E + critères de Hashin)
    m.FieldOutputRequest(
        name='F-Output-Plies', createStepName='Step-Traction',
        variables=('S', 'E', 'HSNFTCRT', 'HSNFCCRT', 'HSNMTCRT', 'HSNMCCRT'),
        layupNames=('Specimen-1.' + layup_name,), layupLocationMethod=SPECIFIED,
        outputAtPlyTop=OFF, outputAtPlyMid=ON, outputAtPlyBottom=OFF, rebar=EXCLUDE
    )
    m.HistoryOutputRequest(
        name='H-Output-Load', createStepName='Step-Traction',
        variables=('U1', 'RF1'), region=rp_set, sectionPoints=DEFAULT, rebar=EXCLUDE
    )

    m.EncastreBC(name='BC_Encastre', createStepName='Initial', region=set_left, localCsys=None)
    m.DisplacementBC(
        name='BC_Traction', createStepName='Step-Traction', region=rp_set,
        u1=float(u_max), u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0,
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None
    )

    p.setMeshControls(regions=p.faces, elemShape=QUAD, technique=FREE, algorithm=ADVANCING_FRONT)
    p.setElementType(
        regions=(p.faces,),
        elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),
                   mesh.ElemType(elemCode=S3, elemLibrary=STANDARD))
    )
    p.seedPart(size=float(mesh_size), deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()
    mdb.Job(name='Job_' + model_name, model=model_name, description='ASTM D638 S4R %.2fmm' % mesh_size)

build_composite_model('HFRC_Ref', LAYUP_REF, mesh_size=MESH_SIZE, u_max=U_MAX)
build_composite_model('HFRC_Opt', LAYUP_OPT, mesh_size=MESH_SIZE, u_max=U_MAX)
if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']
mdb.saveAs(pathName='EcoComp_Opt_Abaqus.cae')
print('Modele EcoComp_Opt_Abaqus.cae pret avec sortie F-Output-Plies sur tous les plis.')
"""


# ==============================================================================
# 5. INTERFACE UTILISATEUR STREAMLIT
# ==============================================================================
st.title("🌿 EcoComp-Opt — Optimisation Métaheuristique d'Éco-Composites")
st.caption(
    "Système d'Aide à la Décision (DSS) couplant Théorie Classique des Stratifiés (CLT), "
    "critère de rupture de Hashin 2D, métaheuristiques (GA / PSO) et export Abaqus S4R."
)

# --- BARRE LATÉRALE : PARAMÉTRAGE ---
with st.sidebar:
    st.header("⚙️ Cahier des charges")
    n_jute = st.slider("Nombre de plis naturels en Jute (sur 5)", 0, 5, 2,
                       help="2 plis = 40 % Jute (configuration HFRC de référence)")
    st.info(f"Composition : **{n_jute*20} % Jute** ({n_jute} plis) / **{(5-n_jute)*20} % Verre** ({5-n_jute} plis)")

    symmetric = st.checkbox("Symétrie stricte (B = 0, zéro gauchissement)", value=True)
    force_jute_skins = st.checkbox("Maintenir le Jute en peaux externes", value=True,
                                   disabled=(not symmetric or n_jute not in (2, 3)),
                                   help="Protège le cœur en verre contre l'humidité et les chocs")

    st.divider()
    st.subheader("🎯 Objectifs & Contraintes")
    w_Ex = st.slider("Poids Rigidité Ex vs Tenue Hashin", 0.0, 1.0, 0.50, 0.05,
                     help="0.50 = équilibre 50 % Rigidité Ex / 50 % Tenue au premier pli")
    min_Ey = st.slider("Maintien minimal Ey (% de la réf.)", 50, 100, 75, 5) / 100.0
    min_Gxy = st.slider("Maintien minimal Gxy (% de la réf.)", 50, 100, 75, 5) / 100.0

    st.divider()
    st.subheader("🧬 Paramètres Métaheuristiques")
    algo_choice = st.radio("Algorithme", ["Comparaison GA vs PSO", "PSO uniquement", "GA uniquement"])
    pop_size = st.slider("Taille population / essaim", 40, 200, 100, 20)
    n_iter = st.slider("Nombre d'itérations / générations", 30, 150, 100, 10)
    n_runs = st.slider("Lancements indépendants (robustesse)", 1, 10, 5, 1)

    launch_opt = st.button("🚀 Lancer l'optimisation", type="primary", use_container_width=True)

# Initialisation du state au premier chargement
ref_mats = SYM_MAP[n_jute][0]
ref_angs = [0.0, 45.0, 90.0, -45.0, 0.0]
ref_res = evaluate_laminate(ref_mats, ref_angs, detailed=True)

if launch_opt or "opt_result" not in st.session_state:
    lb = np.array([-90.0, -90.0, -90.0, 0.0]) if symmetric else np.array([-90.0]*5 + [0.0]*5)
    ub = np.array([ 90.0,  90.0,  90.0, 1.0]) if symmetric else np.array([ 90.0]*5 + [1.0]*5)
    obj = build_objective(ref_res, n_jute, symmetric, force_jute_skins, w_Ex, min_Ey, min_Gxy)

    with st.spinner("Exécution des métaheuristiques en cours..."):
        ga_hists, pso_hists, ga_times, pso_times = [], [], [], []
        best_ga_sol, best_ga_fit = None, 1e9
        best_pso_sol, best_pso_fit = None, 1e9

        for r in range(n_runs):
            if algo_choice in ("Comparaison GA vs PSO", "GA uniquement"):
                t0 = time.perf_counter()
                s_ga, f_ga, h_ga = run_GA(obj, lb, ub, pop_size=pop_size, n_gen=n_iter, seed=100 + r)
                ga_times.append(time.perf_counter() - t0)
                ga_hists.append(h_ga)
                if f_ga < best_ga_fit:
                    best_ga_fit, best_ga_sol = f_ga, s_ga.copy()

            if algo_choice in ("Comparaison GA vs PSO", "PSO uniquement"):
                t0 = time.perf_counter()
                s_pso, f_pso, h_pso = run_PSO(obj, lb, ub, swarm_size=pop_size, n_iter=n_iter, seed=200 + r)
                pso_times.append(time.perf_counter() - t0)
                pso_hists.append(h_pso)
                if f_pso < best_pso_fit:
                    best_pso_fit, best_pso_sol = f_pso, s_pso.copy()

        # Sélection de la meilleure solution globale
        if best_pso_fit <= best_ga_fit:
            winner_algo, winner_sol = "PSO", best_pso_sol
        else:
            winner_algo, winner_sol = "GA", best_ga_sol

        opt_mats, opt_angs = decode_solution(winner_sol, n_jute, symmetric, force_jute_skins)
        opt_res = evaluate_laminate(opt_mats, opt_angs, detailed=True)

        st.session_state["opt_result"] = {
            "ref_mats": ref_mats, "ref_angs": ref_angs, "ref_res": ref_res,
            "opt_mats": opt_mats, "opt_angs": opt_angs, "opt_res": opt_res,
            "winner_algo": winner_algo,
            "ga_hists": np.array(ga_hists) if ga_hists else None,
            "pso_hists": np.array(pso_hists) if pso_hists else None,
            "ga_time": float(np.mean(ga_times)) if ga_times else None,
            "pso_time": float(np.mean(pso_times)) if pso_times else None,
            "n_iter": n_iter, "n_runs": n_runs
        }

res_pack = st.session_state["opt_result"]
r_ref, r_opt = res_pack["ref_res"], res_pack["opt_res"]

# --- ONGLETS PRINCIPAUX ---
tab1, tab2, tab3 = st.tabs([
    "📊 Optimisation & Architecture",
    "🌱 Front de Pareto (Éco-Conception)",
    "🛠️ Validation Abaqus & Export"
])

# ==============================================================================
# ONGLET 1 : OPTIMISATION & ARCHITECTURE
# ==============================================================================
with tab1:
    st.subheader(f"Résultats clés — Meilleur optimum trouvé par {res_pack['winner_algo']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Module longitudinal Ex", f"{r_opt['Ex']:.1f} MPa",
              f"{(r_opt['Ex']/r_ref['Ex'] - 1)*100:+.1f} %")
    c2.metric("Tenue Hashin σ_FPF", f"{r_opt['sigma_fpf']:.2f} MPa",
              f"{(r_opt['sigma_fpf']/r_ref['sigma_fpf'] - 1)*100:+.1f} %")
    c3.metric("Module transversal Ey", f"{r_opt['Ey']:.1f} MPa",
              f"{(r_opt['Ey']/r_ref['Ey'] - 1)*100:+.1f} %")
    c4.metric("Cisaillement Gxy", f"{r_opt['Gxy']:.1f} MPa",
              f"{(r_opt['Gxy']/r_ref['Gxy'] - 1)*100:+.1f} %")
    c5.metric("Couplage B_norm", f"{r_opt['B_norm']:.5f}",
              "Symétrique (B=0)" if r_opt["B_norm"] < 1e-6 else "Asymétrique", delta_color="off")

    st.markdown("#### Schéma comparatif de l'empilement")
    fig_stack = plot_laminate_schematic(
        res_pack["ref_mats"], res_pack["ref_angs"],
        res_pack["opt_mats"], res_pack["opt_angs"]
    )
    st.pyplot(fig_stack)
    plt.close(fig_stack)

    col_left, col_right = st.columns([1.15, 1.0])
    with col_left:
        st.markdown("#### Courbe de convergence métaheuristique")
        fig_conv, ax = plt.subplots(figsize=(6.5, 4.0))
        iters = np.arange(res_pack["n_iter"] + 1)
        if res_pack["ga_hists"] is not None:
            m_ga, s_ga = res_pack["ga_hists"].mean(axis=0), res_pack["ga_hists"].std(axis=0)
            ax.plot(iters, m_ga, color="#1f77b4", lw=2.2,
                    label=f"GA (Moy: {m_ga[-1]:.3f} | {res_pack['ga_time']:.2f}s)")
            if res_pack["n_runs"] > 1:
                ax.fill_between(iters, m_ga - s_ga, m_ga + s_ga, color="#1f77b4", alpha=0.18)
        if res_pack["pso_hists"] is not None:
            m_pso, s_pso = res_pack["pso_hists"].mean(axis=0), res_pack["pso_hists"].std(axis=0)
            ax.plot(iters, m_pso, color="#d62728", lw=2.2,
                    label=f"PSO (Moy: {m_pso[-1]:.3f} | {res_pack['pso_time']:.2f}s)")
            if res_pack["n_runs"] > 1:
                ax.fill_between(iters, m_pso - s_pso, m_pso + s_pso, color="#d62728", alpha=0.18)

        ax.set_xlabel("Génération / Itération")
        ax.set_ylabel("Indice de performance composite")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig_conv)
        plt.close(fig_conv)

    with col_right:
        st.markdown("#### Analyse pli par pli (Stratifié Optimisé)")
        df_plies = pd.DataFrame(r_opt["ply_details"])
        st.dataframe(df_plies, use_container_width=True, hide_index=True)
        st.caption(
            f"Épaisseur totale : **{r_opt['H']:.2f} mm** | "
            f"Masse surfacique estimée : **{r_opt['mass_g_m2']:.0f} g/m²** | "
            f"Pli critique : **Pli {r_opt['crit_ply']} ({r_opt['crit_mode']})**"
        )

# ==============================================================================
# ONGLET 2 : FRONT DE PARETO (ÉCO-CONCEPTION)
# ==============================================================================
with tab2:
    st.subheader("Compromis Performance Mécanique vs Taux de Fibres Naturelles (Jute)")
    st.write(
        "Balayage complet de **0 % à 100 % de Jute** (stratifiés symétriques $B = 0$) "
        "comparant l'empilement standard `[0/45/90/-45/0]` aux séquences optimisées par PSO."
    )

    if st.button("📈 Calculer le Front de Pareto complet (0% à 100% Jute)"):
        rows_pareto = []
        lb_s = np.array([-90.0, -90.0, -90.0, 0.0])
        ub_s = np.array([ 90.0,  90.0,  90.0, 1.0])
        prog = st.progress(0)

        for k in range(6):
            b_mat = SYM_MAP[k][0]
            b_res = evaluate_laminate(b_mat, [0.0, 45.0, 90.0, -45.0, 0.0])
            obj_k = build_objective(b_res, k, True, False, w_Ex, min_Ey, min_Gxy)

            best_sk, best_fk = None, 1e9
            for s in range(3):
                sk, fk, _ = run_PSO(obj_k, lb_s, ub_s, swarm_size=70, n_iter=70, seed=700 + 10*k + s)
                if fk < best_fk:
                    best_fk, best_sk = fk, sk.copy()

            mk, ak = decode_solution(best_sk, k, True, False)
            rk = evaluate_laminate(mk, ak)
            seq_str = " / ".join(f"{MATERIALS[m]['code']}({int(a)}°)" for m, a in zip(mk, ak))
            rows_pareto.append({
                "Taux Jute (%)": k * 20,
                "Ex Réf (GPa)": round(b_res["Ex"] / 1000.0, 2),
                "Ex Opt (GPa)": round(rk["Ex"] / 1000.0, 2),
                "Gain Ex (%)": round((rk["Ex"] / b_res["Ex"] - 1) * 100.0, 1),
                "FPF Réf (MPa)": round(b_res["sigma_fpf"], 1),
                "FPF Opt (MPa)": round(rk["sigma_fpf"], 1),
                "Gain FPF (%)": round((rk["sigma_fpf"] / b_res["sigma_fpf"] - 1) * 100.0, 1),
                "Masse (g/m²)": int(round(rk["mass_g_m2"])),
                "Séquence optimale PSO": f"[{seq_str}]"
            })
            prog.progress(int((k + 1) / 6 * 100))

        st.session_state["pareto_df"] = pd.DataFrame(rows_pareto)

    if "pareto_df" in st.session_state:
        df_p = st.session_state["pareto_df"]
        fig_p, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

        ax1.plot(df_p["Taux Jute (%)"], df_p["Ex Réf (GPa)"], "o--", color="#7f7f7f", lw=1.8, label="Réf. [0/45/90/-45/0]")
        ax1.plot(df_p["Taux Jute (%)"], df_p["Ex Opt (GPa)"], "s-", color="#2ca02c", lw=2.3, label="Optimisé PSO (B=0)")
        ax1.set_xlabel("Proportion de plis naturels en Jute (%)")
        ax1.set_ylabel("Module longitudinal Ex (GPa)")
        ax1.set_title("Rigidité longitudinale vs Taux de Jute")
        ax1.set_xticks(df_p["Taux Jute (%)"])
        ax1.grid(True, linestyle="--", alpha=0.45)
        ax1.legend()

        ax2.plot(df_p["Taux Jute (%)"], df_p["FPF Réf (MPa)"], "o--", color="#7f7f7f", lw=1.8, label="Réf. [0/45/90/-45/0]")
        ax2.plot(df_p["Taux Jute (%)"], df_p["FPF Opt (MPa)"], "s-", color="#d66127", lw=2.3, label="Optimisé PSO (B=0)")
        ax2.set_xlabel("Proportion de plis naturels en Jute (%)")
        ax2.set_ylabel("Contrainte initiation Hashin (MPa)")
        ax2.set_title("Tenue au premier pli vs Taux de Jute")
        ax2.set_xticks(df_p["Taux Jute (%)"])
        ax2.grid(True, linestyle="--", alpha=0.45)
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)
        st.dataframe(df_p, use_container_width=True, hide_index=True)

# ==============================================================================
# ONGLET 3 : VALIDATION ABAQUS & EXPORT
# ==============================================================================
with tab3:
    st.subheader("Pont automatique vers Abaqus/CAE (S4R 0,5 mm — ASTM D638)")
    c_exp1, c_exp2 = st.columns(2)

    abaqus_code = generate_abaqus_script(
        res_pack["ref_mats"], res_pack["ref_angs"],
        res_pack["opt_mats"], res_pack["opt_angs"]
    )
    csv_summary = pd.DataFrame(r_opt["ply_details"]).to_csv(index=False, sep=";").encode("utf-8-sig")

    with c_exp1:
        st.download_button(
            label="📥 Télécharger le script Abaqus (.py)",
            data=abaqus_code.encode("utf-8"),
            file_name="simulation_abaqus_ecocomp_opt.py",
            mime="text/x-python",
            type="primary",
            use_container_width=True
        )
    with c_exp2:
        st.download_button(
            label="📄 Télécharger le tableau des plis (.csv)",
            data=csv_summary,
            file_name="recap_stratifies_ecocomp.csv",
            mime="text/csv",
            use_container_width=True
        )

    with st.expander("Voir l'aperçu du script Python Abaqus généré"):
        st.code(abaqus_code, language="python")

    st.divider()
    st.subheader("Validation croisée avec les courbes Abaqus (abaqus_curves.csv)")
    csv_path = "abaqus_curves.csv"
    if os.path.exists(csv_path):
        eps_r, sig_r, eps_o, sig_o = [], [], [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) >= 4:
                    eps_r.append(float(row[0]))
                    sig_r.append(float(row[1]))
                    eps_o.append(float(row[2]))
                    sig_o.append(float(row[3]))

        fig_abq, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.plot(eps_r, sig_r, color="#555555", lw=2.2, label="Abaqus HFRC Réf. (E = 2173 MPa | FPF = 16,63 MPa)")
        ax.plot(eps_o, sig_o, color="#d62728", lw=2.2, label="Abaqus HFRC Opt. (E = 3095 MPa | FPF = 30,55 MPa)")
        ax.scatter([0.77], [16.63], color="#555555", s=80, zorder=5, edgecolor="black")
        ax.scatter([0.99], [30.55], color="#d62728", s=85, zorder=5, marker="s", edgecolor="black")
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 65)
        ax.set_xlabel("Déformation axiale globale (%)")
        ax.set_ylabel("Contrainte nominale (MPa)")
        ax.set_title("Courbes extraites d'Abaqus/Standard (S4R 0,5 mm — ASTM D638)")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig_abq)
        plt.close(fig_abq)
        st.success("Fichier `abaqus_curves.csv` détecté et chargé automatiquement.")
    else:
        st.info("Placez `abaqus_curves.csv` dans le même dossier pour afficher ici la courbe de validation Abaqus.")