"""
init.py

Starting script to run NetPyNE-based M1 model.

Usage:
    python src/init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 4 nrniv -python -mpi src/init.py
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers

from netpyne import sim
from neuron import h
from cfg import cfg

cfg, netParams = sim.loadFromIndexFile('index.npjson')

# schedule application at runtime
if not hasattr(cfg, 'afterSim'): cfg.afterSim = []
def apply_tdcs_field():
    amp = cfg.tDCS['amp']
    ynormRange = cfg.tDCS.get('ynormRange', [0.0, 1.0])

    for cell in sim.net.cells:
        ynorm = cell.tags.get('ynorm', 0.5)
        if not (ynormRange[0] <= ynorm <= ynormRange[1]):
            continue
        for secName, sec in cell.secs.items():
            hSec = sec['hObj']
            if 'extracellular' not in sec['mechs']:
                sec['mechs']['extracellular'] = {}
            for seg in hSec:
                seg.e_extracellular = amp * ynorm

# Attach to afterSim
if not hasattr(cfg, 'afterSim'):
    cfg.afterSim = []
cfg.afterSim.append(apply_tdcs_field)

sim.create(netParams, cfg)
# 🔍 Check for spine initialization in PT cells by inspecting cell.secs
pt_cells = [cell for cell in sim.net.cells if cell.tags.get('cellType') == 'PT']
if not pt_cells:
    print("⚠️  No PT cells found in network.")
else:
    for idx, cell in enumerate(pt_cells):
        # Count sections named 'spine_neck' (one per spine)
        spine_neck_secs = [secName for secName in cell.secs if secName.startswith('spine_neck')]
        n_spines = len(spine_neck_secs)
        if n_spines > 0:
            print(f"✅ PT cell {idx}: {n_spines} spines initialized ({len(spine_neck_secs)} neck sections).")
        else:
            print(f"❌ PT cell {idx}: No spines found in cell.secs")

from neuron import h
from math import cos, sin, pi

def add_spine_at(cell, parent_sec, x, spine_idx, spine_length=1.5, offset=0.5):
    """Attach a spine at location `x` along parent_sec with proper 3D orientation."""
    import math
    from neuron import h
    import numpy as np

    # Interpolate 3D coordinates and tangent vector
    x0, y0, z0 = interpolate_pt3d(parent_sec, x)
    x1 = min(x + 0.01, 1.0)
    x1_pos = interpolate_pt3d(parent_sec, x1)
    tangent = np.array([x1_pos[0] - x0, x1_pos[1] - y0, x1_pos[2] - z0])
    tangent_norm = np.linalg.norm(tangent)

    if tangent_norm < 1e-6:
        tangent = np.array([1, 0, 0])
    else:
        tangent = tangent / tangent_norm

    # Use a fixed up vector and take cross product for normal
    up = np.array([0, 0, 1]) if abs(tangent[2]) < 0.99 else np.array([0, 1, 0])
    normal = np.cross(tangent, up)
    normal /= np.linalg.norm(normal)

    # Create neck section
    neck = h.Section(name=f"spine_neck_{spine_idx}")
    neck.L = spine_length
    neck.diam = 0.2
    neck.insert("pas")
    neck.g_pas = 0.001
    neck.e_pas = -65
    neck.connect(parent_sec(x), 0.0)

    # Apply 3D coordinates to visualize properly
    h.pt3dclear(sec=neck)
    h.pt3dadd(x0, y0, z0, neck.diam, sec=neck)
    h.pt3dadd(x0 + offset * normal[0],
              y0 + offset * normal[1],
              z0 + offset * normal[2], neck.diam, sec=neck)

    # Register for NetPyNE visualization
    cell.secs[f'spine_neck_{spine_idx}'] = {
        'hObj': neck,
        'geom': {'L': neck.L, 'diam': neck.diam},
        'topol': {'parentSec': parent_sec.name(), 'parentX': x, 'childX': 0.0},
        'mechs': {'pas': {'g': 0.001, 'e': -65}},
        'pt3d': [
            [x0, y0, z0, neck.diam],
            [x0 + offset * normal[0], y0 + offset * normal[1], z0 + offset * normal[2], neck.diam]
        ],
        'color': 'red'
    }

    return neck




def add_spines_to_PT5B_cells():
    spine_idx = 0
    offset_distance = 0.5  # microns; radial offset from dendrite
    for cell in sim.net.cells:
        if cell.tags.get('cellType') == 'PT' and cell.tags.get('cellModel') == 'HH_full':
            for secName in list(cell.secs.keys()):
                if 'apic' in secName or 'dend' in secName:
                    parent = cell.secs[secName]['hObj']
                    for x in [i / 20 for i in range(1, 20)]:  # 0.05 to 0.95
                        add_spine_at(cell, parent, x, spine_idx)
                        spine_idx += 1



if cfg.useExplicitSpines:
    add_spines_to_PT5B_cells()

for cell in sim.net.cells:
    if cell.tags.get('cellType') == 'PT':
        for secName, sec in cell.secs.items():
            if secName.startswith('spine_neck'):
                topol = sec.get('topol', {})
                parent_name = topol.get('parentSec', 'unknown')
                parent_x = topol.get('parentX', 'unknown')
                print(f"{secName} connected to: {parent_name} at x={parent_x}")

def get_3d_path(sec):
    return [(h.x3d(i), h.y3d(i), h.z3d(i)) for i in range(int(h.n3d()))]

for cell in sim.net.cells:
    if cell.tags.get('cellType') == 'PT':
        for secName in cell.secs:
            if secName.startswith('spine_neck'):
                pt3d = get_3d_path(cell.secs[secName]['hObj'])
                print(f"{secName} 3D path: {pt3d}")

# Apply any custom logic like tDCS (already added in cfg.afterSim)
sim.simulate()

# 🔥 Remove non-serializable objects before saving
if hasattr(cfg, 'afterSim'):
    del cfg.afterSim

sim.analyze()

# Remove afterSim from cfg before saving (avoid JSON error)
if hasattr(cfg, 'afterSim'):
    del cfg.afterSim
