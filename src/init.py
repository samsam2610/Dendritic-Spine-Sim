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

def interpolate_pt3d(sec, x_norm):
    """Return interpolated (x, y, z) position at normalized location x in sec"""
    n3d = int(h.n3d(sec=sec))
    if n3d < 2:
        raise ValueError("Section has fewer than 2 pt3d points")

    arc_len_total = h.arc3d(n3d-1, sec=sec)
    target_arc = x_norm * arc_len_total

    for i in range(1, n3d):
        arc_i = h.arc3d(i, sec=sec)
        arc_prev = h.arc3d(i-1, sec=sec)
        if arc_i >= target_arc:
            # Interpolate between pt3d(i-1) and pt3d(i)
            frac = (target_arc - arc_prev) / (arc_i - arc_prev + 1e-12)
            x0, y0, z0 = h.x3d(i-1, sec=sec), h.y3d(i-1, sec=sec), h.z3d(i-1, sec=sec)
            x1, y1, z1 = h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)
            x3d = x0 + frac * (x1 - x0)
            y3d = y0 + frac * (y1 - y0)
            z3d = z0 + frac * (z1 - z0)
            return x3d, y3d, z3d

    # Fallback if loop fails
    return h.x3d(n3d-1, sec=sec), h.y3d(n3d-1, sec=sec), h.z3d(n3d-1, sec=sec)


def add_spines_to_PT5B_cells():
    spine_idx = 0
    offset_distance = 0.5  # microns; radial offset from dendrite

    for cell in sim.net.cells:
        if cell.tags.get('cellType') == 'PT' and cell.tags.get('cellModel') == 'HH_full':
            for secName in list(cell.secs.keys()):
                if 'apic' in secName or 'dend' in secName:
                    parent = cell.secs[secName]['hObj']
                    for x in [i / 20 for i in range(1, 20)]:  # 0.05 to 0.95
                        # Get 3D coordinates at position x along the dendrite
                        x3d, y3d, z3d = interpolate_pt3d(parent, x)

                        # Offset vector (you can randomize angle if you want)
                        angle = 2 * math.pi * (spine_idx % 10) / 10  # simple rotation
                        dx = offset_distance * math.cos(angle)
                        dy = offset_distance * math.sin(angle)
                        dz = 0

                        # Create and connect the spine neck
                        neck = h.Section(name=f'spine_neck_{spine_idx}')
                        neck.L = 1.5
                        neck.diam = 0.2
                        neck.insert('pas')
                        neck.g_pas = 0.001
                        neck.e_pas = -65
                        neck.connect(parent(x))

                        h.pt3dclear(sec=neck)
                        h.pt3dadd(x3d, y3d, z3d, neck.diam, sec=neck)
                        h.pt3dadd(x3d + dx, y3d + dy, z3d + dz, neck.diam, sec=neck)

                        # Register with NetPyNE so it shows in plotShape
                        cell.secs[f'spine_neck_{spine_idx}'] = {
                            'hObj': neck,
                            'geom': {'L': neck.L, 'diam': neck.diam},
                            'topol': {'parentSec': secName, 'parentX': x, 'childX': 0.0},
                            'mechs': {'pas': {'g': 0.001, 'e': -65}},
                        }

                        spine_idx += 1



if cfg.useExplicitSpines:
    add_spines_to_PT5B_cells()

# Apply any custom logic like tDCS (already added in cfg.afterSim)
sim.simulate()

# 🔥 Remove non-serializable objects before saving
if hasattr(cfg, 'afterSim'):
    del cfg.afterSim

sim.analyze()

# Remove afterSim from cfg before saving (avoid JSON error)
if hasattr(cfg, 'afterSim'):
    del cfg.afterSim
