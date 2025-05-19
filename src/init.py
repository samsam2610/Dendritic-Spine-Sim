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

def add_spines_to_PT5B_cells():
    from neuron import h
    spine_idx = 0
    for cell in sim.net.cells:
        if cell.tags.get('cellType') == 'PT' and cell.tags.get('cellModel') == 'HH_full':
            for secName in cell.secs:
                if 'apic' in secName or 'dend' in secName:  # only dendritic
                    parent = cell.secs[secName]['hObj']
                    for x in [i / 20 for i in range(1, 20)]:  # 0.05 to 0.95
                        neck = h.Section(name=f'spine_neck_{spine_idx}')
                        neck.L = 1.5
                        neck.diam = 0.2
                        neck.insert('pas')
                        neck.g_pas = 0.001
                        neck.e_pas = -65
                        neck.connect(parent(x))

                        # Register the spine in NetPyNE so plotShape can see it
                        cell.secs[f'spine_neck_{spine_idx}'] = {
                            'hObj': neck,
                            'geom': {'L': 1.5, 'diam': 0.2},
                            'topol': {'parentSec': secName, 'parentX': x, 'childX': 0.0},
                            'mechs': {'pas': {'g': 0.001, 'e': -65}}
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
