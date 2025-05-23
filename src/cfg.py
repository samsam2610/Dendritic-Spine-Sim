"""
cfg.py

Simulation configuration for M1 model (using NetPyNE)
"""

from netpyne import specs
import pickle
import os
from datetime import datetime
cfg = specs.SimConfig()

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.duration = 1*1e3
cfg.dt = 0.05
cfg.seeds = {'conn': 4321, 'stim': 1234, 'loc': 4321}
cfg.hParams = {'celsius': 34, 'v_init': -80}
cfg.coreneuron = False # run via coreneuron (alpha support)
cfg.gpu = False # needs coreneuron enabled
cfg.random123 = False # enable random123 for coreneuron compatibility
cfg.verbose = 1
cfg.createNEURONObj = True
cfg.createPyStruct = True
cfg.cvode_active = False
cfg.cvode_atol = 1e-3
cfg.cache_efficient = True
cfg.printRunTime = 0.1

cfg.includeParamsLabel = False
cfg.printPopAvgRates = True

cfg.checkErrors = False
cfg.saveInterval = 100 # define how often the data is saved, this can be used with interval run if you want to update the weights more often than you save
cfg.intervalFolder = 'interval_saving'

#------------------------------------------------------------------------------
# Recording
#------------------------------------------------------------------------------
allpops = ['IT2','PV2','SOM2','IT4','IT5A','PV5A','SOM5A','IT5B','PT5B','PV5B','SOM5B','IT6','CT6','PV6','SOM6']

cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc':0.5, 'var':'v'}}

cfg.recordStim = True
cfg.recordTime = False
cfg.recordStep = 0.1

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------
cfg.simLabel = 'DendriticSpinySim'
base_folder = '.'  # or another path if you want

# Generate a folder name like: ./DendriticSpinySim_2024-05-19_15-30-02
dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_folder = os.path.join(base_folder, f"{cfg.simLabel}_{dt_str}")

# Create the directory if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Assign to cfg.saveFolder so NetPyNE saves there
cfg.saveFolder = save_folder
cfg.savePickle = False
cfg.saveJson = True
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']
cfg.backupCfgFile = None #['cfg.py', 'backupcfg/']
cfg.gatherOnlySimData = False
cfg.saveCellSecs = True
cfg.saveCellConns = True

#------------------------------------------------------------------------------
# Analysis and plotting
#------------------------------------------------------------------------------
with open('cells/popColors.pkl', 'rb') as fileObj: popColors = pickle.load(fileObj)['popColors']
cfg.analysis['plotTraces'] = {'include': [('IT5A',0), ('PT5B',00)], 'timeRange': [0,500], 'oneFigPer': 'cell', 'figSize': (10,4), 'saveFig': True, 'showFig': False}

cfg.analysis['plotRaster'] = {'include': allpops, 'saveFig': True, 'showFig': False, 'popRates': True, 'orderInverse': True,
							'timeRange': [0,500], 'popColors': popColors, 'figSize': (10,10), 'lw': 0.3, 'markerSize':10, 'marker': '.', 'dpi': 600}

cfg.analysis['plotSpikeHist'] = {'include': ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6'], 'timeRange': [0,500],
								'saveFig': True, 'showFig': False, 'popColors': popColors, 'figSize': (10,10), 'dpi': 600}

cfg.analysis['plotConn'] = {'includePre': ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6'], 'includePost': ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6'], 'feature': 'strength', 'figSize': (10,10), 'groupBy': 'pop', \
 						'graphType': 'matrix', 'synOrConn': 'conn', 'synMech': None, 'saveData': None, 'saveFig': 1, 'showFig': 0}
cfg.analysis['plotShape'] = {
    'includePre': ['PT5B'], 
    'includePost': ['PT5B'], 
    'figSize': (10, 10), 
    'dpi': 600, 
    'saveFig': True,     
    'shapeParams': {
        'cvals': 'spine',       # Color sections where cell.secs[secName]['spine']==True
        'clim': [0, 1],         # 0: not spine, 1: spine
        'colormap': 'viridis',  # or any
        'sectionList': None, 
	}
}

#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------
cfg.cellmod =  {'IT2': 'HH_reduced',
				'IT4': 'HH_reduced',
				'IT5A': 'HH_reduced',
				'IT5B': 'HH_reduced',
				'PT5B': 'HH_full',
				'IT6': 'HH_reduced',
				'CT6': 'HH_reduced'}

cfg.ihModel = 'migliore'  # ih model
cfg.ihGbar = 1.0  # multiplicative factor for ih gbar in PT cells
cfg.ihGbarZD = None # multiplicative factor for ih gbar in PT cells
cfg.ihGbarBasal = 1.0 # 0.1 # multiplicative factor for ih gbar in PT cells
cfg.ihlkc = 0.2 # ih leak param (used in Migliore)
cfg.ihlkcBasal = 1.0
cfg.ihlkcBelowSoma = 0.01
cfg.ihlke = -86  # ih leak param (used in Migliore)
cfg.ihSlope = 14*2

cfg.removeNa = False  # simulate TTX; set gnabar=0s
cfg.somaNa = 5
cfg.dendNa = 0.3
cfg.axonNa = 7
cfg.axonRa = 0.005

cfg.gpas = 0.5  # multiplicative factor for pas g in PT cells
cfg.epas = 0.9  # multiplicative factor for pas e in PT cells

#------------------------------------------------------------------------------
# Synapses
#------------------------------------------------------------------------------
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionSOME = [0.9, 0.1] # SOM -> E GABAASlow to GABAB ratio

cfg.synsperconn = {'HH_full': 5, 'HH_reduced': 1, 'HH_simple': 1}
cfg.AMPATau2Factor = 1.0

#------------------------------------------------------------------------------
# Network
#------------------------------------------------------------------------------
cfg.singleCellPops = 0  # Create pops with 1 single cell (to debug)
cfg.weightNormThreshold = 4.0  # weight normalization factor threshold

cfg.addConn = 1
cfg.scale = 1.0
cfg.sizeY = 1350.0
cfg.sizeX = 400.0
cfg.sizeZ = 400.0
cfg.scaleDensity = 0.01

cfg.EEGain = 1.0
cfg.EIGain = 1.0
cfg.IEGain = 1.0
cfg.IIGain = 1.0

cfg.IEdisynapticBias = None  # increase prob of I->Ey conns if Ex->I and Ex->Ey exist

#------------------------------------------------------------------------------
## E->I gains
cfg.EPVGain = 1.0
cfg.ESOMGain = 1.0

#------------------------------------------------------------------------------
## I->E gains
cfg.PVEGain = 1.0
cfg.SOMEGain = 1.0

#------------------------------------------------------------------------------
## I->I gains
cfg.PVSOMGain = None #0.25
cfg.SOMPVGain = None #0.25
cfg.PVPVGain = None # 0.75
cfg.SOMSOMGain = None #0.75

#------------------------------------------------------------------------------
## I->E/I layer weights (L2/3+4, L5, L6)
cfg.IEweights = [1.2, 1.2, 1.2]
cfg.IIweights = [0.8, 1.2, 0.8]

cfg.IPTGain = 1.0
cfg.IFullGain = 1.0

#------------------------------------------------------------------------------
# Subcellular distribution
#------------------------------------------------------------------------------
cfg.addSubConn = 1

#------------------------------------------------------------------------------
# Long range inputs
#------------------------------------------------------------------------------
cfg.addLongConn = 1
cfg.numCellsLong = 10  # num of cells per population
cfg.noiseLong = 1.0  # firing rate random noise
cfg.delayLong = 5.0  # (ms)
cfg.weightLong = 0.5*2  # corresponds to unitary connection somatic EPSP (mV)
cfg.startLong = 0  # start at 0 ms
cfg.ratesLong = {'TPO': [0,20], 'TVL': [0,20], 'S1': [0,20], 'S2': [0,20], 'cM1': [0,20], 'M2': [0,20], 'OC': [0,20]}

## input pulses
cfg.addPulses = 1
cfg.pulse = {'pop': 'None', 'start': 1000, 'end': 1200, 'rate': 20, 'noise': 0.5}
#cfg.pulse2 = {'pop': 'None', 'start': 1000, 'end': 1200, 'rate': 20, 'noise': 0.5}


#------------------------------------------------------------------------------
# Current inputs
#------------------------------------------------------------------------------
cfg.addIClamp = 0
cfg.IClamp1 = {'pop': 'PT5B', 'sec': 'soma', 'loc': 0.5, 'start': 0, 'dur': 2000, 'amp': 6}

cfg.addTDCS = 1
cfg.tDCS = {
    'amp': 6e-6 / (0.3 * (3e-3 * 3e-3)),  # convert to field strength (V/m)
    'dir': [0, 1, 0],  # y-axis
    'ynormRange': [0.5, 1.0]  # apply to deep layers only 
}


#------------------------------------------------------------------------------
# NetStim inputs
#------------------------------------------------------------------------------
cfg.addNetStim = 0

cfg.NetStim1 = {'pop': 'IT2', 'ynorm':[0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0],
				'start': 500, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 'weight': 30.0, 'delay': 0}


#------------------------------------------------------------------------------
# SpinyCell configuration and spine plasticity
#------------------------------------------------------------------------------

# Whether to explicitly attach SpinyCell compartments to dendrites
cfg.useExplicitSpines = True

# Control dynamic plasticity of spine necks
cfg.useSpinePlasticity = True

# Spine plasticity update frequency (in ms)
cfg.spinePlasticityInterval = 5.0

# Voltage threshold (mV) above which spine neck elongates
cfg.spinePlasticityVThreshold = -40.0

# Rate of change in spine neck length (μm)
cfg.spineNeckGrowStep = 0.05  # if v > threshold
cfg.spineNeckShrinkStep = 0.01  # if v <= threshold

# Clamp bounds on spine neck length
cfg.spineNeckMinL = 0.3
cfg.spineNeckMaxL = 5.0