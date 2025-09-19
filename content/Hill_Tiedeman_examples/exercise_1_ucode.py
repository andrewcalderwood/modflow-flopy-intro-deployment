# standard python utilities
import pandas as pd
import numpy as np
import os 
import sys
from os.path import join, exists, basename, dirname, expanduser

## Set up directory referencing
usr_dir = expanduser("~")
# Package data
git_dir = join(usr_dir, 'Documents','GitHub')
    
## to use a GitHub version of flopy insert the filepath at position 0 to be found before the conda version
def add_path(dir_path):
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

add_path(git_dir+'/flopy/')
import flopy


# save modflow workspace file to WRDAPP sub folder to improve runtime calculations
loadpth = 'C:/WRDAPP/GWFlowModel/training'
# load model with only DIS to reduce load time
# the model will run off of the .nam file connection so flopy doesn't need them
model_ws = join(loadpth, 'exercise_1') 
#model_ws = os.getcwd() # depends on script location
m = flopy.modflow.Modflow.load('MF.nam', model_ws=model_ws, 
                                exe_name='mf2005', version='mf2005')

# parameters
param = pd.read_csv(join(model_ws, 'exercise1_hydraulic_parameters_ucode.csv'))

nlay = m.dis.nlay
nrow= m.dis.nrow
ncol = m.dis.ncol

###############################################################################
## LPF Package ##

# allocate arrays for hydraulic conductivity (HK), vertical HK, specific storage (Ss), specific yield (Sy)
hk = np.zeros(m.dis.botm.shape)
vka = np.zeros(m.dis.botm.shape)

# define vertical confining bed conductivity
vkcb = np.zeros(m.dis.botm.shape)

for k in np.arange(0,nlay):
    param_k = param[param.layer==k+1]
    hk[k] = param_k.loc[param_k.GroupName=='hk','StartValue']
    vka[k] = param_k.loc[param_k.GroupName=='vani','StartValue']
    vkcb[k] = param_k.loc[param_k.GroupName=='vkcb','StartValue']

# VKA is defined where: 0 is vertical hydraulic conductivity, and 1 is vertical anisotropy
layvka = 1
# chani is defined where: 0 means HANI defines HANI and >0  means CHANI defines HANI for entire layers
chani = 0
# hani is the ratio of horizontal anisotropy (along columns vs rows (x vs. y))
hani = 1
# LAYTYP MUST BE GREATER THAN ZERO WHEN IUZFOPT IS 2
# 0 is confined, >0 convertible, <0 convertible unless the THICKSTRT option is in effect
laytyp = np.zeros(m.dis.nlay)
laywet = laytyp[:]
# Laywet must be 0 if laytyp is confined laywet = [1,1,1,1,1]
# laywet = 1 means layers can be rewetted.
#ipakcb = 53 means cell-by-cell budget is saved because it is non zero (default is 53)
lpf = flopy.modflow.ModflowLpf(model = m, 
                               hk =hk, chani = chani, hani = hani,
                               layvka = layvka, vka = vka, 
                               vkcb = vkcb,
                               laytyp=laytyp, laywet = laywet, ipakcb=53,
                              )

# multipliers are not directly defineable within flopy, but
# it is easily applicable with an array to mulitply the value
mult_1d = np.linspace(1, 9, ncol)
# the array must be reshaped to have a 2nd axis to then repeat along that axis
mult_2d = np.repeat(np.reshape(mult_1d, (1, ncol)), nrow, axis=0)

# we can now scale the HK in layer 2 by the multiplier array
hk_mult = np.copy(hk)
hk_mult[1] = hk[1] * mult_2d

# this can then be redefined in the LPF package
lpf.hk = hk_mult

# write output
lpf.write_file()

###############################################################################
## RIV Package ##
riv = m.riv
# the riverbed conductance is calculated with the dimensions of the model and assigned K value
K = param.loc[param.ParamName=='k_rb','StartValue'].values[0] # m/s)
L = 1000 # length of the river in each cell (m)
W = 10 # river width (m)
M = 10 # river bed thickness (m)
# calculate the conductance of the river (values that are not affected by internal head calculations)
C = K*L*W/M # conductance (m^2/s)

# assign the stress_period data to the flopy RIV object
riv_dict = {}
for n in np.arange(0, m.dis.nper):
    riv_dict[n] = riv.stress_period_data[n]
    riv_dict[n].cond = C

# re-assign dictionary
riv.stress_period_data = riv_dict

riv.write_file()

###############################################################################
## GHB Package ##

ghb = m.ghb
# the GHB conductance is calculated with the dimensions of the model and assigned K value
K = param.loc[param.ParamName=='hk_hillside','StartValue'].values[0]
L = 1000 # length of the ghb in each cell (m)
W = 50 # vertical thickness for each layer (m)
M = 1000 # low permeability granite thickness (m)
# calculate the conductance of the leaky reservoir (values that are not affected by internal head calculations)
C = K*L*W/M # conductance (m^2/s)

# assign the stress_period data to the flopy RIV object
ghb_dict = {}
for n in np.arange(0, m.dis.nper):
    ghb_dict[n] = ghb.stress_period_data[n]
    ghb_dict[n].cond = C

# re-assign dictionary
ghb.stress_period_data = ghb_dict

###############################################################################
# run the modflow model
success, buff = m.run_model()