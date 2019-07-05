import os, sys
import numpy as np
import pandas as pd
import datetime as dt
from scipy.io import loadmat

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
mat_dir = os.path.join(proj_dir, 'mat')

mouse_names = np.array(['Krebs', 'Waksman', 'Robbins'])
spon_start_times = np.array([3811, 3633, 3323])

regions = np.array(['FrCtx','FrMoCtx','SomMoCtx','SSCtx','V1','V2','RSP','CP','LS','LH','HPF','TH','SC','MB'])

def loadSpikeForMouse(mouse_name):
    # write this function
    return 0
